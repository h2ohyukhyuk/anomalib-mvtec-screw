"""Qt image template editor.

Run: python make_template.py [image_path]
Dependencies: numpy, opencv-python, PySide6-Essentials
Coordinates use pixel centers: x grows right, y grows down, center x=(width-1)/2.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Linux opencv-python wheels can set paths to their own incompatible Qt plugins.
# This program uses PySide6 for the GUI, so discard only paths injected by cv2.
for _key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "/cv2/" in os.environ.get(_key, "").replace("\\", "/"):
        os.environ.pop(_key, None)

from PySide6.QtCore import Qt, QTimer, Signal, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QGraphicsScene,
    QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

MAX_IMAGE_PIXELS = 32_000_000
POINT_COLORS = ("#ff6666", "#55c9ff")


def manual_transform(width, height, angle, dx, dy):
    """Original -> working image: rotation about center, then translation."""
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[:, 2] += (dx, dy)
    return matrix


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ matrix[:, :2].T + matrix[:, 2]


def similarity_transform(source, target):
    """Unique orientation-preserving rotation + uniform scale + translation."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    vector_src = source[1] - source[0]
    vector_dst = target[1] - target[0]
    norm_sq = float(vector_src @ vector_src)
    if norm_sq < 1e-12:
        raise ValueError("입력 P1과 P2의 높이를 서로 다르게 설정하세요.")
    if float(vector_dst @ vector_dst) < 1e-12:
        raise ValueError("목적 P1과 P2를 서로 다른 좌표로 설정하세요.")
    a = float(vector_src @ vector_dst) / norm_sq
    b = float(vector_src[0] * vector_dst[1] - vector_src[1] * vector_dst[0]) / norm_sq
    linear = np.array([[a, -b], [b, a]], dtype=np.float64)
    offset = target[0] - linear @ source[0]
    return np.column_stack((linear, offset))


def comparison_image(working):
    """Original left half + absolute mirror difference on the right half."""
    difference = cv2.absdiff(working, cv2.flip(working, 1))
    width = working.shape[1]
    second = working.copy()
    second[:, (width + 1) // 2:] = difference[:, (width + 1) // 2:]
    return second


def warp_zero(image, matrix, width, height):
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )


def double_spin(minimum, maximum, value=0, step=0.1):
    spin = QDoubleSpinBox()
    spin.setDecimals(3)
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setMinimumWidth(100)
    return spin


class StepControl(QWidget):
    """One editable value with independent +/-10, +/-1, +/-0.1 controls."""
    valueChanged = Signal(float)

    def __init__(self, unit, parent=None, spin=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.spin = spin if spin is not None else double_spin(-100000, 100000)
        self.spin.setSuffix(f" {unit}")
        layout.addWidget(self.spin)
        for step in (10, 1, 0.1):
            pair = QVBoxLayout()
            pair.setSpacing(2)
            for sign, text in ((1, "+"), (-1, "−")):
                button = QPushButton(f"{text}{step:g}")
                button.setMinimumWidth(46)
                button.setAutoRepeat(True)
                button.setToolTip(f"{text}{step:g} {unit}")
                button.clicked.connect(
                    lambda checked=False, delta=sign * step:
                    self.spin.setValue(self.spin.value() + delta)
                )
                pair.addWidget(button)
            layout.addLayout(pair)
        self.spin.valueChanged.connect(self.valueChanged.emit)

    def value(self):
        return self.spin.value()

    def setValue(self, value):
        self.spin.setValue(value)


def labeled_input(text, control):
    """Keep a visible label directly beside its numeric input."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    label = QLabel(text)
    label.setProperty("inputLabel", True)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
    label.setBuddy(control.spin if isinstance(control, StepControl) else control)
    layout.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(control, 1)
    return row


class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.focus_point = None
        self.focus_width = None
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#0a0d12"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumSize(180, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image, canvas_size, points=None, center_line=False,
                  focus_index=None):
        self.focus_point = None
        self.focus_width = None
        if focus_index is not None and image is not None and points is not None:
            self.focus_point = np.asarray(points[focus_index], dtype=np.float64) + 0.5
            # Show one quarter of the working image width around the selected point.
            self.focus_width = max(1.0, image.shape[1] / 4)
        scene = self.scene()
        scene.clear()
        canvas_w, canvas_h = canvas_size
        scene.setSceneRect(0, 0, canvas_w, canvas_h)
        # Full-image views share a common scene extent; detail views crop below.
        # Unoccupied areas are black zero padding; images are never stretched.
        scene.setBackgroundBrush(QColor("black"))
        if image is not None:
            height, width = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb.data, width, height, rgb.strides[0],
                            QImage.Format.Format_RGB888).copy()
            scene.addPixmap(QPixmap.fromImage(qimage))
            border_pen = QPen(QColor("#697787"))
            border_pen.setCosmetic(True)
            scene.addRect(QRectF(0, 0, width, height), border_pen)
            if center_line:
                pen = QPen(QColor("#ffd166"))
                pen.setCosmetic(True)
                pen.setStyle(Qt.PenStyle.DashLine)
                x = (width - 1) / 2 + 0.5
                scene.addLine(x, 0, x, height, pen)
        if points is not None:
            radius = (self.focus_width * 0.016 if self.focus_width is not None
                      else max(canvas_w, canvas_h) * 0.008)
            for index, point in enumerate(points):
                if focus_index is not None and index != focus_index:
                    continue
                # QPixmap occupies [0,w] x [0,h]; pixel centers are offset by 0.5.
                x, y = float(point[0]) + 0.5, float(point[1]) + 0.5
                color = QColor(POINT_COLORS[index])
                pen = QPen(color, 2)
                pen.setCosmetic(True)
                scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, pen)
                scene.addLine(x - radius * 1.7, y, x + radius * 1.7, y, pen)
                scene.addLine(x, y - radius * 1.7, x, y + radius * 1.7, pen)
                label = scene.addSimpleText(f"P{index + 1}")
                label.setBrush(QColor("white") if index == 1 else color)
                label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations)
                label.setPos(x + radius * 1.7, y)
        self.fit_image()

    def fit_image(self):
        if self.focus_point is not None:
            viewport_w = max(1, self.viewport().width() - 4)
            viewport_h = max(1, self.viewport().height() - 4)
            crop_w = self.focus_width
            crop_h = crop_w * viewport_h / viewport_w
            x, y = self.focus_point
            # Keep the point centered even at the image edge. Outside is black.
            self.scene().setSceneRect(QRectF(x - crop_w / 2, y - crop_h / 2,
                                            crop_w, crop_h))
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image()


class PointDetailView(QWidget):
    """Two equally sized views: P1 above, P2 below, in working coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.point_views = []
        self.point_labels = []
        for index in range(2):
            label = QLabel(f"P{index + 1}")
            layout.addWidget(label)
            view = ImageView()
            view.setMinimumHeight(60)
            layout.addWidget(view, 1)
            self.point_labels.append(label)
            self.point_views.append(view)

    def set_image(self, image, canvas_size, points=None, center_line=False):
        for index, view in enumerate(self.point_views):
            view.set_image(image, canvas_size, points, center_line=center_line,
                           focus_index=index)
            if points is not None:
                x, y = points[index]
                self.point_labels[index].setText(f"P{index + 1} 중심 · X {x:.3f}, Y {y:.3f}")


class TemplateWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이미지 템플릿 만들기")
        self.resize(1560, 960)
        self.original = None
        self.result_image = None
        self.image_path = None
        self._updating = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self.refresh)
        self._build_ui()
        self.statusBar().showMessage("이미지 열기를 눌러 시작하세요.")

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)
        toolbar = QHBoxLayout()
        open_button = QPushButton("이미지 열기…")
        open_button.clicked.connect(self.choose_image)
        toolbar.addWidget(open_button)
        self.reset_button = QPushButton("회전·이동 초기화")
        self.reset_button.clicked.connect(self.reset_transform)
        self.reset_button.setEnabled(False)
        toolbar.addWidget(self.reset_button)
        self.save_button = QPushButton("목적 이미지 저장…")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        toolbar.addWidget(self.save_button)
        self.file_label = QLabel("이미지를 선택하세요")
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        toolbar.addWidget(self.file_label, 1)
        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        viewer_widget = QWidget()
        viewers_layout = QHBoxLayout(viewer_widget)
        viewers_layout.setContentsMargins(0, 0, 0, 0)
        viewers_layout.setSpacing(8)
        titles = (
            ("1 · 원본", "역산한 원본 좌표 표시"),
            ("2 · 왼쪽 기준", "왼쪽: 조정 영상  |  오른쪽: 절댓값 차이"),
            ("3 · 두 점 확대", "위: P1 중심  |  아래: P2 중심"),
            ("4 · 목적 이미지", "두 점을 목적 좌표에 맞춘 결과"),
        )
        self.views, self.image_size_labels = [], []
        for index, (title, subtitle) in enumerate(titles):
            panel = QGroupBox(title)
            layout = QVBoxLayout(panel)
            description = QLabel(subtitle)
            description.setWordWrap(True)
            layout.addWidget(description)
            view = PointDetailView() if index == 2 else ImageView()
            layout.addWidget(view, 1)
            dimensions = QLabel("—")
            layout.addWidget(dimensions)
            self.views.append(view)
            self.image_size_labels.append(dimensions)
            viewers_layout.addWidget(panel, 1)
        splitter.addWidget(viewer_widget)

        controls = QWidget()
        controls.setObjectName("numericControls")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 8, 0, 0)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self._transform_group(), 1)
        controls_layout.addWidget(self._source_group(), 1)
        controls_layout.addWidget(self._target_group(), 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumHeight(490)
        splitter.addWidget(scroll)
        splitter.setSizes([370, 500])
        root_layout.addWidget(splitter, 1)
        self.message = QLabel("P1: 빨강  ·  P2: 파랑  |  빈 영역은 검정(0)으로 표시됩니다.")
        self.message.setWordWrap(True)
        root_layout.addWidget(self.message)
        self.setCentralWidget(root)

    def _transform_group(self):
        group = QGroupBox("회전·이동 · 2/3번 공통")
        layout = QGridLayout(group)
        note = QLabel("값을 직접 입력하거나 ±10 / ±1 / ±0.1 버튼으로 조절하세요.")
        note.setWordWrap(True)
        layout.addWidget(note, 0, 0, 1, 2)
        self.angle = StepControl("°")
        self.dx = StepControl("px")
        self.dy = StepControl("px")
        for row, (text, control) in enumerate((("회전", self.angle), ("X 이동", self.dx), ("Y 이동", self.dy)), 1):
            layout.addWidget(labeled_input(text, control), row, 0, 1, 2)
            control.valueChanged.connect(self.schedule_refresh)
        hint = QLabel("회전 +: 반시계 방향 · X +: 오른쪽 · Y +: 아래쪽\n영상 중심에서 회전한 뒤 이동합니다.")
        hint.setWordWrap(True)
        layout.addWidget(hint, 4, 0, 1, 2)
        layout.setRowStretch(5, 1)
        return group

    def _source_group(self):
        group = QGroupBox("입력 두 점 · 2/3번 공통")
        layout = QGridLayout(group)
        self.center_label = QLabel("X = 영상 중심 (고정)")
        layout.addWidget(self.center_label, 0, 0, 1, 2)
        self.source_y = [double_spin(0, 100000), double_spin(0, 100000)]
        for index, spin in enumerate(self.source_y):
            layout.addWidget(
                labeled_input(f"P{index + 1} 높이 Y", StepControl("px", spin=spin)),
                index + 1, 0, 1, 2,
            )
            spin.valueChanged.connect(self.schedule_refresh)
        layout.addWidget(QLabel("원본 영상의 좌표 (회전·이동 역산)"), 3, 0, 1, 2)
        self.original_points_label = QLabel("P1: —\nP2: —")
        self.original_points_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.original_points_label, 4, 0, 1, 2)
        hint = QLabel("2/3번은 동일한 좌표계를 사용합니다.\nX는 (영상 너비 − 1) / 2로 고정됩니다.")
        hint.setWordWrap(True)
        layout.addWidget(hint, 5, 0, 1, 2)
        layout.setRowStretch(6, 1)
        return group

    def _target_group(self):
        group = QGroupBox("목적 이미지 · 4번")
        layout = QGridLayout(group)
        self.target_w, self.target_h = QSpinBox(), QSpinBox()
        for spin in (self.target_w, self.target_h):
            spin.setRange(1, 16384)
            spin.setValue(512)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self.target_size_changed)
        resolution_row = QWidget()
        resolution_layout = QHBoxLayout(resolution_row)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.addWidget(labeled_input("가로 (px)", self.target_w), 1)
        resolution_layout.addWidget(labeled_input("세로 (px)", self.target_h), 1)
        layout.addWidget(resolution_row, 0, 0, 1, 3)
        self.target_points = []
        for index in range(2):
            pair = [double_spin(0, 1023), double_spin(0, 1023)]
            self.target_points.append(pair)
            for axis_index, (axis_name, spin) in enumerate(zip(("X", "Y"), pair)):
                row = 1 + index * 2 + axis_index
                layout.addWidget(
                    labeled_input(f"목적 P{index + 1} {axis_name}", StepControl("px", spin=spin)),
                    row, 0, 1, 3,
                )
                spin.valueChanged.connect(self.schedule_refresh)
        self.mapping_label = QLabel("원본 → 목적 변환: —")
        self.mapping_label.setWordWrap(True)
        self.mapping_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.mapping_label, 5, 0, 1, 3)
        note = QLabel("P1 → 목적 P1, P2 → 목적 P2\n회전·이동·균일 배율을 적용하며, 영상 밖은 가장자리 픽셀로 채웁니다.")
        note.setWordWrap(True)
        layout.addWidget(note, 6, 0, 1, 3)
        layout.setRowStretch(7, 1)
        return group

    def schedule_refresh(self, *_):
        if not self._updating and self.original is not None:
            self._timer.start()

    def target_size_changed(self, *_):
        if self._updating:
            return
        for x_spin, y_spin in self.target_points:
            x_spin.setMaximum(self.target_w.value() - 1)
            y_spin.setMaximum(self.target_h.value() - 1)
        self.schedule_refresh()

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", str(self.image_path.parent) if self.image_path else "",
            "이미지 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;모든 파일 (*)",
        )
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("이미지 파일을 읽을 수 없습니다.")
            height, width = image.shape[:2]
            if width * height > MAX_IMAGE_PIXELS or max(width, height) > 16384:
                raise ValueError("이미지는 한 변 16,384 px, 총 3,200만 픽셀 이하여야 합니다.")
            if min(width, height) < 2:
                raise ValueError("가로·세로가 2 px 이상인 이미지를 선택하세요.")
        except (OSError, ValueError, cv2.error) as error:
            QMessageBox.warning(self, "이미지 로드 오류", str(error))
            return False
        self._timer.stop()
        self._updating = True
        self.original = image
        self.image_path = Path(path)
        self.angle.setValue(0)
        self.dx.setValue(0)
        self.dy.setValue(0)
        self.target_w.setValue(width)
        self.target_h.setValue(height)
        for index, fraction in enumerate((0.25, 0.75)):
            self.source_y[index].setMaximum(height - 1)
            self.source_y[index].setValue((height - 1) * fraction)
            x_spin, y_spin = self.target_points[index]
            x_spin.setMaximum(width - 1)
            y_spin.setMaximum(height - 1)
            x_spin.setValue((width - 1) / 2)
            y_spin.setValue(self.source_y[index].value())
        self._updating = False
        self.center_label.setText(f"X = {(width - 1) / 2:.3f} px (영상 중심, 고정)")
        self.file_label.setText(f"{self.image_path.name} · {width} × {height} px")
        self.file_label.setToolTip(str(self.image_path))
        self.reset_button.setEnabled(True)
        self.refresh()
        return True

    def reset_transform(self):
        self._updating = True
        for control in (self.angle, self.dx, self.dy):
            control.setValue(0)
        self._updating = False
        self.refresh()

    def refresh(self):
        if self.original is None:
            return
        self._timer.stop()
        height, width = self.original.shape[:2]
        dst_w, dst_h = self.target_w.value(), self.target_h.value()
        canvas_size = (max(width, dst_w), max(height, dst_h))
        matrix = manual_transform(width, height, self.angle.value(), self.dx.value(), self.dy.value())
        working = warp_zero(self.original, matrix, width, height)
        second = comparison_image(working)
        source_points = np.array([[(width - 1) / 2, spin.value()] for spin in self.source_y])
        target_points = np.array([[x.value(), y.value()] for x, y in self.target_points])
        original_points = transform_points(source_points, cv2.invertAffineTransform(matrix))
        self.original_points_label.setText("\n".join(
            f"P{index + 1}:  X = {point[0]:.3f},  Y = {point[1]:.3f}"
            for index, point in enumerate(original_points)
        ))
        self.result_image = None
        self.save_button.setEnabled(False)
        try:
            if dst_w * dst_h > MAX_IMAGE_PIXELS:
                raise ValueError("목적 이미지의 총 픽셀 수는 3,200만 이하여야 합니다.")
            alignment = similarity_transform(source_points, target_points)
            # Compose original->working->target; sample the original only once.
            combined = (np.vstack((alignment, [0, 0, 1])) @ np.vstack((matrix, [0, 0, 1])))[:2]
            self.result_image = cv2.warpAffine(
                self.original, combined, (dst_w, dst_h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            scale = np.hypot(combined[0, 0], combined[1, 0])
            angle = np.degrees(np.arctan2(-combined[1, 0], combined[0, 0]))
            self.mapping_label.setText(
                f"원본 → 목적: 배율 {scale:.5f} · 회전 {angle:.3f}°\n"
                f"행렬 이동항: X {combined[0, 2]:.3f}, Y {combined[1, 2]:.3f}"
            )
            inside = ((original_points[:, 0] >= 0) & (original_points[:, 0] <= width - 1)
                      & (original_points[:, 1] >= 0) & (original_points[:, 1] <= height - 1))
            self.message.setText(
                "P1: 빨강 · P2: 파랑 | 크기가 다른 영상은 검정 여백으로 표시합니다."
                if inside.all() else
                "입력한 점 중 일부가 원본 영상 밖에 있습니다. 목적 영상의 바깥 영역은 가장자리 픽셀로 채웁니다."
            )
            self.save_button.setEnabled(True)
        except (ValueError, cv2.error) as error:
            self.mapping_label.setText("원본 → 목적 변환: —")
            self.message.setText(str(error))
        images = (self.original, second, working, self.result_image)
        point_sets = (original_points, source_points, source_points, target_points)
        for index, (view, image, points) in enumerate(zip(self.views, images, point_sets)):
            view.set_image(image, canvas_size, points, center_line=index in (1, 2))
            actual_w, actual_h = (dst_w, dst_h) if index == 3 else (width, height)
            self.image_size_labels[index].setText(
                f"각 뷰 가로 범위: {max(1.0, width / 4):g} px" if index == 2
                else f"{actual_w} × {actual_h} px"
            )
        self.statusBar().showMessage("값 입력 후 Enter 또는 다른 항목을 선택하면 영상에 반영됩니다.")

    def save_result(self):
        # A queued control edit must be reflected in the exported pixels.
        self.refresh()
        if self.result_image is None:
            return
        suggested = self.image_path.with_name(f"{self.image_path.stem}_template.png")
        path, _ = QFileDialog.getSaveFileName(self, "목적 이미지 저장", str(suggested), "PNG 이미지 (*.png)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".png"
        try:
            ok, encoded = cv2.imencode(".png", self.result_image)
            if not ok:
                raise ValueError("PNG 인코딩에 실패했습니다.")
            encoded.tofile(path)
        except (OSError, ValueError, cv2.error) as error:
            QMessageBox.warning(self, "저장 오류", str(error))
            return
        self.statusBar().showMessage(f"저장 완료: {path} (표시용 점과 선은 저장되지 않습니다.)")


def main():
    parser = argparse.ArgumentParser(description="Qt image template editor")
    parser.add_argument("image", nargs="?", help="시작할 때 열 이미지")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget { font-family: 'Noto Sans CJK KR', sans-serif; font-size: 12px; }
        QMainWindow, QScrollArea { background: #eef1f5; }
        QGroupBox { background: #ffffff; border: 1px solid #cbd2dc;
                    border-radius: 5px; margin-top: 12px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                          color: #ffffff; background: #4b5563;
                          border-radius: 3px; font-weight: bold; }
        QWidget#numericControls QGroupBox { margin-top: 28px; padding-top: 16px; }
        QWidget#numericControls QGroupBox::title {
            subcontrol-position: top left; top: 6px; padding: 2px 4px;
        }
        QWidget#numericControls QLabel { padding-top: 4px; padding-bottom: 2px; }
        QWidget#numericControls QLabel[inputLabel="true"] {
            color: #ffffff; background: #4b5563; border-radius: 3px;
            padding: 4px 6px;
        }
        QPushButton { padding: 5px 9px; }
        QDoubleSpinBox, QSpinBox { padding: 4px; }
    """)
    window = TemplateWindow()
    window.show()
    if args.image:
        window.load_image(args.image)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
