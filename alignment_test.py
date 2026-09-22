from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def find_end_points(foreground, hull_mask):
    """Return the two extreme hull pixels on the PCA axis as (x, y)."""
    if foreground is not None and len(foreground) >= 2:
        points = foreground.reshape(-1, 2).astype(np.float64)
        center = points.mean(axis=0)
        centered = points - center
        covariance = centered.T @ centered / len(points)
        _, eigenvectors = np.linalg.eigh(covariance)
        principal_direction = eigenvectors[:, -1]
        # Fix the sign so endpoint ordering is consistent.
        dominant_axis = np.argmax(np.abs(principal_direction))
        if principal_direction[dominant_axis] < 0:
            principal_direction = -principal_direction

        # Rasterize the center axis, then keep only pixels inside the hull.
        height, width = hull_mask.shape
        reach = np.hypot(width, height)
        line_start = center - reach * principal_direction
        line_end = center + reach * principal_direction
        axis_mask = np.zeros_like(hull_mask)
        shift = 8
        fixed_start = tuple(np.rint(line_start * (1 << shift)).astype(int).tolist())
        fixed_end = tuple(np.rint(line_end * (1 << shift)).astype(int).tolist())
        cv2.line(axis_mask, fixed_start, fixed_end, 255, 1, cv2.LINE_8, shift)
        axis_pixels = cv2.findNonZero(cv2.bitwise_and(axis_mask, hull_mask))
        if axis_pixels is not None and len(axis_pixels) >= 2:
            axis_pixels = axis_pixels.reshape(-1, 2)
            projection = (axis_pixels - center) @ principal_direction
            end_point_1 = tuple(axis_pixels[projection.argmin()].tolist())
            end_point_2 = tuple(axis_pixels[projection.argmax()].tolist())

            return end_point_1, end_point_2
    return None, None

def find_front_head(foreground, hull_mask, end_pt1, end_pt2,
                    start_ratio=0.05, end_ratio=0.2, num_sections=5):
    """Return (front, head) by comparing median widths near the endpoints.

    Widths are approximated by perpendicular pixel spans in axial strips.
    Return (None, None) for missing data, coincident endpoints or equal widths.
    """
    if not 0 <= start_ratio < end_ratio <= 0.5:
        raise ValueError('Expected 0 <= start_ratio < end_ratio <= 0.5')
    if not isinstance(num_sections, (int, np.integer)) or num_sections < 1:
        raise ValueError('num_sections must be a positive integer')
    if end_pt1 is None or end_pt2 is None:
        return None, None

    if foreground is None:
        return None, None

    point1 = np.asarray(end_pt1, dtype=np.float64)
    point2 = np.asarray(end_pt2, dtype=np.float64)
    axis = point2 - point1
    length = np.linalg.norm(axis)
    if length == 0:
        return None, None
    axis /= length
    perpendicular = np.array([-axis[1], axis[0]])

    # 모든 마스크 좌표를 나사 축 방향과 수직 방향으로 투영합니다.
    relative = foreground.reshape(-1, 2).astype(np.float64) - point1
    along = relative @ axis
    across = relative @ perpendicular
    boundaries = np.linspace(start_ratio * length, end_ratio * length,
                             num_sections + 1)

    def median_width(distance_from_end):
        widths = []
        for index in range(num_sections):
            lower, upper = boundaries[index:index + 2]
            # 각 구간의 수직 좌표 최댓값 - 최솟값으로 단면 폭을 근사합니다.
            selected = (distance_from_end >= lower) & (distance_from_end < upper)
            if index == num_sections - 1:
                selected = (distance_from_end >= lower) & (distance_from_end <= upper)
            section = across[selected]
            if section.size < 2:
                return None
            widths.append(float(np.ptp(section)))
        return float(np.median(widths))

    width1 = median_width(along)
    width2 = median_width(length - along)
    if width1 is None or width2 is None or np.isclose(width1, width2):
        return None, None
    if width1 < width2:
        return end_pt1, end_pt2
    return end_pt2, end_pt1


def draw_end_points(image, end_pt1, end_pt2):
    """Draw the infinite line through both endpoints and mark them in place."""
    point1 = np.asarray(end_pt1, dtype=np.float64)
    point2 = np.asarray(end_pt2, dtype=np.float64)
    direction = point2 - point1
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError('Two distinct endpoints are required to draw a line')

    height, width = image.shape[:2]
    direction /= length
    reach = np.hypot(width, height)
    # OpenCV clips the extended line at the image boundaries.
    shift = 8
    line_start = tuple(np.rint((point1 - reach * direction) * (1 << shift)).astype(int).tolist())
    line_end = tuple(np.rint((point2 + reach * direction) * (1 << shift)).astype(int).tolist())
    cv2.line(image, line_start, line_end, (0, 255, 0), 1, cv2.LINE_AA, shift)
    cv2.circle(image, end_pt1, 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(image, end_pt2, 3, (255, 0, 0), -1, cv2.LINE_AA)


def coarse_alignment_mvtecad_screw(img, min_component_pixels, resize_ratio):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray_resize = cv2.resize(img_gray, (0, 0), fx=resize_ratio, fy=resize_ratio)
    resize_h, resize_w = img_gray_resize.shape

    # Edge detection
    scharr_x = cv2.Scharr(img_gray_resize, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(img_gray_resize, cv2.CV_32F, 0, 1)
    scharr_magnitude = cv2.magnitude(scharr_x, scharr_y)

    scharr_display = cv2.normalize(
        scharr_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    _, scharr_display = cv2.threshold(
        scharr_display, 70, 255, cv2.THRESH_BINARY
    )

    # Delete small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        scharr_display, connectivity=8
    )
    # Label 0 is the background; filter only foreground components.
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_component_pixels:
            scharr_display[labels == label] = 0

    # convex hull
    edge_points = cv2.findNonZero(scharr_display)
    if edge_points is not None:
        hull = cv2.convexHull(edge_points)
        scharr_display.fill(0)
        cv2.fillConvexPoly(scharr_display, hull, 255)

    # mean, covariance, end point
    hull_mask = scharr_display.copy() # 이미지 배열, 0 or 255,
    foreground = cv2.findNonZero(hull_mask) # 0이 아닌 픽셀 좌표 목록

    # find end points
    end_pt1, end_pt2 = find_end_points(foreground, hull_mask)

    if end_pt1 is None or end_pt2 is None:
        raise ValueError(f'Two distinct endpoints not found: {image_path}')

    # find front and head
    front_pt, head_pt = find_front_head(foreground, hull_mask, end_pt1, end_pt2)

    if front_pt is None or head_pt is None:
        raise ValueError(f'front_pt or head_pt not found: {image_path}')

    # 실제 리사이즈 크기를 기준으로 원본 좌표로 환산합니다.
    original_h, original_w = img.shape[:2]
    scale_x = original_w / resize_w
    scale_y = original_h / resize_h
    original_front_pt = (round(front_pt[0] * scale_x), round(front_pt[1] * scale_y))
    original_head_pt = (round(head_pt[0] * scale_x), round(head_pt[1] * scale_y))
    # img_resize_display = cv2.cvtColor(img_gray_resize, cv2.COLOR_GRAY2BGR)
    # scharr_display = cv2.cvtColor(scharr_display, cv2.COLOR_GRAY2BGR)
    #
    # draw_end_points(img_resize_display, front_pt, head_pt)
    # draw_end_points(scharr_display, front_pt, head_pt)
    #
    # comparison = cv2.hconcat([img_resize_display, scharr_display])

    return original_front_pt, original_head_pt


if __name__ == '__main__':

    image_dir = Path('/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw/train/good')
    min_component_pixels = 70
    resize_ratio = 0.09

    for image_path in sorted(image_dir.glob('*.png')):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')
        started_at = perf_counter()

        original_front_pt, original_head_pt = coarse_alignment_mvtecad_screw(img, min_component_pixels, resize_ratio)

        draw_end_points(img, original_front_pt, original_head_pt)

        elapsed_ms = (perf_counter() - started_at) * 1000
        print(f'[{image_path.name}] proc duration: {elapsed_ms:.3f} ms')

        cv2.imshow('result', img )
        key = cv2.waitKey(0)

        if key == ord('q'):
            break