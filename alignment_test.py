import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from common.coarse_alignment import coarse_alignment_pts_mvtecad_screw, align_end_points_to_target

# OpenCV의 Qt5 창은 XWayland를 명시적으로 사용합니다.
if os.environ.get('XDG_SESSION_TYPE') == 'wayland' and os.environ.get('DISPLAY'):
    platform = os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    if platform == 'xcb':
        os.environ['XDG_SESSION_TYPE'] = 'x11'

# cv2 import가 지정한 fonts 경로가 없으면 실제 시스템 폰트 경로로 바꿉니다.
qt_font_dir = os.environ.get('QT_QPA_FONTDIR')
if qt_font_dir and not Path(qt_font_dir).is_dir():
    system_font_dir = Path('/usr/share/fonts/truetype/dejavu')
    if system_font_dir.is_dir():
        os.environ['QT_QPA_FONTDIR'] = str(system_font_dir)
    else:
        os.environ.pop('QT_QPA_FONTDIR', None)



def draw_end_points(image, end_pt1, end_pt2, c1=(0, 0, 255), c2=(255, 0, 0)):
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
    # 원 중심과 반지름도 shift=8 고정소수점으로 전달해 소수점 좌표를 표시합니다.
    center1 = tuple(np.rint(point1 * (1 << shift)).astype(int).tolist())
    center2 = tuple(np.rint(point2 * (1 << shift)).astype(int).tolist())
    radius = 3 * (1 << shift)
    cv2.circle(image, center1, radius, c1, -1, cv2.LINE_AA, shift)
    cv2.circle(image, center2, radius, c2, -1, cv2.LINE_AA, shift)


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def process_image_directory(image_dir, output_dir, tpl_target_img, tpl_w, tpl_h,
                            tpl_front_pt, tpl_head_pt,
                            min_component_pixels=70, resize_ratio=0.09,
                            filename_prefix=''):
    """Align images and save template/aligned/difference comparison strips.

    The saved image keeps its original filename, preceded by filename_prefix
    when one is supplied. Return (saved_paths, failures).
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f'Image directory not found: {image_dir}')
    if tpl_target_img is None or tpl_target_img.size == 0:
        raise ValueError('Template image must not be empty')
    if tpl_w <= 0 or tpl_h <= 0:
        raise ValueError('Template width and height must be positive')
    if tpl_target_img.shape[:2] != (tpl_h, tpl_w):
        raise ValueError(
            'Template image size does not match tpl_w and tpl_h: '
            f'{tpl_target_img.shape[1]}x{tpl_target_img.shape[0]} != {tpl_w}x{tpl_h}'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    saved_paths = []
    failures = []

    for image_path in image_paths:
        if "mask" in image_path.stem:
            continue

        started_at = perf_counter()
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError('Image could not be decoded')

            coarse_front_pt, coarse_head_pt = coarse_alignment_pts_mvtecad_screw(
                img, min_component_pixels, resize_ratio,
            )
            aligned_img, _ = align_end_points_to_target(
                img, coarse_front_pt, coarse_head_pt,
                tpl_front_pt, tpl_head_pt, tpl_w, tpl_h,
            )
            diff_img = cv2.absdiff(tpl_target_img, aligned_img)
            comparison_img = cv2.hconcat(
                (tpl_target_img, aligned_img, diff_img)
            )

            output_name = (
                f'{filename_prefix}_{image_path.name}'
                if filename_prefix else image_path.name
            )
            output_path = output_dir / output_name
            if not cv2.imwrite(str(output_path), comparison_img):
                raise OSError(f'Failed to write comparison image: {output_path}')
        except (cv2.error, OSError, ValueError) as error:
            failures.append((image_path, str(error)))
            print(f'[FAILED] {image_path}: {error}')
            continue

        saved_paths.append(output_path)
        elapsed_ms = (perf_counter() - started_at) * 1000
        print(f'[{image_path.name}] -> {output_name}: {elapsed_ms:.3f} ms')

    return saved_paths, failures


def process_image_subdirectories(root_dir, output_dir, tpl_target_img, tpl_w, tpl_h,
                                 tpl_front_pt, tpl_head_pt,
                                 min_component_pixels=70, resize_ratio=0.09):
    """Recursively align images and flatten all results into output_dir.

    Relative subdirectory names are joined with underscores and prepended to
    each output filename, so equal image names from different folders do not
    overwrite each other.
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    if not root_dir.is_dir():
        raise NotADirectoryError(f'Root image directory not found: {root_dir}')

    root_resolved = root_dir.resolve()
    output_resolved = output_dir.resolve()
    image_dirs = set()
    for image_path in root_dir.rglob('*'):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved_path = image_path.resolve()
        if resolved_path == output_resolved or output_resolved in resolved_path.parents:
            continue
        image_dirs.add(image_path.parent)

    all_saved_paths = []
    all_failures = []
    for image_dir in sorted(image_dirs):
        relative_dir = image_dir.resolve().relative_to(root_resolved)
        prefix = root_dir.name if relative_dir == Path('.') else '_'.join(relative_dir.parts)
        saved_paths, failures = process_image_directory(
            image_dir, output_dir, tpl_target_img, tpl_w, tpl_h,
            tpl_front_pt, tpl_head_pt,
            min_component_pixels=min_component_pixels,
            resize_ratio=resize_ratio,
            filename_prefix=prefix,
        )
        all_saved_paths.extend(saved_paths)
        all_failures.extend(failures)

    return all_saved_paths, all_failures



if __name__ == '__main__':
    image_root_dir = Path(
        '/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw'
    )
    output_dir = Path(
        '/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw_coarse_alignment_test'
    )
    tpl_target_img_path = Path(
        '/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/'
        'screw_templates/000_t_256.png'
    )
    tpl_target_img = cv2.imread(str(tpl_target_img_path))
    if tpl_target_img is None:
        raise FileNotFoundError(f'Could not read template: {tpl_target_img_path}')

    min_component_pixels = 70
    resize_ratio = 0.09
    tpl_w, tpl_h = 128, 256
    tpl_h, tpl_w = tpl_target_img.shape[:2]
    tpl_front_pt, tpl_head_pt = (64.0, 20.0), (64.0, 236.0)

    saved_paths, failures = process_image_subdirectories(
        image_root_dir, output_dir, tpl_target_img, tpl_w, tpl_h,
        tpl_front_pt, tpl_head_pt,
        min_component_pixels=min_component_pixels,
        resize_ratio=resize_ratio,
    )
    print(f'Saved {len(saved_paths)} aligned images to: {output_dir}')
    print(f'Failed images: {len(failures)}')
