import numpy as np
import cv2

def coarse_alignment_pts_mvtecad_screw(img, min_component_pixels=70, resize_ratio=0.09):
    '''
    1.그레이변환
    2.영상 축소, 원본대비 resize_ratio
    3.에지
    4.이진화
    5.작은 연결 컴펀너트 제거, 리싸이즈된 상태에서 픽셀단위 min_component_pixels
    6.컨벡스홀
    7.무게중심 공분산
    8.라인위의 끝점 2개 찾음
    9.라인위의 끝점 2개를 비교해서 두께비교 프론트와 헤드 구분
    '''

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
        raise ValueError('Two distinct endpoints not found')

    # find front and head
    front_pt, head_pt = find_front_head(foreground, hull_mask, end_pt1, end_pt2)

    if front_pt is None or head_pt is None:
        raise ValueError('front_pt or head_pt not found')

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


def align_end_points_to_target(img, source_front_pt, source_head_pt,
                                 target_front_pt=(128.0, 20.0), target_head_pt=(128.0, 236.0), target_w=128, target_h=256):
    """Map two source endpoints to the template endpoints with one affine warp."""
    source_points = np.asarray(
        [source_front_pt, source_head_pt], dtype=np.float64
    )
    target_points = np.asarray(
        [target_front_pt, target_head_pt], dtype=np.float64
    )
    source_axis = source_points[1] - source_points[0]
    target_axis = target_points[1] - target_points[0]
    source_length_sq = float(source_axis @ source_axis)
    if source_length_sq == 0 or float(target_axis @ target_axis) == 0:
        raise ValueError('Front and head must be distinct in both coordinate systems')

    # 회전과 균일 스케일을 동시에 계산합니다.
    a = float(source_axis @ target_axis) / source_length_sq
    b = float(source_axis[0] * target_axis[1]
              - source_axis[1] * target_axis[0]) / source_length_sq
    linear = np.array([[a, -b], [b, a]], dtype=np.float64)
    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = target_points[0] - linear @ source_points[0]

    aligned_img = cv2.warpAffine(
        img, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned_img, matrix

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

if __name__ == "__main__":
    from pathlib import Path
    import shutil

    src_dir = '/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw'
    target_dir = '/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw_coarse_alignment'

    source = Path(src_dir).resolve()
    target = Path(target_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and target directories must not overlap")

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    image_paths = []
    mask_paths = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in image_extensions:
            continue
        relative = path.relative_to(source)
        if relative.parts[0] == 'ground_truth':
            mask_paths.append(path)
        else:
            # PNG 출력과 원래 파일명을 동시에 보장합니다.
            if path.suffix.lower() != '.png':
                raise ValueError(f"Expected a PNG image to preserve its filename: {path}")
            image_paths.append(path)

    # MVTec: ground_truth/<defect>/<id>_mask.png -> test/<defect>/<id>.png
    # 마스크 자체에서 끝점을 추정하지 않고 대응 원본의 변환행렬을 사용합니다.
    image_lookup = {
        path.relative_to(source).with_suffix(''): path for path in image_paths
    }
    mask_sources = {}
    for mask_path in mask_paths:
        relative = mask_path.relative_to(source)
        stem = mask_path.stem
        if stem.endswith('_mask'):
            stem = stem[:-len('_mask')]
        image_key = Path('test').joinpath(*relative.parts[1:-1], stem)
        if image_key not in image_lookup:
            raise FileNotFoundError(f"No matching test image for mask: {mask_path}")
        mask_sources[mask_path] = image_lookup[image_key]

    def prepare_output_file(output_path):
        # 부모 폴더가 없으면 생성하고, 이전 실행의 읽기 전용 파일도 저장 가능하게 합니다.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.chmod(0o777)
        if output_path.exists():
            output_path.chmod(0o777)

    # copytree는 원본의 읽기 전용 폴더 권한까지 복사하므로 직접 생성합니다.
    # 권한 변경은 출력 경로에만 적용합니다.
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o777)
    transformed_paths = set(image_paths) | set(mask_paths)
    for path in sorted(source.rglob('*')):
        output_path = target / path.relative_to(source)
        if path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            output_path.chmod(0o777)
        elif path.is_file() and path not in transformed_paths:
            # LICENSE, README 등 모든 비이미지 파일도 복사합니다.
            prepare_output_file(output_path)
            shutil.copy2(path, output_path)
            output_path.chmod(0o777)

    matrices = {}
    for index, image_path in enumerate(image_paths, start=1):
        # 저장용 원본은 채널 수와 비트 깊이를 그대로 유지합니다.
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        if (height, width) != (1024, 1024):
            raise ValueError(f"Expected a 1024x1024 image: {image_path}, got {image.shape}")
        # 끝점 검출 함수에는 기존과 동일하게 8비트 BGR 이미지를 전달합니다.
        detection_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if detection_image is None:
            raise ValueError(f"Cannot read image for endpoint detection: {image_path}")
        try:
            front_pt, head_pt = coarse_alignment_pts_mvtecad_screw(detection_image)
            aligned, matrix = align_end_points_to_target(
                image, front_pt, head_pt,
                target_front_pt=(512, 80), target_head_pt=(512, 944),
                target_w=width, target_h=height,
            )
        except ValueError as exc:
            raise ValueError(f"Alignment failed for {image_path}: {exc}") from exc
        output_path = target / image_path.relative_to(source)
        prepare_output_file(output_path)
        if not cv2.imwrite(str(output_path), aligned):
            raise OSError(f"Cannot write image: {output_path}")
        output_path.chmod(0o777)
        matrices[image_path] = matrix
        print(f"Image {index}/{len(image_paths)}: {image_path.relative_to(source)}")

    for index, mask_path in enumerate(mask_paths, start=1):
        # 원래 마스크의 dtype과 채널 수를 유지합니다.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        if mask.shape[:2] != (1024, 1024):
            raise ValueError(f"Mask size does not match its image: {mask_path}, got {mask.shape}")
        aligned_mask = cv2.warpAffine(
            mask, matrices[mask_sources[mask_path]], (1024, 1024),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        output_path = target / mask_path.relative_to(source)
        prepare_output_file(output_path)
        if not cv2.imwrite(str(output_path), aligned_mask):
            raise OSError(f"Cannot write mask: {output_path}")
        output_path.chmod(0o777)
        print(f"Mask {index}/{len(mask_paths)}: {mask_path.relative_to(source)}")

    print(f"Done: {len(image_paths)} images, {len(mask_paths)} masks -> {target}")
