
import cv2

from coarse_alignment import coarse_alignment_pts_mvtecad_screw, align_end_points_to_target


if __name__ == "__main__":
    from pathlib import Path
    import shutil

    src_dir = '/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw'
    target_dir = '/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw_crop'

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