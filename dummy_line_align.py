import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform


class MVTecScrewAlignment(Transform):
    _transformed_types = (tv_tensors.Image, tv_tensors.Mask)

    def __init__(self, template=None):
        super().__init__()
        self.template = template

    def make_params(self, flat_inputs):
        image = next(
            x for x in flat_inputs
            if isinstance(x, tv_tensors.Image)
        )

        # 입력: RGB float32, [C, H, W], 0~1
        rgb = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()

        rgb_u8 = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

        # 1. 이미지에서 에지 추출
        edges = cv2.Canny(gray, 50, 150)

        # 2. 에지에서 긴 선분 검출
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 1800,
            threshold=35,
            minLineLength=60,
            maxLineGap=12,
        )

        if lines is None:
            raise RuntimeError("이미지에서 정렬할 선분을 찾지 못했습니다.")

        lines = lines.reshape(-1, 4)
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]

        angles = np.degrees(np.arctan2(dy, dx))
        lengths = np.hypot(dx, dy)

        # 선 방향은 180도 주기.
        # 예: 1도와 179도를 그대로 평균/중앙값 내면 잘못될 수 있으므로
        # 가장 긴 선분의 방향을 기준으로 각도를 펼칩니다.
        reference = angles[np.argmax(lengths)]
        unwrapped = reference + (angles - reference + 90) % 180 - 90

        # 3. 선분 각도들의 중앙값
        estimated_angle = float(np.median(unwrapped) % 180)

        # 에지 좌표의 중앙값을 회전 중심으로 사용
        ys, xs = np.nonzero(edges)
        center = (float(np.median(xs)), float(np.median(ys)))

        # OpenCV 좌표계에서 수직 방향은 90도
        rotation_angle = estimated_angle - 90.0

        matrix = cv2.getRotationMatrix2D(
            center,
            rotation_angle,
            1.0,
        )

        # 예제 시각화용 정보
        self.last_edges = edges
        self.last_estimated_angle = estimated_angle
        self.last_rotation_angle = rotation_angle

        # 이미지와 마스크에 공유할 파라미터
        return {
            "matrix": matrix,
            "size": (gray.shape[1], gray.shape[0]),  # width, height
        }

    def transform(self, inpt, params):
        is_mask = isinstance(inpt, tv_tensors.Mask)

        tensor = inpt.detach().cpu().as_subclass(torch.Tensor)

        if is_mask:
            # 이 예제의 마스크: [H, W], 0 또는 1
            array = tensor.to(torch.uint8).numpy()
        else:
            array = tensor.permute(1, 2, 0).contiguous().numpy()

        # 4. 동일한 행렬을 이미지와 마스크 양쪽에 적용
        output = cv2.warpAffine(
            np.ascontiguousarray(array),
            params["matrix"],
            params["size"],
            flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        result = torch.from_numpy(np.ascontiguousarray(output))

        if not is_mask:
            result = result.permute(2, 0, 1).contiguous()

        result = result.to(device=inpt.device, dtype=inpt.dtype)
        return tv_tensors.wrap(result, like=inpt)


def make_dummy_pair(angle, size=256):
    """검은 배경에 두꺼운 흰 선과 동일한 위치의 마스크 생성."""
    radians = np.deg2rad(angle)

    center = np.array([size / 2, size / 2])
    direction = 85 * np.array([
        np.cos(radians),
        np.sin(radians),
    ])

    point1 = tuple(np.rint(center - direction).astype(int))
    point2 = tuple(np.rint(center + direction).astype(int))

    canvas = np.zeros((size, size), dtype=np.uint8)

    cv2.line(
        canvas,
        point1,
        point2,
        color=255,
        thickness=15,
        lineType=cv2.LINE_8,
    )

    rgb = np.repeat(canvas[..., None], 3, axis=2)

    image = tv_tensors.Image(
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .contiguous()
        .float() / 255.0
    )

    mask = tv_tensors.Mask(
        torch.from_numpy((canvas > 0).astype(np.uint8))
    )

    return image, mask


def main():
    rng = np.random.default_rng(42)
    alignment = MVTecScrewAlignment()

    fig, axes = plt.subplots(5, 5, figsize=(7, 20))

    for i in range(5):
        # 매번 다른 각도로 더미 이미지·마스크 생성
        input_angle = float(rng.uniform(0, 180))
        img, mask = make_dummy_pair(input_angle)

        # 원하는 호출 형태
        r_img, r_mask = alignment(img, mask)

        print(
            f"{i + 1:02d} | "
            f"생성 각도: {input_angle:7.2f}° | "
            f"추정 각도: {alignment.last_estimated_angle:7.2f}° | "
            f"회전량: {alignment.last_rotation_angle:7.2f}°"
        )

        original_rgb = img.permute(1, 2, 0).numpy()
        aligned_rgb = r_img.permute(1, 2, 0).numpy()
        aligned_mask = r_mask.numpy()

        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title(f"Input: {input_angle:.1f} deg")

        axes[i, 1].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Input mask")

        axes[i, 2].imshow(alignment.last_edges, cmap="gray")
        axes[i, 2].set_title("Detected edges")

        axes[i, 3].imshow(aligned_rgb)
        axes[i, 3].set_title("Aligned image")

        axes[i, 4].imshow(aligned_rgb)
        axes[i, 4].contour(
            aligned_mask,
            levels=[0.5],
            colors="lime",
            linewidths=1,
        )
        axes[i, 4].set_title("Aligned mask overlay")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()