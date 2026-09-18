"""Dataset 단계에서 사용하는 CPU / NumPy 기반 이미지·마스크 전처리.

Compose에서 Normalize 앞에 MVTecScrewAlignment()를 배치한다.
1024×1024 입력을 warpAffine에서 1/4로 축소해 256×256으로 출력한다.
입력은 이미지 한 장(Image, RGB float32 [3,H,W], 0~1)과 선택적 마스크
(Mask, uint8 또는 bool [H,W], 0/1)이다. 배치/CUDA 입력은 지원하지 않는다.

처리 순서:
    원본 이미지 → NumPy → estimate_srt → 공통 affine 행렬
    이미지 → 선형 보간 warp → 이미지 전용 밝기/색상 변환 → Image
    마스크 → 최근접 보간 warp → Mask

estimate_srt만 교체하면 실제 정렬 알고리즘을 연결할 수 있다.
이미지 전용 변환에는 회전/크롭 등 공간 변환을 넣지 않는다.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SRTParameters:
    """입력 이미지에 적용할 보정값. 감지한 자세 자체가 아니라 보정량이다.

    rotation: OpenCV 기준 반시계 방향 양수, 단위 degree.
    scale: 양수인 균일 배율. 1은 원래 크기.
    translation: 회전·스케일 적용 후 이동할 (dx, dy), 단위 pixel.
                 양수 dx는 오른쪽, 양수 dy는 아래쪽이다.
    center: 회전·스케일 중심 (x,y). None이면 이미지 중심을 사용한다.
    """
    rotation: float = 0.0
    scale: float = 1.0
    translation: tuple[float, float] = (0.0, 0.0)
    center: tuple[float, float] | None = None


class MVTecScrewAlignment(Transform):
    """이미지에서 구한 SRT를 이미지와 마스크에 동일하게 적용한다.

    image_augmentation:
        정렬된 RGB NumPy 배열(float32 [H,W,3], 0~1)을 받아 같은 크기,
        자료형, 범위의 배열을 반환하는 선택적 callable.
        밝기/대비/색상/노이즈 등 픽셀 값 변환만 사용한다.
        공간 변환은 반드시 공통 SRT 경로에 구현해야 한다.

    기본 SRT는 항등 정렬이며, 정렬 후 원점 기준 1/4 축소를 합성한다.
    출력 캔버스는 256×256이다. 1024×1024 입력을 대상으로 하며,
    이동·회전으로 생기는 빈 영역은 이미지와 마스크 모두 0으로 채운다.
    캔버스 밖으로 이동한 내용은 잘린다.
    """
    _transformed_types = (tv_tensors.Image, tv_tensors.Mask)

    def __init__(
        self,
        image_augmentation: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        super().__init__()
        self.image_augmentation = image_augmentation

    def check_inputs(self, flat_inputs: list[Any]) -> None:
        images = [x for x in flat_inputs if isinstance(x, tv_tensors.Image)]
        if len(images) != 1:
            raise ValueError("Image 한 장을 전달하세요. 이미지와 마스크는 함께 전달합니다.")
        image = images[0]
        if image.device.type != "cpu":
            raise ValueError("CUDA 이동 전 Dataset의 augmentations에서 사용하세요.")
        if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[-2:]) == 0:
            raise ValueError("이미지는 RGB [3,H,W] 한 장이어야 합니다.")
        if image.dtype != torch.float32:
            raise TypeError("이미지는 float32여야 합니다.")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("이미지는 0~1 범위여야 합니다. Normalize보다 앞에 배치하세요.")
        for mask in (x for x in flat_inputs if isinstance(x, tv_tensors.Mask)):
            if mask.device.type != "cpu" or mask.shape != image.shape[-2:]:
                raise ValueError("마스크는 이미지와 크기가 같은 CPU [H,W] 텐서여야 합니다.")
            if mask.dtype not in (torch.uint8, torch.bool):
                raise TypeError("마스크는 uint8 또는 bool이어야 합니다.")
            if not ((mask == 0) | (mask == 1)).all():
                raise ValueError("마스크 값은 0 또는 1이어야 합니다.")

    def estimate_srt(self, rgb: np.ndarray) -> SRTParameters:
        """정렬 알고리즘을 구현할 자리.

        rgb는 Normalize 전 RGB float32 NumPy 배열 [H,W,3], 0~1이다.
        원본 텐서와 분리된 복사본이므로 OpenCV에서 분석할 수 있다.
        현재는 rotation=0, scale=1, translation=(0,0)을 반환한다.
        """
        return SRTParameters()

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        image = next(x for x in flat_inputs if isinstance(x, tv_tensors.Image))
        rgb = self._to_numpy(image)
        height, width = rgb.shape[:2]
        # 한 쌍당 한 번 추정한 뒤, 양쪽 transform 호출에서 같은 행렬을 쓴다.
        srt = self.estimate_srt(rgb)
        if not isinstance(srt, SRTParameters):
            raise TypeError("estimate_srt는 SRTParameters를 반환해야 합니다.")
        center = srt.center if srt.center is not None else ((width - 1) / 2, (height - 1) / 2)
        if len(srt.translation) != 2 or len(center) != 2:
            raise ValueError("translation과 center는 (x,y) 두 값이어야 합니다.")
        if srt.scale <= 0 or not np.isfinite(
            [srt.rotation, srt.scale, *srt.translation, *center]
        ).all():
            raise ValueError("SRT 값은 유한해야 하며 scale은 양수여야 합니다.")
        matrix = cv2.getRotationMatrix2D(center, srt.rotation, srt.scale)
        matrix[:, 2] += np.asarray(srt.translation, dtype=np.float64)
        # 정렬 후 출력 좌표를 1/4로 축소한다. 이동량도 함께 축소해야 한다.
        # 기본 SRT에서는 [[0.25, 0, 0], [0, 0.25, 0]]이 된다.
        matrix *= 0.25
        return {"matrix": matrix, "size": (256, 256), "srt": srt}

    @staticmethod
    def _to_numpy(inpt: tv_tensors.Image | tv_tensors.Mask) -> np.ndarray:
        tensor = inpt.detach().as_subclass(torch.Tensor)
        if isinstance(inpt, tv_tensors.Mask):
            tensor = tensor.to(torch.uint8)  # OpenCV는 bool 배열을 지원하지 않는다.
        else:
            tensor = tensor.permute(1, 2, 0)
        # NumPy 측 변경이 입력 텐서에 영향을 주지 않도록 반드시 복사한다.
        return tensor.contiguous().numpy().copy()

    def transform(self, inpt, params):
        is_mask = isinstance(inpt, tv_tensors.Mask)
        output = cv2.warpAffine(
            self._to_numpy(inpt),
            params["matrix"],
            params["size"],
            flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if not is_mask and self.image_augmentation is not None:
            expected_shape = output.shape
            output = self.image_augmentation(output)
            if not isinstance(output, np.ndarray) or output.dtype != np.float32:
                raise TypeError("image_augmentation은 float32 NumPy 배열을 반환해야 합니다.")
            if output.shape != expected_shape:
                raise ValueError("image_augmentation은 이미지 크기와 채널을 유지해야 합니다.")
            if not np.isfinite(output).all() or output.min() < 0 or output.max() > 1:
                raise ValueError("image_augmentation 출력은 유한한 0~1 값이어야 합니다.")

        result = torch.from_numpy(np.ascontiguousarray(output))
        if not is_mask:
            result = result.permute(2, 0, 1).contiguous()
        result = result.to(device=inpt.device, dtype=inpt.dtype)
        return tv_tensors.wrap(result, like=inpt)


from torchvision import tv_tensors
from multiprocessing import get_context

class DebugTransform(Transform):
    _transformed_types = (tv_tensors.Image, tv_tensors.Mask)

    def __init__(self):
        super().__init__()
        # 이미지/마스크 출력 여부를 worker 및 train/val/test 사이에서 공유한다.
        self._logged = get_context("spawn").Array("b", [False, False])

    def __deepcopy__(self, memo):
        # Identity 변환이며, Dataset 복사 시에도 같은 로그 상태를 유지한다.
        memo[id(self)] = self
        return self

    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.Image):
            index, kind = 0, "image"
        elif isinstance(inpt, tv_tensors.Mask):
            index, kind = 1, "mask"
        else:
            return inpt

        # 여러 worker가 동시에 접근해도 각 타입의 로그는 한 번만 출력한다.
        with self._logged.get_lock():
            if not self._logged[index]:
                logger.warning(f"===== DebugTransform called ({kind}) =====")
                logger.warning(f"type: {type(inpt)}")
                logger.warning(f"shape: {inpt.shape}")
                logger.warning(f"dtype: {inpt.dtype}")
                logger.warning(f"device: {inpt.device}")
                logger.warning(f"min: {inpt.min().item()}")
                logger.warning(f"max: {inpt.max().item()}")
                logger.warning(f"params: {params}")
                logger.warning("==============================")
                self._logged[index] = True

        return inpt

    '''
    ===== DebugTransform called (image) =====
    type: <class 'torchvision.tv_tensors._image.Image'>
    shape: torch.Size([3, 1024, 1024])
    dtype: torch.float32
    device: cpu
    min: 0.1568627506494522
    max: 0.8352941870689392
    params: {}
    ==============================
    ===== DebugTransform called (mask) =====
    type: <class 'torchvision.tv_tensors._mask.Mask'>
    shape: torch.Size([1024, 1024])
    dtype: torch.uint8
    device: cpu
    min: 0
    max: 0
    params: {}
    ==============================
    '''