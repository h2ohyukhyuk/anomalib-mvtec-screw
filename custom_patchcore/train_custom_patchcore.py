'''
anomalib 공식 예제 코드
https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html?utm_source=chatgpt.com
Towards Total Recall in Industrial Anomaly Detection, CVPR 2022. PatchCore를 제안한 원논문 방법을 구현

This file usage
python train_custom_patchcore.py 2>&1 | tee /home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/patchcore-custom-python/log.txt

'''
from pathlib import Path

#from custom_mvtecad import CustomMVTecAD as MVTecAD
from custom_mvtecad_datamodule import MVTecAD
from custom_engine import CustomEngine as Engine
from custom_patchcore import CustomPatchcore as Patchcore

from lightning import seed_everything
from custom_preprocessor_model import CustomPreProcessorModel
from torchvision.transforms.v2 import Compose, Normalize, Resize

import logging
from torchvision.transforms.v2 import Transform
from torchvision import tv_tensors
from torch.utils.data import get_worker_info
import os

logger = logging.getLogger(__name__)
logger.warning("jsh this is a warning")


class MyTransform(Transform):

    def __init__(self):
        super().__init__()
        self.once = True

    def transform(self, inpt, params):

        if self.once:
            self.once = False

            worker = get_worker_info()

            if worker.id == 0:
                logger.warning("===== MyTransform called =====")
                logger.warning(f"type: {type(inpt)}")
                logger.warning(f"shape: {inpt.shape}")
                logger.warning(f"dtype: {inpt.dtype}")
                logger.warning(f"device: {inpt.device}")
                logger.warning(f"min: {inpt.min().item()}")
                logger.warning(f"max: {inpt.max().item()}")
                logger.warning(f"params: {params}")
                logger.warning("==============================")

        return inpt

def main():

    # ------------------------------------------------------------
    # 0. Seed
    # ------------------------------------------------------------
    seed_everything(42, workers=True)

    # ============================================================
    # 1. Dataset
    # CLI:
    # --data anomalib.data.MVTecAD
    # --data.root /home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection
    # --data.category screw
    # ============================================================

    train_augmentations = Compose([
            MyTransform(),
            Resize((256, 256), antialias=True),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    datamodule = MVTecAD(
        root="/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection",
        category="screw",
        train_augmentations=train_augmentations,
        val_augmentations=train_augmentations,
        test_augmentations=train_augmentations
    )

    # ============================================================
    # 2. PatchCore model
    # YAML:
    #
    # backbone: wide_resnet50_2
    # layers:
    #   - layer2
    #   - layer3
    # pre_trained: true
    # coreset_sampling_ratio: 0.1
    # num_neighbors: 9
    # ============================================================
    pre_processor_for_model = CustomPreProcessorModel()

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_processor=pre_processor_for_model,
    )

    # ============================================================
    # 3. Engine
    # CLI:
    # --default_root_dir ./
    # ============================================================
    engine = Engine(
        default_root_dir=Path("/home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/patchcore-custom-python"),
    )

    # ============================================================
    # 4. CLI "anomalib train"과 동일한 의미
    #
    # fit()
    #   -> 정상 이미지 feature extraction
    #   -> memory bank 생성
    #   -> coreset sampling
    #
    # test()
    #   -> MVTec screw test set 평가
    #   -> image/pixel AUROC, F1 계산
    # ============================================================
    engine.train(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()