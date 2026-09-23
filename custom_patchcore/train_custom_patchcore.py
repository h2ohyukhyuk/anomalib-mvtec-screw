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
from custom_preprocessor_data import DebugTransform, MVTecScrewAlignment

from lightning import seed_everything
from custom_preprocessor_model import CustomPreProcessorModel
from torchvision.transforms.v2 import Compose, Normalize

import logging


logger = logging.getLogger(__name__)

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
            #DebugTransform(),
            MVTecScrewAlignment(),
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