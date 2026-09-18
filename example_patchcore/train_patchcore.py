'''
anomalib 공식 예제 코드
https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html?utm_source=chatgpt.com
Towards Total Recall in Industrial Anomaly Detection, CVPR 2022. PatchCore를 제안한 원논문 방법을 구현

Example CLI usage
anomalib train --config config_cli_patchcore.yaml 2>&1 | tee log.txt
config_cli_patchcore.yaml
model:
  class_path: anomalib.models.Patchcore
  init_args:
    backbone: wide_resnet50_2
    layers:
      - layer2
      - layer3
    pre_trained: true
    coreset_sampling_ratio: 0.1
    num_neighbors: 9

data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: /home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection
    category: screw

default_root_dir: /home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/patchcore-python/patchcore-cli/


This file usage
python train_patchcore.py 2>&1 | tee /home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/patchcore-python/log.txt

'''
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
    # ============================================================
    # 1. Dataset
    # CLI:
    # --data anomalib.data.MVTecAD
    # --data.root /home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection
    # --data.category screw
    # ============================================================
    datamodule = MVTecAD(
        root="/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection",
        category="screw",
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
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    # ============================================================
    # 3. Engine
    # CLI:
    # --default_root_dir ./
    # ============================================================
    engine = Engine(
        default_root_dir=Path("/home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/patchcore-python"),
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