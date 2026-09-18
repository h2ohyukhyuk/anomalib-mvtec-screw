'''
anomalib 공식 예제 코드
https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/padim.html
PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization, ICPR 2020
Example CLI usage
anomalib train --config config_wide_resnet50_2.yaml 2>&1 | tee log_wide_resnet50_2.txt
config_cli_padim.yaml
model:
  class_path: anomalib.models.Padim
  init_args:
    backbone: wide_resnet50_2
    layers:
      - layer2
      - layer3
    pre_trained: true
    n_features: 256

data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: /home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection
    category: screw

default_root_dir: /home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/padim-cli/


anomalib train --config config_resnet18.yaml 2>&1 | tee log_resnet18.txt
model:
  class_path: anomalib.models.Padim
  init_args:
    backbone: resnet18
    layers:
      - layer1
      - layer2
      - layer3
    pre_trained: true
    n_features: 100

data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: /home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection
    category: screw

default_root_dir: /home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/padim-cli/


This file usage
python train_padim.py 2>&1 | tee /home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/padim-python/log.txt

'''

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

from lightning import seed_everything


SEED = 42


def main():
    # ------------------------------------------------------------
    # 0. Seed
    # ------------------------------------------------------------
    seed_everything(SEED, workers=True)

    # ------------------------------------------------------------
    # 1. Dataset
    # CLI resolved config와 동일하게 명시
    # ------------------------------------------------------------
    datamodule = MVTecAD(
        root="/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection",
        category="screw",

        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,

        train_augmentations=None,
        val_augmentations=None,
        test_augmentations=None,
        augmentations=None,

        test_split_mode="from_dir",
        test_split_ratio=0.2,

        val_split_mode="same_as_test",
        val_split_ratio=0.5,

        seed=None,
    )

    # ------------------------------------------------------------
    # 2. PaDiM
    # CLI resolved config와 동일하게 명시
    # ------------------------------------------------------------
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        n_features=100,

        pre_processor=True,
        post_processor=True,
        evaluator=True,
        visualizer=True,
    )

    # ------------------------------------------------------------
    # 3. Engine
    # CLI resolved config에서 주요 값 명시
    # ------------------------------------------------------------
    engine = Engine(
        accelerator="auto",
        strategy="auto",
        devices="auto",
        num_nodes=1,

        default_root_dir=(
            "/home/sh-jeon/dnn_models/"
            "Implant-SNUCONE/exam-anomalib/padim-python/"
        ),

        inference_mode=True,
        use_distributed_sampler=True,

        detect_anomaly=False,
        benchmark=None,
        deterministic=None,
    )

    # ------------------------------------------------------------
    # 4. train
    #
    # PaDiM의 경우:
    #
    # fit()
    #   -> 정상 이미지 feature 추출
    #   -> spatial embedding 수집
    #   -> Gaussian fitting
    #
    # test()
    #   -> anomaly score / anomaly map
    #   -> AUROC / F1 계산
    # ------------------------------------------------------------
    results = engine.train(
        model=model,
        datamodule=datamodule,
    )

    print("\n=== Test Results ===")
    print(results)


if __name__ == "__main__":
    main()