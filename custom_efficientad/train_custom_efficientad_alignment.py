"""정렬된 screw 데이터로 기존 anomalib 2.6.2 CLI 학습 조건을 재현한다.

실행:
    /home/sh-jeon/venvs/venv-implant-anomalib/bin/python \
        /home/sh-jeon/projects_ori/anomalib-mvtec-screw/custom_efficientad/train_custom_efficientad_alignment.py

기준: efficientad-cli-alignment에 저장된 CLI config.yaml 및 log.txt.
기존 CLI는 max_epochs 미지정으로 1000 epochs를 사용했다.
이 버전의 CLI는 seed_everything 옵션을 파싱하지만 실행 경로에서 호출하지
않으므로 여기서도 seed를 고정하지 않는다. 이전 실행의 난수 상태까지
재현하는 것은 아니므로 결과 수치가 완전히 같지는 않을 수 있다.
"""

from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize


def main() -> None:
    # 이미 정렬된 이미지와 마스크를 사용한다. 추가 affine 변환이나
    # Normalize 없이 모델 기본 Resize(256x256, antialias=True)를 적용한다.
    datamodule = MVTecAD(
        root="/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection",
        category="screw_coarse_alignment",
        train_batch_size=1,
        eval_batch_size=32,
        num_workers=8,
        train_augmentations=None,
        val_augmentations=None,
        test_augmentations=None,
        augmentations=None,
        test_split_mode=TestSplitMode.FROM_DIR,
        test_split_ratio=0.2,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        val_split_ratio=0.5,
        seed=None,
    )

    # ImageNet mean/std 정규화는 EfficientAd 모델 내부에서 수행한다.
    model = EfficientAd(
        imagenet_dir="/home/sh-jeon/database/imagenette2",
        teacher_out_channels=384,
        model_size=EfficientAdModelSize.S,
        lr=0.0001,
        weight_decay=0.00001,
        padding=False,
        pad_maps=True,
        pre_processor=True,
        post_processor=True,
        evaluator=True,
        visualizer=True,
    )

    engine = Engine(
        default_root_dir=Path(
            "/home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/"
            "efficientad-python-alignment"
        ),
        accelerator="auto",
        strategy="auto",
        devices="auto",
        max_epochs=1000,
        max_steps=-1,
        check_val_every_n_epoch=1,
        deterministic=None,
        benchmark=None,
    )

    # CLI의 anomalib train과 같이 학습 후 테스트까지 수행한다.
    engine.train(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
