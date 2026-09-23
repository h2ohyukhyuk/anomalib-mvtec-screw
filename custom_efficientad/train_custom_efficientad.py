'''
anomalib 공식 예제 코드
https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/efficient_ad.html
EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies, WACV 2024

anomalib train --config config.yaml 2>&1 | tee log.txt

seed_everything: 42

model:
  class_path: anomalib.models.EfficientAd
  init_args:
    imagenet_dir: /home/sh-jeon/database/imagenette2
    teacher_out_channels: 384
    model_size: s
    lr: 0.0001
    weight_decay: 0.00001
    padding: false
    pad_maps: true
    pre_processor: true
    post_processor: true
    evaluator: true
    visualizer: true

data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: /home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection
    category: screw

    train_batch_size: 1
    eval_batch_size: 32
    num_workers: 8

    train_augmentations: null
    val_augmentations: null
    test_augmentations: null
    augmentations: null

    test_split_mode: FROM_DIR
    test_split_ratio: 0.2
    val_split_mode: SAME_AS_TEST
    val_split_ratio: 0.5
    seed: null

default_root_dir: /home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/efficientad-cli/
'''