# anomalib MVTec Screw 실습

MVTec AD의 **screw(나사)** 데이터를 사용해 이상 탐지를 실험하고, anomalib과 PyTorch Lightning의 데이터 처리 구조를 익히는 프로젝트입니다.

PaDiM·PatchCore 실행 예제, 커스텀 PatchCore 확장 코드, Dataset/DataModule 단계별 실습, 이미지·마스크 정렬 실험을 포함합니다. 현재 코드는 로컬 환경의 절대 경로를 사용합니다.

## 환경

문서 작성 시 로컬 가상환경에서 확인한 버전입니다. 의존성 잠금 파일이나 모든 환경에서 검증된 설치 조합을 의미하지는 않습니다.

| 항목 | 확인된 버전 |
|---|---|
| Python | 3.12 계열 가상환경 |
| anomalib | 2.6.2 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| Lightning | 2.6.6 |

현재 가상환경을 활성화합니다.

```bash
cd /home/sh-jeon/projects_ori/anomalib-mvtec-screw
source /home/sh-jeon/venvs/venv-implant-anomalib/bin/activate
```

프로젝트에는 아직 `requirements.txt`나 `pyproject.toml`이 없습니다. 다른 환경에서 사용할 때는 anomalib 외에도 실행 파일이 사용하는 pandas, Pillow, OpenCV, matplotlib, NumPy 등의 의존성을 준비해야 합니다. 사전 학습 가중치가 캐시에 없으면 모델 실행 시 다운로드가 발생할 수 있습니다.

## 프로젝트 구성

```text
anomalib-mvtec-screw/
├── example_padim/
│   ├── train_padim.py                 # Python으로 PaDiM 실행
│   ├── train_padim_from_config.py     # YAML을 읽어 PaDiM 실행
│   ├── config_python_padim.yaml
│   └── config_cli_padim.yaml
├── example_patchcore/
│   ├── train_patchcore.py             # Python으로 PatchCore 실행
│   └── config_cli_patchcore.yaml
├── example_efficientad/               # EfficientAD 설정 참고 메모
├── custom_patchcore/
│   ├── train_custom_patchcore.py      # 커스텀 클래스들을 연결하는 진입점
│   ├── custom_mvtecad.py              # MVTecAD 확장
│   ├── custom_engine.py               # Engine 확장
│   ├── custom_patchcore.py            # PatchCore 확장
│   ├── custom_model_preprocessor.py   # 전처리 및 학습 배치 시각화
│   ├── custom_mvtecad_dataset.py      # 작성 중인 Dataset
│   ├── custom_mvtecad_datamodule.py    # 빈 실습 파일
│   └── my_mvtecad.py                  # 빈 실습 파일
├── tutorial_custom_dataset/           # Dataset → Lightning → anomalib 실습
│   ├── README.md
│   ├── step1_dataset.py
│   ├── step2_lightning.py
│   ├── step3_anomalib.py
│   └── step4_train.py
├── dummy_line_align.py               # 더미 이미지·마스크 회전 정렬 실험
├── main.py                           # import만 있는 준비 파일
├── preprocess_test.py                # 빈 실습 파일
└── outputs/                          # 로컬 생성 결과
```

`main.py`는 학습 진입점이 아닙니다. 아래의 모델별 실행 파일을 사용합니다.

## 데이터 준비

기본 예제의 데이터 루트는 다음 경로입니다.

```text
/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/
└── screw/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   ├── scratch_head/
    │   └── ...
    └── ground_truth/
        ├── scratch_head/
        │   └── 000_mask.png
        └── ...
```

- `train/good`: 정상 이미지로 학습합니다.
- `test/good`: 평가용 정상 이미지입니다.
- `test/<불량 유형>`: 평가용 불량 이미지입니다.
- `ground_truth/<불량 유형>`: 불량 위치를 표시하는 마스크입니다. 이미지 `000.png`에 대응하는 이름은 `000_mask.png`입니다.
- 이미지 라벨은 정상 `0`, 불량 `1`입니다. 정상 이미지의 마스크는 모두 0입니다.

기본 `MVTecAD`에는 **screw의 상위 폴더**를 `root`로 전달하고 `category="screw"`를 지정합니다. 반면 `tutorial_custom_dataset/step1_dataset.py`의 `ROOT`는 **screw 폴더 자체**를 가리킵니다.

다른 컴퓨터에서는 Python 실행 파일의 데이터 경로, YAML의 `data.init_args.root`, 결과 저장 경로인 `default_root_dir`를 환경에 맞춰야 합니다.

## 기본 모델 실행

아래 명령은 프로젝트 루트에서 실행합니다. 각 학습 명령은 별도의 실험이며, 모두 연속 실행할 필요는 없습니다.

### PaDiM

Python 설정으로 실행:

```bash
python example_padim/train_padim.py
```

YAML을 Python에서 읽어 실행:

```bash
python example_padim/train_padim_from_config.py
```

anomalib CLI로 실행:

```bash
anomalib train --config example_padim/config_cli_padim.yaml
```

### PatchCore

Python 설정으로 실행:

```bash
python example_patchcore/train_patchcore.py
```

anomalib CLI로 실행:

```bash
anomalib train --config example_patchcore/config_cli_patchcore.yaml
```

현재 PatchCore 설정은 `wide_resnet50_2`의 `layer2`, `layer3` 특징과 `coreset_sampling_ratio=0.1`, `num_neighbors=9`를 사용합니다.

### 커스텀 PatchCore

```bash
python custom_patchcore/train_custom_patchcore.py
```

현재 이 실행 파일은 `custom_mvtecad.py`의 `CustomMVTecAD`를 사용합니다. 작성 중인 `custom_mvtecad_dataset.py`와 빈 DataModule 파일은 아직 이 실행 경로에 연결되어 있지 않습니다.

`CustomPreProcessor`는 학습 배치의 첫 이미지를 OpenCV 창에 표시하고 `cv2.waitKey(0)`으로 키 입력을 기다립니다. 따라서 GUI가 있는 환경에서 실행하며, 멈춘 것처럼 보이면 이미지 창에서 키를 눌러 다음 배치로 진행합니다.

현재 커스텀 실행 파일은 DataModule의 변환에서 Resize와 Normalize를 적용하고, 커스텀 PreProcessor의 변환은 Identity로 설정합니다. 전처리를 수정할 때는 Normalize가 중복되지 않도록 두 위치를 함께 확인합니다.

### EfficientAD의 현재 상태

`example_efficientad/train_efficientad.py`는 설정 예시를 담은 문자열 메모이며, 실제 학습 실행 코드가 없습니다. `config_cli_efficientad.yaml`도 현재 비어 있습니다. 따라서 이 폴더는 완성된 실행 예제가 아닙니다. 메모에는 학습 배치 크기 `1`과 별도의 Imagenette 경로 등이 기록되어 있습니다.

## 커스텀 Dataset 학습 순서

자세한 설명은 [단계별 실습 문서](tutorial_custom_dataset/README.md)를 참고합니다.

| 단계 | 배우는 내용 | 실행 명령 |
|---|---|---|
| 1 | 파일 목록, `__len__`, `__getitem__`, 배치 | `python tutorial_custom_dataset/step1_dataset.py` |
| 2 | LightningDataModule과 데이터 분할 | `python tutorial_custom_dataset/step2_lightning.py` |
| 3 | AnomalibDataset·AnomalibDataModule 연결 | `python tutorial_custom_dataset/step3_anomalib.py` |
| 4 | 커스텀 DataModule로 PatchCore 학습·평가 | `python tutorial_custom_dataset/step4_train.py` |

1~3단계는 데이터 로딩 확인용이고, 4단계는 실제 모델 학습을 실행합니다.

역할은 다음과 같이 나뉩니다.

- **Dataset**: 이미지 한 장과 라벨·마스크를 읽습니다.
- **DataLoader**: 여러 샘플을 배치로 묶습니다.
- **DataModule**: train/val/test Dataset과 DataLoader를 준비합니다.
- **모델**: 배치로부터 정상 특징을 학습하거나 이상 점수를 계산합니다.
- **Engine**: Lightning Trainer를 통해 학습·검증·평가 흐름을 실행합니다.

### Engine 메서드

| 메서드 | 일반적인 역할 |
|---|---|
| `engine.fit()` | 학습과 설정된 검증 수행 |
| `engine.test()` | 학습된 모델을 테스트 데이터로 평가 |
| `engine.train()` | 학습·검증 후 테스트까지 실행 |

위 설명은 이 프로젝트의 PaDiM·PatchCore 사용 흐름을 기준으로 합니다. `model.train()`은 PyTorch 모델의 학습 모드 전환이며, `engine.train()`과 역할이 다릅니다.

### 데이터 변환 설정

`train_augmentations`, `val_augmentations`, `test_augmentations`는 해당 분할에 사용할 변환입니다. `augmentations`는 단계별 변환이 지정되지 않았을 때 사용하는 공통 기본값입니다.

공통 변환과 단계별 변환은 자동으로 합쳐지지 않습니다. 예를 들어 공통 Resize와 학습 전용 회전을 별도로 지정하면, 학습용 사용자 변환에는 회전이 선택됩니다. 둘 다 필요하면 학습용 변환 안에 함께 구성합니다. 모델 PreProcessor의 변환은 별도로 확인해야 합니다.

### 검증과 테스트 분할

기본 예제의 여러 설정은 `SAME_AS_TEST`를 사용해 같은 평가 이미지를 검증과 테스트에 사용합니다. 튜토리얼은 원래 평가 데이터를 겹치지 않는 val/test로 나눕니다. 따라서 두 방식의 점수를 동일한 평가 조건으로 직접 비교하면 안 됩니다.

## 정렬 전처리 실험

```bash
python dummy_line_align.py
```

실제 screw 데이터 대신 더미 선 이미지와 마스크를 생성하고, 검출한 선의 방향을 이용해 함께 회전합니다. matplotlib 창에서 원본, 에지, 정렬된 이미지와 마스크를 비교합니다. 이미지와 마스크에 동일한 공간 변환을 적용하는 방법을 살펴보는 실험입니다.

## 결과 저장과 Git 관리

기본 예제의 결과 경로는 Python 파일 또는 YAML의 `default_root_dir`에 정의되어 있습니다.

- 여러 Python 예제와 PaDiM CLI: `/home/sh-jeon/dnn_models/Implant-SNUCONE/exam-anomalib/` 아래의 모델별 폴더.
- PatchCore CLI: 현재 설정이 `default_root_dir: .`이므로 실행 디렉터리를 기준으로 저장.
- Dataset 튜토리얼 학습: `tutorial_custom_dataset/results/`.

`.gitignore`는 가상환경, Python 캐시, IDE 설정, 로컬 데이터 폴더, 학습 결과, 체크포인트, 로그 등을 제외합니다. 소스 코드와 YAML 설정은 관리 대상으로 남기며, 이미지 파일 전체를 확장자로 제외하지는 않습니다.

이미 Git에서 추적 중인 파일은 `.gitignore`에 추가해도 자동으로 추적 해제되지 않습니다.

## 확인 범위

튜토리얼 1~3단계는 로컬 screw 데이터로 배치 생성과 검증·테스트 중복 여부를 확인했습니다. 이 README 작성 작업에서는 학습이나 전체 모델 평가를 새로 실행하지 않았습니다. 실행 예제, 작성 중인 파일, 참고용 메모의 상태를 구분해 사용하세요.
