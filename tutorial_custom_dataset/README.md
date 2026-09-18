# MVTec screw로 직접 만드는 Dataset과 DataModule

이 예제는 현재 설치된 anomalib 2.6.2와 실제 screw 폴더에 맞췄습니다.
기존 custom_patchcore/custom_mvtecad.py는 MVTecAD의 동작을 부모에게 위임합니다.
여기서는 파일을 찾는 규칙부터 직접 작성하고, 이후 anomalib의 부모 기능과 연결합니다.
한 번에 모두 외우기보다 각 파일을 실행하고 다음 단계로 넘어가세요.

## 0. 무엇을 만드는가?

새 이미지 데이터를 생성하는 것이 아니라, 기존 이미지 폴더를 모델이 읽을 수 있는
Python 객체로 만드는 실습입니다. 다음 세 가지는 서로 다른 역할입니다.

| 구성 요소 | 담당하는 일 | 이번 예제 |
|---|---|---|
| PyTorch Dataset | i번째 이미지와 정답 읽기 | ScrewDataset |
| PyTorch DataLoader | 여러 샘플을 배치로 묶기 | DataLoader(dataset, batch_size=4) |
| LightningDataModule | train/val/test Dataset과 Loader 관리 | ScrewDataModule |

Dataset은 PyTorch 클래스입니다. Lightning에서는 이 Dataset을 그대로 사용하면서
데이터 준비 과정을 LightningDataModule 안에 정리합니다.
LightningModule은 모델/학습 로직 담당이고 DataModule과 별개입니다.

데이터 흐름: 파일 경로 → Dataset[i] → DataLoader의 배치 → 모델

실제 데이터 루트는 아래 경로입니다. step1_dataset.py의 ROOT에 들어 있습니다.

```text
/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw/
  train/good/000.png
  test/good/000.png
  test/scratch_head/000.png
  ground_truth/scratch_head/000_mask.png
```

train/good는 정상 학습 이미지입니다. test에는 정상과 여러 불량 유형이 있습니다.
불량의 마스크는 같은 유형 폴더에서 이미지 이름 뒤에 `_mask`가 붙습니다.
라벨은 이미지 전체에 대한 정답(정상 0, 불량 1), 마스크는 픽셀별 정답입니다.
정상 이미지는 마스크 파일 대신 전부 0인 마스크를 만듭니다.

## 1. step1_dataset.py — 한 장을 직접 읽기

첫 번째로 collect_samples만 읽어보세요. 이 함수는 이미지 내용을 읽지 않습니다.
파일 경로, 정상/불량, 마스크 경로를 리스트에 기록합니다. 각 행이 한 이미지입니다.

그 다음 ScrewDataset의 세 메서드를 확인하세요.

1. `__init__`: 파일 목록과 출력 이미지 크기를 저장합니다.
2. `__len__`: 목록 길이를 반환합니다. `len(dataset)`이 호출합니다.
3. `__getitem__`: 목록에서 한 행을 골라 실제 이미지를 읽습니다. `dataset[0]`이 호출합니다.

```python
dataset = ScrewDataset(split='test')
item = dataset[0]
```

위 코드의 두 번째 줄에서 이미지가 열립니다. 전체 이미지를 생성자에서 메모리에
올리지 않기 때문에 데이터가 늘어나도 목록을 먼저 준비할 수 있습니다.

이미지는 RGB로 통일하고 256×256으로 변환합니다. `to_tensor`는 채널 순서를
[H,W,C]에서 [C,H,W]로 바꾸고 0~255 정수를 0~1 실수로 변환합니다.
이는 ImageNet 평균/표준편차 정규화와 다른 과정입니다.
마스크는 nearest 보간으로 크기를 바꾸고 bool 값으로 만듭니다.
이미지와 마스크의 원본 크기가 다르면 오류를 내도록 했습니다.

```python
loader = DataLoader(dataset, batch_size=4, num_workers=0)
batch = next(iter(loader))
```

DataLoader는 여러 번 __getitem__을 호출하고, 반환된 딕셔너리의 텐서를 쌓습니다.
한 장의 image는 [3,256,256], 배치의 image는 [4,3,256,256]입니다.
마지막 배치는 남은 장수에 따라 4보다 작을 수 있습니다.
처음에는 num_workers=0을 써서 오류와 호출 흐름을 한 프로세스에서 확인합니다.

직접 해보기: batch_size를 2로 바꾸고 출력에서 어떤 차원이 변하는지 확인하세요.
또 train과 test의 gt_label을 확인해 보세요. 학습에는 정상만 들어갑니다.

## 2. step2_lightning.py — Lightning에 데이터 준비 순서 알려주기

공식 MNIST DataModule 예제의 역할 구분을 screw 데이터에 적용했습니다.

- `__init__`: 경로, 배치 크기, seed 같은 설정을 저장합니다.
- `prepare_data`: 다운로드/압축 해제 같은 디스크 준비 작업입니다. 이미 데이터가 있으므로 pass입니다.
- `setup`: Dataset 객체와 train/val/test 분할을 준비합니다.
- `train_dataloader`, `val_dataloader`, `test_dataloader`: 해당 Dataset을 감싼 Loader를 반환합니다.

prepare_data에서 self.train_data를 만들지 않는 이유는 분산 학습에서 이 훅이
모든 프로세스에서 실행되는 것이 아니기 때문입니다. 각 프로세스에 필요한
Dataset 객체는 setup에서 생성합니다.

```python
dm = ScrewDataModule()
dm.prepare_data()
dm.setup('fit')
batch = next(iter(dm.train_dataloader()))
```

이렇게 모델 없이 데이터 부분만 확인할 수 있습니다. 일반적인 Lightning 학습에서는
`trainer.fit(model, datamodule=dm)`이 필요한 훅을 호출합니다.
이 단계의 딕셔너리 배치는 일반 Lightning 모델이 해석하도록 만든 형식입니다.
anomalib 모델에 연결할 때는 다음 단계의 ImageBatch 규약을 사용합니다.

공식 예제는 stage에 따라 Dataset 생성을 나눕니다. 이 작은 예제에서는 모든 분할을
함께 만들고 반복 호출에는 재사용합니다. 그래서 setup('test')만 호출해도 동작합니다.

MVTec에는 별도 val 폴더가 없습니다. 여기서는 train/good를 정상 학습에 모두 쓰고,
원래 test를 정상/불량 각각 절반씩 나누어 val과 최종 test로 사용합니다.
seed는 같은 분할을 재현하는 용도이며 검증·테스트에는 shuffle을 사용하지 않습니다.
MNIST처럼 분류 손실을 학습하는 예제를 그대로 복사하면 train에 불량 클래스가 없다는
특성을 놓칠 수 있습니다. 이번 학습 모델은 정상 특징을 기억하는 PatchCore입니다.

## 3. step3_anomalib.py — anomalib이 기대하는 형식 연결하기

상속 관계는 다음과 같습니다.

```text
PyTorch Dataset → AnomalibDataset → CustomScrewDataset
LightningDataModule → AnomalibDataModule → CustomScrewDataModule
```

CustomScrewDataset은 collect_samples의 목록을 pandas DataFrame으로 바꾸어
self.samples에 넣습니다. image_path, mask_path, label_index, split 등이 들어갑니다.
`samples.attrs['task'] = 'segmentation'`은 마스크를 사용하는 작업이라는 뜻입니다.

이제 부모 AnomalibDataset의 __getitem__이 이미지/마스크를 읽고 ImageItem을
반환합니다. 1단계에서 직접 작성했던 부분을 라이브러리가 처리하는 것입니다.
정상 마스크 생성과 증강 적용도 부모 구현에 들어 있습니다.

CustomScrewDataModule에서는 **setup 대신 _setup을 작성**합니다.
부모 setup이 _setup을 부른 뒤 val 분리와 증강 연결 등을 처리하기 때문입니다.
train_data와 test_data를 제공하면 val_split_mode='from_test'에 따라 val_data가 생깁니다.
SAME_AS_TEST처럼 같은 이미지를 검증과 최종 평가에 반복 사용하는 설정은 쓰지 않았습니다.
따라서 기존 실험과 평가 이미지 구성이 달라져 점수를 직접 동일 조건으로 비교하면 안 됩니다.

anomalib DataLoader는 부모가 지정한 collate_fn을 통해 ImageItem들을 ImageBatch로
묶습니다. 직접 DataLoader를 만들 때는 `collate_fn=dataset.collate_fn`을 지정해야 합니다.

```python
batch = next(iter(dm.train_dataloader()))
print(batch.image.shape)   # 딕셔너리 접근 대신 속성 접근
print(batch.gt_label)
print(batch.gt_mask.shape)
```

2단계는 직접 작성한 분할, 3단계는 anomalib의 분할 함수를 사용합니다.
둘 다 겹치지 않지만, 반올림 방식과 난수 구현 때문에 같은 이미지 구성은 아닙니다.

온라인 가이드에 transform 인자가 보이더라도, 현재 설치된 2.6.2 생성자의 인자는
augmentations입니다. 이 예제는 실제 설치 코드의 서명과 실행 결과를 기준으로 작성했습니다.

## 4. step4_train.py — PatchCore에 연결하기

데이터 세 단계를 이해한 다음 이 파일을 실행하세요. 모델 다운로드와 실제 학습이
발생할 수 있습니다. 이 예제 준비 과정에서는 전체 학습은 실행하지 않았습니다.

`Engine.fit(model, datamodule=...)`에는 커스텀 DataModule을 넣습니다.
모델의 기본 PreProcessor가 정규화를 맡으므로 Dataset에 Normalize를 추가하지 않았습니다.
학습 후 Engine.test는 분리해 둔 test 데이터로 평가합니다.
학습 결과는 이 예제 디렉터리의 results 아래에 저장됩니다.

## 실행 순서

실제 프로젝트 폴더에서 아래처럼 실행합니다.

```bash
cd /home/sh-jeon/projects_ori/anomalib-mvtec-screw
source /home/sh-jeon/venvs/venv-implant-anomalib/bin/activate
python tutorial_custom_dataset/step1_dataset.py
python tutorial_custom_dataset/step2_lightning.py
python tutorial_custom_dataset/step3_anomalib.py
# 데이터 확인이 끝난 뒤 학습을 실행할 때:
python tutorial_custom_dataset/step4_train.py
```

직접 작성하며 배울 때는 step1의 코드를 작은 덩어리씩 입력한 뒤 실행해 보세요.
처음 목표는 학습 성공보다 `dataset[0]`과 `next(iter(loader))`의 차이를 이해하는 것입니다.
나중에 자신만의 데이터 폴더를 사용하려면 먼저 collect_samples의 경로/라벨 규칙을 바꿉니다.
다양한 크기의 이미지라면 같은 출력 크기로 맞추고, 회전/크롭 증강을 추가할 때는
이미지와 마스크에 동일한 공간 변환을 적용해야 합니다.

## 참고한 공식 자료

- [Lightning의 MNIST DataModule 실습](https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/datamodules.html): prepare_data, setup, Loader의 역할 비교.
- [anomalib 커스텀 Dataset 가이드](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/datasets.html): samples 표와 Dataset 상속 구조.
- 설치된 anomalib 2.6.2의 data/datasets/base/image.py, data/datamodules/base/image.py,
  data/datamodules/image/mvtecad.py: 정확한 생성자와 _setup, 배치 연결 구현 확인.
