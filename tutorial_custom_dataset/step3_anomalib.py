"""3단계: anomalib 2.6.2의 Dataset / DataModule 규약에 연결한다."""
import pandas as pd
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.datamodules.base import AnomalibDataModule
from torchvision.transforms.v2 import Resize
from step1_dataset import ROOT, collect_samples


class CustomScrewDataset(AnomalibDataset):
    def __init__(self, root=ROOT, split='train', augmentations=None):
        super().__init__(augmentations=augmentations)
        self.category = 'screw'
        samples = pd.DataFrame(collect_samples(root, split))
        samples.attrs['task'] = 'segmentation'
        self.samples = samples
        # __len__, __getitem__ 및 ImageItem 생성은 부모 구현을 이용한다.


class CustomScrewDataModule(AnomalibDataModule):
    def __init__(self, root=ROOT, batch_size=4, seed=42):
        super().__init__(
            train_batch_size=batch_size, eval_batch_size=batch_size,
            num_workers=0, augmentations=Resize((256, 256), antialias=True),
            train_augmentations=Resize((256, 256), antialias=True),
            val_augmentations=Resize((256, 256), antialias=True),
            test_augmentations=Resize((256, 256), antialias=True),
            test_split_mode='from_dir', val_split_mode='from_test',
            val_split_ratio=0.5, seed=seed,
        )
        self.root = root
        self.category = 'screw'

    def prepare_data(self):
        pass  # 다운로드는 하지 않는다. 경로 검증은 collect_samples에서 한다.

    def _setup(self, _stage=None):
        # setup()은 부모가 처리한다. 우리는 _setup()에 원본 두 집합을 제공한다.
        self.train_data = CustomScrewDataset(self.root, 'train')
        self.test_data = CustomScrewDataset(self.root, 'test')
        # 부모가 test에서 val을 분리하고, 증강 및 collate_fn을 연결한다.


if __name__ == '__main__':
    dm = CustomScrewDataModule()
    dm.setup()
    print('train / val / test:', len(dm.train_data), len(dm.val_data), len(dm.test_data))
    print(dm.train_data.samples.head())
    batch = next(iter(dm.train_dataloader()))
    print('배치 타입:', type(batch).__name__)
    print('이미지:', batch.image.shape, '마스크:', batch.gt_mask.shape)
