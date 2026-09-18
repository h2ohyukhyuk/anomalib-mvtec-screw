"""2단계: Dataset과 DataLoader를 LightningDataModule로 묶는다."""
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
from step1_dataset import ROOT, ScrewDataset


class ScrewDataModule(LightningDataModule):
    def __init__(self, root=ROOT, batch_size=4, seed=42):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.seed = seed

    def prepare_data(self):
        # 이미 내려받은 데이터를 사용한다. 다운로드가 필요하면 이곳에 작성한다.
        # self.train_data 같은 Dataset 객체는 setup에서 만든다.
        pass

    def setup(self, stage=None):
        if stage not in (None, 'fit', 'validate', 'test'):
            raise ValueError(f'이 학습 예제에서 지원하지 않는 stage: {stage}')
        if hasattr(self, 'train_data'):
            return
        self.train_data = ScrewDataset(self.root, 'train')
        evaluation = ScrewDataset(self.root, 'test')

        # 평가셋을 나눠서 벨리데이션 과 테스트 두 데이터셋을 만드는 과정
        # label == 0 정상, label == 1 불량, 두가지가 있는 상황
        # 정상의 반, 불량의 반 을 모아 val을 만든다.
        # 정상/불량 각각 절반을 val로 분리한다. val과 test는 겹치지 않는다.
        generator = torch.Generator().manual_seed(self.seed)
        val_indices, test_indices = [], []
        for label in (0, 1): # 정상 불량
            # 전체 인다이스에서 정상 또는 불량의 인다이스만 선택
            indices = [i for i, row in enumerate(evaluation.samples)
                       if row['label_index'] == label]
            if len(indices) < 2:
                raise ValueError('val/test 분리를 위해 정상과 불량이 각각 2장 이상 필요합니다.')
            # 선택된 인다이스 개수만큼 퍼뮤테이션 인덱스 만듬
            order = torch.randperm(len(indices), generator=generator).tolist()
            # 선택된 퍼뮤테이션 인다스로 실제 인다이스 셔플링
            shuffled = [indices[i] for i in order]
            # 셔플링된 인다이스의 반을 선택
            cut = len(shuffled) // 2
            # 정상 또는 불량 인다이스를 cut 비율로 각각 추가
            val_indices.extend(shuffled[:cut])
            test_indices.extend(shuffled[cut:])

        self.val_data = Subset(evaluation, val_indices)
        self.test_data = Subset(evaluation, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=0)


if __name__ == '__main__':
    dm = ScrewDataModule()
    dm.prepare_data()
    dm.setup('fit')
    print('train / val / test:', len(dm.train_data), len(dm.val_data), len(dm.test_data))
    print('학습 배치:', next(iter(dm.train_dataloader()))['image'].shape)
