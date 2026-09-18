"""1단계: 파일 목록 → 이미지 한 장 → PyTorch DataLoader."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F, InterpolationMode

ROOT = Path('/home/sh-jeon/database/MVTec-AD/mvtec_anomaly_detection/screw')


def collect_samples(root, split):
    """이미지를 아직 열지 않고, 읽는 데 필요한 목록만 만든다."""
    if split not in ('train', 'test'):
        raise ValueError('split은 train 또는 test여야 합니다.')
    root = Path(root).expanduser().resolve()
    rows = []
    for image_path in sorted((root / split).glob('*/*.png')):
        defect_type = image_path.parent.name
        label = int(defect_type != 'good')  # 정상=0, 불량=1
        if split == 'train' and label == 1:
            raise ValueError(f'학습 폴더에 불량 이미지가 있습니다: {image_path}')
        mask_path = ''
        if label == 1:
            mask = root / 'ground_truth' / defect_type / f'{image_path.stem}_mask.png'
            if not mask.is_file():
                raise FileNotFoundError(mask)
            mask_path = str(mask)
        rows.append(dict(image_path=str(image_path), mask_path=mask_path,
                         label_index=label, label=defect_type, split=split))
    if not rows:
        raise FileNotFoundError(f'PNG 이미지가 없습니다: {root / split}')
    return rows


class ScrewDataset(Dataset):
    def __init__(self, root=ROOT, split='train', image_size=(256, 256)):
        self.samples = collect_samples(root, split)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        with Image.open(row['image_path']) as source:
            image = source.convert('RGB')
            original_size = image.size
            image = F.resize(image, self.image_size, antialias=True)
            image = F.to_tensor(image)  # uint8 [0,255] → float32 [0,1], [C,H,W]

        if row['label_index'] == 0:
            mask = torch.zeros(self.image_size, dtype=torch.bool)
        else:
            with Image.open(row['mask_path']) as source:
                if source.size != original_size:
                    raise ValueError(f"이미지/마스크 크기 불일치: {row['image_path']}")
                # 마스크에는 중간값이 생기지 않도록 nearest 보간을 쓴다.
                mask = F.resize(source.convert('L'), self.image_size,
                                interpolation=InterpolationMode.NEAREST)
                mask = F.pil_to_tensor(mask).squeeze(0) > 0
        return dict(image=image, gt_label=torch.tensor(row['label_index']),
                    gt_mask=mask, image_path=row['image_path'])


if __name__ == '__main__':
    dataset = ScrewDataset(split='test')
    item = dataset[0]
    print('이미지 수:', len(dataset))
    print('경로:', item['image_path'])
    print('한 장:', item['image'].shape, '라벨:', item['gt_label'].item())
    print('마스크:', item['gt_mask'].shape, item['gt_mask'].unique())
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print('배치:', batch['image'].shape)
