from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Identity, Normalize, Resize

# train_transform = Compose([
#     Resize((256, 256), antialias=True),
#     Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

train_transform = Identity()

# eval_transform = Compose([
#     Resize((256, 256), antialias=True),
#     Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

eval_transform = Identity()

class CustomPreProcessorModel(PreProcessor):
    def __init__(
        self
    ):
        super().__init__(transform=Identity())

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
    ):
        #print(batch.image.shape, "   ", batch.image[0,0,0,10])
        batch.image = self.train_transform(batch.image)

        if False:
            import cv2
            import torch
            # 배치의 첫 이미지 한 장만 CPU로 복사
            image = batch.image[0].detach().to(
                device="cpu", dtype=torch.float32, copy=True
            )

            # Normalize 역변환: x = normalized * std + mean
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = image.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean

            # CHW RGB float → HWC RGB uint8
            image = (
                image.clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .numpy()
            )

            # OpenCV 표시용 RGB → BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow("Train batch - first image", image)
            cv2.waitKey(0)  # 키를 누르면 다음 배치로 진행
            #cv2.destroyWindow("Train batch - first image")

    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
    ):
        self._preprocess_eval_batch(batch)

    def on_test_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._preprocess_eval_batch(batch)

    def _preprocess_eval_batch(self, batch):
        # anomalib이 제공한 Mask를 그대로 사용하며 사용자 정의 평가 전처리를 적용합니다.
        if self.eval_transform:
            batch.image, batch.gt_mask = self.eval_transform(
                batch.image,
                batch.gt_mask,
            )
