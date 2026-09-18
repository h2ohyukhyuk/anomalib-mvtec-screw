"""4단계: 데이터 확인이 끝난 후 직접 실행할 PatchCore 학습 예제."""
from pathlib import Path
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning import seed_everything
from step3_anomalib import CustomScrewDataModule


if __name__ == '__main__':
    seed_everything(42, workers=True)
    datamodule = CustomScrewDataModule()
    # 기본 모델 PreProcessor가 정규화를 담당한다.
    model = Patchcore()
    engine = Engine(default_root_dir=Path(__file__).parent / 'results')
    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)
