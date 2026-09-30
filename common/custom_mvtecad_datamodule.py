import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from custom_mvtecad_dataset import MVTecADDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract
from anomalib.utils.path import resolve_dataset_root

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="mvtecad",
    url="https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/"
    "download/420938113-1629960298/mvtec_anomaly_detection.tar.xz",
    hashsum="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
)

class MVTecAD(AnomalibDataModule):
    def __init__(
            self,
    root: Path | str | None = "./datasets/MVTecAD",
    category: str = "bottle",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 8,
    train_augmentations: Transform | None = None,
    val_augmentations: Transform | None = None,
    test_augmentations: Transform | None = None,
    augmentations: Transform | None = None,
    test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
    test_split_ratio: float | int = 0.2,
    val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
    val_split_ratio: float | int= 0.5,
    seed:int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        root = resolve_dataset_root(root, "MVTecAD")
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = MVTecADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTecADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.

        Example:
            Assume the dataset is not available on the file system::

                >>> datamodule = MVTecAD(
                ...     root="./datasets/MVTecAD",
                ...     category="bottle"
                ... )
                >>> datamodule.prepare_data()

            Directory structure after download::

                datasets/
                └── MVTecAD/
                    ├── bottle/
                    ├── cable/
                    └── ...
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)