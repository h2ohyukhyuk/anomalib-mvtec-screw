from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils.path import get_datasets_dir

IMG_EXTENSIONS = (".png", ".PNG")
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

class MVTecadDataset(AnomalibDataset):

    def __init__(self,
                 root: Path | str | None = None,
                 category: str = "bottle",
                 augmentations = None,
                 split: str | Split | None = None,
                 ) -> None:
        super().__init__(augmentations=augmentations)

        root = root if root is not None else get_datasets_dir() / "MVTecAD"

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_mvtec_ad_dataset()



def make_mvtec_ad_dataset(root: str | Path | None = None) -> DataFrame:
