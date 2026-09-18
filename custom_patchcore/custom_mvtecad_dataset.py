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

class MVTecADDataset(AnomalibDataset):

    def __init__(self,
                 root: Path | str | None = None,
                 category: str = "bottle",
                 augmentations: Transform | None = None,
                 split: str | Split | None = None,
                 ) -> None:
        super().__init__(augmentations=augmentations)

        root = root if root is not None else get_datasets_dir() / "MVTecAD"

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_mvtec_ad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS
        )



def make_mvtec_ad_dataset(
        root: str | Path | None = None,
        split: str | Split | None = None,
        extensions: Sequence[str] | None= None,
) -> DataFrame:
    """Create MVTec AD samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``path/to/dataset/split/category/image_filename.png``
        ``path/to/dataset/ground_truth/category/mask_filename.png``

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - split: Dataset split (train/test)
            - label: Class label
            - image_path: Path to image file
            - mask_path: Path to mask file (if available)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> root = Path("./datasets/MVTecAD/bottle")
        >>> samples = make_mvtec_dataset(root, split="train")
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  datasets/MVTecAD/bottle train good  [...]/good/105.png           0
        1  datasets/MVTecAD/bottle train good  [...]/good/017.png           0

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    # ".../MVTec-AD/screw
    root = validate_path(root)
    # 카테고리 아래에 train, test, ground_truth 폴더가 같이 있는 상태
    # [(".../MVTec-AD/screw", "train", "good", "000.png"), .... ]
    # [(".../MVTec-AD/screw", "test", "manipulated_front", "000.png"), .... ]
    # [(".../MVTec-AD/screw", "ground_truth", "manipulated_front", "000.png"), .... ]
    # 모두 리스트로 뽑고
    samples_list = [(str(root), *f.parts[-3:]) for f in root.glob(r"**/*") if f.suffix in extensions]

    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    # 판다스로 변환
    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # 이미지경로를 root path + split + label + img_name.png 생성
    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # label index 생성 정상 0, 비정상 1
    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # ground_truth 마스크 자체는 학습데이터가 아니기때문에 분리하여 연결된 데이터에 path만 연결하는 과정
    # separate masks from samples, ignore_index: 정렬 후 인덱스를 다시 매깁
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
        by="image_path",
        ignore_index=True,
    )
    samples = samples[samples.split != "ground_truth"].sort_values(
        by="image_path",
        ignore_index=True,
    )

    # assign mask paths to anomalous test images
    # Series끼리 대입하면 pandas는 경우에 따라 index를 맞춰서 넣으려고 합니다.
    # 하지만 NumPy 배열로 바꾸면 순서대로 들어감
    samples["mask_path"] = None
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.image_path.to_numpy()

    # 불량샘플만 뽑아서 마스크와 대응되는지 검사
    # assert that the right mask files are associated with the right test images
    abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
    if (
            len(abnormal_samples)
            and not abnormal_samples.apply(
        lambda x: Path(x.image_path).stem in Path(x.mask_path).stem,
        axis=1,
    ).all()
    ):
        msg = (
            "Mismatch between anomalous images and ground truth masks. Make sure "
            "mask files in 'ground_truth' folder follow the same naming "
            "convention as the anomalous images (e.g. image: '000.png', "
            "mask: '000.png' or '000_mask.png')."
        )
        raise MisMatchError(msg)

    # 마스크가 하나도 없으면 classification dataset으로 보고, 마스크가 하나라도 있으면 segmentation dataset으로 판단합니다.
    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # 최종적으로 위에서 Train, Test 가 섞여 있어는데 둘 중에 하나로 분리된다.
    if split:
        split_value = split.value if isinstance(split, Split) else split
        samples = samples[samples.split == split_value].reset_index(drop=True)

    return samples
