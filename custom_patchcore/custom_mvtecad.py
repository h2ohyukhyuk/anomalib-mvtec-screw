"""MVTecAD 확장용 클래스 (anomalib 2.6.2 기준).

부모 및 상속 계층의 Python 메서드/프로퍼티를 명시적으로 제공합니다.
기본 구현은 부모에게 위임합니다. @wraps는 Lightning의 훅 판정을 보존합니다.
메서드에 사용자 로직을 추가할 때는 해당 메서드의 @wraps도 제거하세요.
object 자체의 특수 메서드는 그대로 상속합니다.
"""

from functools import wraps
from inspect import getattr_static, signature
from anomalib.data import MVTecAD

_PARENT = MVTecAD
_DEFAULTS = {}
for _base in reversed(_PARENT.__mro__[:-1]):
    for _name, _member in vars(_base).items():
        _func = getattr(_member, "__func__", _member)
        if callable(_func):
            try:
                _DEFAULTS[_name] = {
                    key: param.default
                    for key, param in signature(_func).parameters.items()
                }
            except (TypeError, ValueError):
                pass


class CustomMVTecAD(MVTecAD):
    """MVTecAD의 동작을 유지하는 사용자 확장 지점."""

    # anomalib.data.datamodules.image.mvtecad.MVTecAD
    @wraps(getattr_static(_PARENT, '__init__'))
    def __init__(
        self,
        root='./datasets/MVTecAD',
        category='bottle',
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        train_augmentations=None,
        val_augmentations=None,
        test_augmentations=None,
        augmentations=None,
        test_split_mode=_DEFAULTS['__init__']['test_split_mode'],
        test_split_ratio=0.2,
        val_split_mode=_DEFAULTS['__init__']['val_split_mode'],
        val_split_ratio=0.5,
        seed=None,
    ):
        return super().__init__(
            root,
            category,
            train_batch_size,
            eval_batch_size,
            num_workers,
            train_augmentations,
            val_augmentations,
            test_augmentations,
            augmentations,
            test_split_mode,
            test_split_ratio,
            val_split_mode,
            val_split_ratio,
            seed,
        )

    # anomalib.data.datamodules.image.mvtecad.MVTecAD
    @wraps(getattr_static(_PARENT, '_setup'))
    def _setup(self, _stage=None):
        return super()._setup(_stage)

    # anomalib.data.datamodules.image.mvtecad.MVTecAD
    @wraps(getattr_static(_PARENT, 'prepare_data'))
    def prepare_data(self):
        return super().prepare_data()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @property
    def name(self):
        return super().name

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, 'setup'))
    def setup(self, stage=None):
        return super().setup(stage)

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, '_update_augmentations'))
    def _update_augmentations(self):
        return super()._update_augmentations()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @staticmethod
    @wraps(getattr_static(_PARENT, '_update_subset_augmentations').__func__)
    def _update_subset_augmentations(dataset, augmentations, model_transform):
        return _PARENT._update_subset_augmentations(dataset, augmentations, model_transform)

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @property
    def category(self):
        return super().category

    @category.setter
    def category(self, value):
        return _PARENT.category.fset(self, value)

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @property
    def task(self):
        return super().task

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, '_create_test_split'))
    def _create_test_split(self):
        return super()._create_test_split()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, '_create_val_split'))
    def _create_val_split(self):
        return super()._create_val_split()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, 'train_dataloader'))
    def train_dataloader(self):
        return super().train_dataloader()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, 'val_dataloader'))
    def val_dataloader(self):
        return super().val_dataloader()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, 'test_dataloader'))
    def test_dataloader(self):
        return super().test_dataloader()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @wraps(getattr_static(_PARENT, 'predict_dataloader'))
    def predict_dataloader(self):
        return super().predict_dataloader()

    # anomalib.data.datamodules.base.image.AnomalibDataModule
    @type(getattr_static(_PARENT, 'from_config'))
    @wraps(getattr_static(_PARENT, 'from_config').__func__)
    def from_config(cls, config_path, **kwargs):
        return super().from_config(config_path, **kwargs)

    # lightning.pytorch.core.datamodule.LightningDataModule
    @type(getattr_static(_PARENT, 'from_datasets'))
    @wraps(getattr_static(_PARENT, 'from_datasets').__func__)
    def from_datasets(
        cls,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        predict_dataset=None,
        batch_size=1,
        num_workers=0,
        **datamodule_kwargs,
    ):
        return super().from_datasets(
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            batch_size,
            num_workers,
            **datamodule_kwargs,
        )

    # lightning.pytorch.core.datamodule.LightningDataModule
    @wraps(getattr_static(_PARENT, 'state_dict'))
    def state_dict(self):
        return super().state_dict()

    # lightning.pytorch.core.datamodule.LightningDataModule
    @wraps(getattr_static(_PARENT, 'load_state_dict'))
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    # lightning.pytorch.core.datamodule.LightningDataModule
    @wraps(getattr_static(_PARENT, 'on_exception'))
    def on_exception(self, exception):
        return super().on_exception(exception)

    # lightning.pytorch.core.datamodule.LightningDataModule
    @type(getattr_static(_PARENT, 'load_from_checkpoint'))
    @wraps(getattr_static(_PARENT, 'load_from_checkpoint').__func__)
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        weights_only=None,
        **kwargs,
    ):
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location,
            hparams_file,
            weights_only,
            **kwargs,
        )

    # lightning.pytorch.core.datamodule.LightningDataModule
    @wraps(getattr_static(_PARENT, '__str__'))
    def __str__(self):
        return super().__str__()

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'teardown'))
    def teardown(self, stage):
        return super().teardown(stage)

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'transfer_batch_to_device'))
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'on_before_batch_transfer'))
    def on_before_batch_transfer(self, batch, dataloader_idx):
        return super().on_before_batch_transfer(batch, dataloader_idx)

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'on_after_batch_transfer'))
    def on_after_batch_transfer(self, batch, dataloader_idx):
        return super().on_after_batch_transfer(batch, dataloader_idx)

    # lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    @wraps(getattr_static(_PARENT, 'save_hyperparameters'))
    def save_hyperparameters(self, *args, ignore=None, frame=None, logger=True):
        return super().save_hyperparameters(*args, ignore=ignore, frame=frame, logger=logger)

    # lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    @wraps(getattr_static(_PARENT, '_set_hparams'))
    def _set_hparams(self, hp):
        return super()._set_hparams(hp)

    # lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    @wraps(getattr_static(_PARENT, 'remove_ignored_hparams'))
    def remove_ignored_hparams(self, ignore_list):
        return super().remove_ignored_hparams(ignore_list)

    # lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    @staticmethod
    @wraps(getattr_static(_PARENT, '_to_hparams_dict').__func__)
    def _to_hparams_dict(hp):
        return _PARENT._to_hparams_dict(hp)

    # lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    @property
    def hparams(self):
        return super().hparams

    # lightning.pytorch.core.mixins.hparams_mixin.HyperparametersMixin
    @property
    def hparams_initial(self):
        return super().hparams_initial
