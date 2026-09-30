"""Engine 확장용 클래스 (anomalib 2.6.2 기준).

부모 및 상속 계층의 Python 메서드/프로퍼티를 명시적으로 제공합니다.
기본 구현은 부모에게 위임합니다. @wraps는 Lightning의 훅 판정을 보존합니다.
메서드에 사용자 로직을 추가할 때는 해당 메서드의 @wraps도 제거하세요.
object 자체의 특수 메서드는 그대로 상속합니다.
"""

from functools import wraps
from inspect import getattr_static, signature
from anomalib.engine import Engine

_PARENT = Engine
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


class CustomEngine(Engine):
    """Engine의 동작을 유지하는 사용자 확장 지점."""

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, '__init__'))
    def __init__(self, callbacks=None, logger=None, default_root_dir='results', **kwargs):
        return super().__init__(callbacks, logger, default_root_dir, **kwargs)

    # anomalib.engine.engine.Engine
    @property
    def trainer(self):
        return super().trainer

    # anomalib.engine.engine.Engine
    @property
    def model(self):
        return super().model

    # anomalib.engine.engine.Engine
    @property
    def checkpoint_callback(self):
        return super().checkpoint_callback

    # anomalib.engine.engine.Engine
    @property
    def best_model_path(self):
        return super().best_model_path

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, '_setup_workspace'))
    def _setup_workspace(
        self,
        model,
        train_dataloaders=None,
        val_dataloaders=None,
        test_dataloaders=None,
        datamodule=None,
        dataset=None,
        versioned_dir=False,
    ):
        return super()._setup_workspace(
            model,
            train_dataloaders,
            val_dataloaders,
            test_dataloaders,
            datamodule,
            dataset,
            versioned_dir,
        )

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, '_setup_trainer'))
    def _setup_trainer(self, model):
        return super()._setup_trainer(model)

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, '_setup_anomalib_callbacks'))
    def _setup_anomalib_callbacks(self):
        return super()._setup_anomalib_callbacks()

    # anomalib.engine.engine.Engine
    @staticmethod
    @wraps(getattr_static(_PARENT, '_should_run_validation').__func__)
    def _should_run_validation(model, ckpt_path):
        return _PARENT._should_run_validation(model, ckpt_path)

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, 'fit'))
    def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None):
        return super().fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, 'validate'))
    def validate(self, model=None, dataloaders=None, ckpt_path=None, verbose=True, datamodule=None):
        return super().validate(model, dataloaders, ckpt_path, verbose, datamodule)

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, 'test'))
    def test(self, model=None, dataloaders=None, ckpt_path=None, verbose=True, datamodule=None):
        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, 'predict'))
    def predict(
        self,
        model=None,
        dataloaders=None,
        datamodule=None,
        dataset=None,
        return_predictions=None,
        ckpt_path=None,
        data_path=None,
    ):
        return super().predict(
            model,
            dataloaders,
            datamodule,
            dataset,
            return_predictions,
            ckpt_path,
            data_path,
        )

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, 'train'))
    def train(
        self,
        model,
        train_dataloaders=None,
        val_dataloaders=None,
        test_dataloaders=None,
        datamodule=None,
        ckpt_path=None,
    ):
        return super().train(
            model,
            train_dataloaders,
            val_dataloaders,
            test_dataloaders,
            datamodule,
            ckpt_path,
        )

    # anomalib.engine.engine.Engine
    @wraps(getattr_static(_PARENT, 'export'))
    def export(
        self,
        model,
        export_type,
        export_root=None,
        model_file_name='model',
        input_size=None,
        compression_type=None,
        datamodule=None,
        metric=None,
        max_drop=0.01,
        ov_args=None,
        ov_kwargs=None,
        onnx_kwargs=None,
        ckpt_path=None,
    ):
        return super().export(
            model,
            export_type,
            export_root,
            model_file_name,
            input_size,
            compression_type,
            datamodule,
            metric,
            max_drop,
            ov_args,
            ov_kwargs,
            onnx_kwargs,
            ckpt_path,
        )

    # anomalib.engine.engine.Engine
    @type(getattr_static(_PARENT, 'from_config'))
    @wraps(getattr_static(_PARENT, 'from_config').__func__)
    def from_config(cls, config_path, **kwargs):
        return super().from_config(config_path, **kwargs)
