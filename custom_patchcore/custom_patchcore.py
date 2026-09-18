"""Patchcore 확장용 클래스 (anomalib 2.6.2 기준).

부모 및 상속 계층의 Python 메서드/프로퍼티를 명시적으로 제공합니다.
기본 구현은 부모에게 위임합니다. @wraps는 Lightning의 훅 판정을 보존합니다.
메서드에 사용자 로직을 추가할 때는 해당 메서드의 @wraps도 제거하세요.
object 자체의 특수 메서드는 그대로 상속합니다.
PyTorch get_extra_state/set_extra_state는 부모가 미구현이며 함수 identity를
검사하므로 별칭으로 보존합니다. 추가 상태를 사용할 때 두 메서드를 함께 구현하세요.
"""

from functools import wraps
from inspect import getattr_static, signature
from anomalib.models import Patchcore

_PARENT = Patchcore
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


class CustomPatchcore(Patchcore):
    """Patchcore의 동작을 유지하는 사용자 확장 지점."""

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @wraps(getattr_static(_PARENT, '__init__'))
    def __init__(
        self,
        backbone='wide_resnet50_2',
        layers=_DEFAULTS['__init__']['layers'],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        precision=_DEFAULTS['__init__']['precision'],
        pre_processor=False,
        post_processor=True,
        evaluator=True,
        visualizer=True,
    ):
        super().__init__(
            backbone,
            layers,
            pre_trained,
            coreset_sampling_ratio,
            num_neighbors,
            precision,
            pre_processor,
            post_processor,
            evaluator,
            visualizer,
        )

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @type(getattr_static(_PARENT, 'configure_pre_processor'))
    @wraps(getattr_static(_PARENT, 'configure_pre_processor').__func__)
    def configure_pre_processor(cls, image_size=None, center_crop_size=None):
        return super().configure_pre_processor(image_size, center_crop_size)

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @staticmethod
    @wraps(getattr_static(_PARENT, 'configure_optimizers').__func__)
    def configure_optimizers():
        return _PARENT.configure_optimizers()

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @wraps(getattr_static(_PARENT, 'training_step'))
    def training_step(self, batch, *args, **kwargs):
        return super().training_step(batch, *args, **kwargs)

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @wraps(getattr_static(_PARENT, 'fit'))
    def fit(self):
        return super().fit()

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @wraps(getattr_static(_PARENT, 'validation_step'))
    def validation_step(self, batch, *args, **kwargs):
        return super().validation_step(batch, *args, **kwargs)

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @property
    def trainer_arguments(self):
        return super().trainer_arguments

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @property
    def learning_type(self):
        return super().learning_type

    # anomalib.models.image.patchcore.lightning_model.Patchcore
    @staticmethod
    @wraps(getattr_static(_PARENT, 'configure_post_processor').__func__)
    def configure_post_processor():
        return _PARENT.configure_post_processor()

    # anomalib.models.components.base.memory_bank_module.MemoryBankMixin
    @wraps(getattr_static(_PARENT, 'on_validation_start'))
    def on_validation_start(self):
        return super().on_validation_start()

    # anomalib.models.components.base.memory_bank_module.MemoryBankMixin
    @wraps(getattr_static(_PARENT, 'on_train_epoch_end'))
    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @property
    def name(self):
        return super().name

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @wraps(getattr_static(_PARENT, 'configure_callbacks'))
    def configure_callbacks(self):
        return super().configure_callbacks()

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @wraps(getattr_static(_PARENT, 'forward'))
    def forward(self, batch, *args, **kwargs):
        return super().forward(batch, *args, **kwargs)

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @wraps(getattr_static(_PARENT, 'predict_step'))
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return super().predict_step(batch, batch_idx, dataloader_idx)

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @wraps(getattr_static(_PARENT, 'test_step'))
    def test_step(self, batch, batch_idx, *args, **kwargs):
        return super().test_step(batch, batch_idx, *args, **kwargs)

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @staticmethod
    @wraps(getattr_static(_PARENT, '_resolve_component').__func__)
    def _resolve_component(component, component_type, default_callable):
        return _PARENT._resolve_component(component, component_type, default_callable)

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @staticmethod
    @wraps(getattr_static(_PARENT, 'configure_evaluator').__func__)
    def configure_evaluator():
        return _PARENT.configure_evaluator()

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @type(getattr_static(_PARENT, 'configure_visualizer'))
    @wraps(getattr_static(_PARENT, 'configure_visualizer').__func__)
    def configure_visualizer(cls):
        return super().configure_visualizer()

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @property
    def input_size(self):
        return super().input_size

    # anomalib.models.components.base.anomalib_module.AnomalibModule
    @type(getattr_static(_PARENT, 'from_config'))
    @wraps(getattr_static(_PARENT, 'from_config').__func__)
    def from_config(cls, config_path, **kwargs):
        return super().from_config(config_path, **kwargs)

    # anomalib.models.components.base.export_mixin.mixin.ExportMixin
    @wraps(getattr_static(_PARENT, 'to_torch'))
    def to_torch(self, export_root, model_file_name='model'):
        return super().to_torch(export_root, model_file_name)

    # anomalib.models.components.base.export_mixin.mixin.ExportMixin
    @wraps(getattr_static(_PARENT, 'to_onnx'))
    def to_onnx(self, export_root, model_file_name='model', input_size=None, **kwargs):
        return super().to_onnx(export_root, model_file_name, input_size, **kwargs)

    # anomalib.models.components.base.export_mixin.mixin.ExportMixin
    @wraps(getattr_static(_PARENT, 'to_openvino'))
    def to_openvino(
        self,
        export_root,
        model_file_name='model',
        input_size=None,
        compression_type=None,
        datamodule=None,
        metric=None,
        task=None,
        max_drop=0.01,
        ov_kwargs=None,
        onnx_kwargs=None,
    ):
        return super().to_openvino(
            export_root,
            model_file_name,
            input_size,
            compression_type,
            datamodule,
            metric,
            task,
            max_drop,
            ov_kwargs,
            onnx_kwargs,
        )

    # anomalib.models.components.base.export_mixin.mixin.ExportMixin
    @wraps(getattr_static(_PARENT, '_compress_ov_model'))
    def _compress_ov_model(
        self,
        model,
        compression_type=None,
        datamodule=None,
        metric=None,
        task=None,
        max_drop=0.01,
    ):
        return super()._compress_ov_model(model, compression_type, datamodule, metric, task, max_drop)

    # anomalib.models.components.base.export_mixin.mixin.ExportMixin
    @staticmethod
    @wraps(getattr_static(_PARENT, '_post_training_quantization_ov').__func__)
    def _post_training_quantization_ov(model, datamodule=None):
        return _PARENT._post_training_quantization_ov(model, datamodule)

    # anomalib.models.components.base.export_mixin.mixin.ExportMixin
    @staticmethod
    @wraps(getattr_static(_PARENT, '_accuracy_control_quantization_ov').__func__)
    def _accuracy_control_quantization_ov(model, datamodule=None, metric=None, task=None, max_drop=0.01):
        return _PARENT._accuracy_control_quantization_ov(model, datamodule, metric, task, max_drop)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'optimizers'))
    def optimizers(self, use_pl_optimizer=True):
        return super().optimizers(use_pl_optimizer)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'lr_schedulers'))
    def lr_schedulers(self):
        return super().lr_schedulers()

    # lightning.pytorch.core.module.LightningModule
    @property
    def trainer(self):
        return super().trainer

    @trainer.setter
    def trainer(self, value):
        return _PARENT.trainer.fset(self, value)

    # lightning.pytorch.core.module.LightningModule
    @property
    def fabric(self):
        return super().fabric

    @fabric.setter
    def fabric(self, value):
        return _PARENT.fabric.fset(self, value)

    # lightning.pytorch.core.module.LightningModule
    @property
    def example_input_array(self):
        return super().example_input_array

    @example_input_array.setter
    def example_input_array(self, value):
        return _PARENT.example_input_array.fset(self, value)

    # lightning.pytorch.core.module.LightningModule
    @property
    def current_epoch(self):
        return super().current_epoch

    # lightning.pytorch.core.module.LightningModule
    @property
    def global_step(self):
        return super().global_step

    # lightning.pytorch.core.module.LightningModule
    @property
    def global_rank(self):
        return super().global_rank

    # lightning.pytorch.core.module.LightningModule
    @property
    def local_rank(self):
        return super().local_rank

    # lightning.pytorch.core.module.LightningModule
    @property
    def on_gpu(self):
        return super().on_gpu

    # lightning.pytorch.core.module.LightningModule
    @property
    def automatic_optimization(self):
        return super().automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, value):
        return _PARENT.automatic_optimization.fset(self, value)

    # lightning.pytorch.core.module.LightningModule
    @property
    def strict_loading(self):
        return super().strict_loading

    @strict_loading.setter
    def strict_loading(self, value):
        return _PARENT.strict_loading.fset(self, value)

    # lightning.pytorch.core.module.LightningModule
    @property
    def logger(self):
        return super().logger

    # lightning.pytorch.core.module.LightningModule
    @property
    def loggers(self):
        return super().loggers

    # lightning.pytorch.core.module.LightningModule
    @property
    def device_mesh(self):
        return super().device_mesh

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, '_call_batch_hook'))
    def _call_batch_hook(self, hook_name, *args):
        return super()._call_batch_hook(hook_name, *args)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, '_on_before_batch_transfer'))
    def _on_before_batch_transfer(self, batch, dataloader_idx=0):
        return super()._on_before_batch_transfer(batch, dataloader_idx)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, '_apply_batch_transfer_handler'))
    def _apply_batch_transfer_handler(self, batch, device=None, dataloader_idx=0):
        return super()._apply_batch_transfer_handler(batch, device, dataloader_idx)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'print'))
    def print(self, *args, **kwargs):
        return super().print(*args, **kwargs)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'log'))
    def log(
        self,
        name,
        value,
        prog_bar=False,
        logger=None,
        on_step=None,
        on_epoch=None,
        reduce_fx='mean',
        enable_graph=False,
        sync_dist=False,
        sync_dist_group=None,
        add_dataloader_idx=True,
        batch_size=None,
        metric_attribute=None,
        rank_zero_only=False,
    ):
        return super().log(
            name,
            value,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            metric_attribute,
            rank_zero_only,
        )

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'log_dict'))
    def log_dict(
        self,
        dictionary,
        prog_bar=False,
        logger=None,
        on_step=None,
        on_epoch=None,
        reduce_fx='mean',
        enable_graph=False,
        sync_dist=False,
        sync_dist_group=None,
        add_dataloader_idx=True,
        batch_size=None,
        rank_zero_only=False,
    ):
        return super().log_dict(
            dictionary,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            rank_zero_only,
        )

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, '_log_dict_through_fabric'))
    def _log_dict_through_fabric(self, dictionary, logger=None):
        return super()._log_dict_through_fabric(dictionary, logger)

    # lightning.pytorch.core.module.LightningModule
    @staticmethod
    @wraps(getattr_static(_PARENT, '_LightningModule__check_not_nested').__func__)
    def _LightningModule__check_not_nested(value, name):
        return _PARENT._LightningModule__check_not_nested(value, name)

    # lightning.pytorch.core.module.LightningModule
    @staticmethod
    @wraps(getattr_static(_PARENT, '_LightningModule__check_allowed').__func__)
    def _LightningModule__check_allowed(v, name, value):
        return _PARENT._LightningModule__check_allowed(v, name, value)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, '_LightningModule__to_tensor'))
    def _LightningModule__to_tensor(self, value, name):
        return super()._LightningModule__to_tensor(value, name)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'all_gather'))
    def all_gather(self, data, group=None, sync_grads=False):
        return super().all_gather(data, group, sync_grads)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'manual_backward'))
    def manual_backward(self, loss, *args, **kwargs):
        return super().manual_backward(loss, *args, **kwargs)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'backward'))
    def backward(self, loss, *args, **kwargs):
        return super().backward(loss, *args, **kwargs)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'toggle_optimizer'))
    def toggle_optimizer(self, optimizer):
        return super().toggle_optimizer(optimizer)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'untoggle_optimizer'))
    def untoggle_optimizer(self, optimizer):
        return super().untoggle_optimizer(optimizer)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'toggled_optimizer'))
    def toggled_optimizer(self, optimizer):
        return super().toggled_optimizer(optimizer)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'clip_gradients'))
    def clip_gradients(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        return super().clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'configure_gradient_clipping'))
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'lr_scheduler_step'))
    def lr_scheduler_step(self, scheduler, metric):
        return super().lr_scheduler_step(scheduler, metric)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'optimizer_step'))
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'optimizer_zero_grad'))
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        return super().optimizer_zero_grad(epoch, batch_idx, optimizer)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'freeze'))
    def freeze(self):
        return super().freeze()

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'unfreeze'))
    def unfreeze(self):
        return super().unfreeze()

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, '_verify_is_manual_optimization'))
    def _verify_is_manual_optimization(self, fn_name):
        return super()._verify_is_manual_optimization(fn_name)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'to_torchscript'))
    def to_torchscript(self, file_path=None, method='script', example_inputs=None, **kwargs):
        return super().to_torchscript(file_path, method, example_inputs, **kwargs)

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, 'to_tensorrt'))
    def to_tensorrt(
        self,
        file_path=None,
        input_sample=None,
        ir='default',
        output_format='exported_program',
        retrace=False,
        default_device='cuda',
        **compile_kwargs,
    ):
        return super().to_tensorrt(
            file_path,
            input_sample,
            ir,
            output_format,
            retrace,
            default_device,
            **compile_kwargs,
        )

    # lightning.pytorch.core.module.LightningModule
    @type(getattr_static(_PARENT, 'load_from_checkpoint'))
    @wraps(getattr_static(_PARENT, 'load_from_checkpoint').__func__)
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        weights_only=None,
        **kwargs,
    ):
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            weights_only,
            **kwargs,
        )

    # lightning.pytorch.core.module.LightningModule
    @wraps(getattr_static(_PARENT, '__getstate__'))
    def __getstate__(self):
        return super().__getstate__()

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @property
    def dtype(self):
        return super().dtype

    @dtype.setter
    def dtype(self, value):
        return _PARENT.dtype.fset(self, value)

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @property
    def device(self):
        return super().device

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @wraps(getattr_static(_PARENT, 'to'))
    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @wraps(getattr_static(_PARENT, 'cuda'))
    def cuda(self, device=None):
        return super().cuda(device)

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @wraps(getattr_static(_PARENT, 'cpu'))
    def cpu(self):
        return super().cpu()

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @wraps(getattr_static(_PARENT, 'type'))
    def type(self, dst_type):
        return super().type(dst_type)

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @wraps(getattr_static(_PARENT, 'float'))
    def float(self):
        return super().float()

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @wraps(getattr_static(_PARENT, 'double'))
    def double(self):
        return super().double()

    # lightning.fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin
    @wraps(getattr_static(_PARENT, 'half'))
    def half(self):
        return super().half()

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

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_fit_start'))
    def on_fit_start(self):
        return super().on_fit_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_fit_end'))
    def on_fit_end(self):
        return super().on_fit_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_train_start'))
    def on_train_start(self):
        return super().on_train_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_train_end'))
    def on_train_end(self):
        return super().on_train_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_end'))
    def on_validation_end(self):
        return super().on_validation_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_start'))
    def on_test_start(self):
        return super().on_test_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_end'))
    def on_test_end(self):
        return super().on_test_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_predict_start'))
    def on_predict_start(self):
        return super().on_predict_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_predict_end'))
    def on_predict_end(self):
        return super().on_predict_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_train_batch_start'))
    def on_train_batch_start(self, batch, batch_idx):
        return super().on_train_batch_start(batch, batch_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_train_batch_end'))
    def on_train_batch_end(self, outputs, batch, batch_idx):
        return super().on_train_batch_end(outputs, batch, batch_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_batch_start'))
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_batch_end'))
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_batch_start'))
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        return super().on_test_batch_start(batch, batch_idx, dataloader_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_batch_end'))
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_predict_batch_start'))
    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        return super().on_predict_batch_start(batch, batch_idx, dataloader_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_predict_batch_end'))
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        return super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_model_zero_grad'))
    def on_validation_model_zero_grad(self):
        return super().on_validation_model_zero_grad()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_model_eval'))
    def on_validation_model_eval(self):
        return super().on_validation_model_eval()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_model_train'))
    def on_validation_model_train(self):
        return super().on_validation_model_train()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_model_eval'))
    def on_test_model_eval(self):
        return super().on_test_model_eval()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_model_train'))
    def on_test_model_train(self):
        return super().on_test_model_train()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_predict_model_eval'))
    def on_predict_model_eval(self):
        return super().on_predict_model_eval()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_train_epoch_start'))
    def on_train_epoch_start(self):
        return super().on_train_epoch_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_epoch_start'))
    def on_validation_epoch_start(self):
        return super().on_validation_epoch_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_validation_epoch_end'))
    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_epoch_start'))
    def on_test_epoch_start(self):
        return super().on_test_epoch_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_test_epoch_end'))
    def on_test_epoch_end(self):
        return super().on_test_epoch_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_predict_epoch_start'))
    def on_predict_epoch_start(self):
        return super().on_predict_epoch_start()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_predict_epoch_end'))
    def on_predict_epoch_end(self):
        return super().on_predict_epoch_end()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_before_zero_grad'))
    def on_before_zero_grad(self, optimizer):
        return super().on_before_zero_grad(optimizer)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_before_backward'))
    def on_before_backward(self, loss):
        return super().on_before_backward(loss)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_after_backward'))
    def on_after_backward(self):
        return super().on_after_backward()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'on_before_optimizer_step'))
    def on_before_optimizer_step(self, optimizer):
        return super().on_before_optimizer_step(optimizer)

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'configure_sharded_model'))
    def configure_sharded_model(self):
        return super().configure_sharded_model()

    # lightning.pytorch.core.hooks.ModelHooks
    @wraps(getattr_static(_PARENT, 'configure_model'))
    def configure_model(self):
        return super().configure_model()

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'prepare_data'))
    def prepare_data(self):
        return super().prepare_data()

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'setup'))
    def setup(self, stage):
        return super().setup(stage)

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'teardown'))
    def teardown(self, stage):
        return super().teardown(stage)

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'train_dataloader'))
    def train_dataloader(self):
        return super().train_dataloader()

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'test_dataloader'))
    def test_dataloader(self):
        return super().test_dataloader()

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'val_dataloader'))
    def val_dataloader(self):
        return super().val_dataloader()

    # lightning.pytorch.core.hooks.DataHooks
    @wraps(getattr_static(_PARENT, 'predict_dataloader'))
    def predict_dataloader(self):
        return super().predict_dataloader()

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

    # lightning.pytorch.core.hooks.CheckpointHooks
    @wraps(getattr_static(_PARENT, 'on_load_checkpoint'))
    def on_load_checkpoint(self, checkpoint):
        return super().on_load_checkpoint(checkpoint)

    # lightning.pytorch.core.hooks.CheckpointHooks
    @wraps(getattr_static(_PARENT, 'on_save_checkpoint'))
    def on_save_checkpoint(self, checkpoint):
        return super().on_save_checkpoint(checkpoint)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_buffer'))
    def register_buffer(self, name, tensor, persistent=True):
        return super().register_buffer(name, tensor, persistent)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_parameter'))
    def register_parameter(self, name, param):
        return super().register_parameter(name, param)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'add_module'))
    def add_module(self, name, module):
        return super().add_module(name, module)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_module'))
    def register_module(self, name, module):
        return super().register_module(name, module)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'get_submodule'))
    def get_submodule(self, target):
        return super().get_submodule(target)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'set_submodule'))
    def set_submodule(self, target, module, strict=False):
        return super().set_submodule(target, module, strict)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'get_parameter'))
    def get_parameter(self, target):
        return super().get_parameter(target)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'get_buffer'))
    def get_buffer(self, target):
        return super().get_buffer(target)

    # torch.nn.modules.module.Module
    # 단순 super() 오버라이드는 PyTorch 체크포인트 저장/로딩을 깨뜨립니다.
    get_extra_state = _PARENT.get_extra_state

    # torch.nn.modules.module.Module
    # 단순 super() 오버라이드는 PyTorch 체크포인트 저장/로딩을 깨뜨립니다.
    set_extra_state = _PARENT.set_extra_state

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_apply'))
    def _apply(self, fn, recurse=True):
        return super()._apply(fn, recurse)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'apply'))
    def apply(self, fn):
        return super().apply(fn)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'ipu'))
    def ipu(self, device=None):
        return super().ipu(device)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'xpu'))
    def xpu(self, device=None):
        return super().xpu(device)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'mtia'))
    def mtia(self, device=None):
        return super().mtia(device)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'bfloat16'))
    def bfloat16(self):
        return super().bfloat16()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'to_empty'))
    def to_empty(self, *, device, recurse=True):
        return super().to_empty(device=device, recurse=recurse)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_full_backward_pre_hook'))
    def register_full_backward_pre_hook(self, hook, prepend=False):
        return super().register_full_backward_pre_hook(hook, prepend)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_backward_hook'))
    def register_backward_hook(self, hook):
        return super().register_backward_hook(hook)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_full_backward_hook'))
    def register_full_backward_hook(self, hook, prepend=False):
        return super().register_full_backward_hook(hook, prepend)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_get_backward_hooks'))
    def _get_backward_hooks(self):
        return super()._get_backward_hooks()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_get_backward_pre_hooks'))
    def _get_backward_pre_hooks(self):
        return super()._get_backward_pre_hooks()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_maybe_warn_non_full_backward_hook'))
    def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn):
        return super()._maybe_warn_non_full_backward_hook(inputs, result, grad_fn)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_forward_pre_hook'))
    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        return super().register_forward_pre_hook(hook, prepend=prepend, with_kwargs=with_kwargs)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_forward_hook'))
    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        return super().register_forward_hook(
            hook,
            prepend=prepend,
            with_kwargs=with_kwargs,
            always_call=always_call,
        )

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_slow_forward'))
    def _slow_forward(self, *input, **kwargs):
        return super()._slow_forward(*input, **kwargs)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_wrapped_call_impl'))
    def _wrapped_call_impl(self, *args, **kwargs):
        return super()._wrapped_call_impl(*args, **kwargs)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_call_impl'))
    def _call_impl(self, *args, **kwargs):
        return super()._call_impl(*args, **kwargs)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '__call__'))
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '__setstate__'))
    def __setstate__(self, state):
        return super().__setstate__(state)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '__getattr__'))
    def __getattr__(self, name):
        return super().__getattr__(name)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '__setattr__'))
    def __setattr__(self, name, value):
        return super().__setattr__(name, value)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '__delattr__'))
    def __delattr__(self, name):
        return super().__delattr__(name)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_register_state_dict_hook'))
    def _register_state_dict_hook(self, hook):
        return super()._register_state_dict_hook(hook)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_state_dict_post_hook'))
    def register_state_dict_post_hook(self, hook):
        return super().register_state_dict_post_hook(hook)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_state_dict_pre_hook'))
    def register_state_dict_pre_hook(self, hook):
        return super().register_state_dict_pre_hook(hook)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_save_to_state_dict'))
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'state_dict'))
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_register_load_state_dict_pre_hook'))
    def _register_load_state_dict_pre_hook(self, hook, with_module=False):
        return super()._register_load_state_dict_pre_hook(hook, with_module)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_load_state_dict_pre_hook'))
    def register_load_state_dict_pre_hook(self, hook):
        return super().register_load_state_dict_pre_hook(hook)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'register_load_state_dict_post_hook'))
    def register_load_state_dict_post_hook(self, hook):
        return super().register_load_state_dict_post_hook(hook)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_load_from_state_dict'))
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'load_state_dict'))
    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict, assign)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_named_members'))
    def _named_members(self, get_members_fn, prefix='', recurse=True, remove_duplicate=True):
        return super()._named_members(get_members_fn, prefix, recurse, remove_duplicate)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'parameters'))
    def parameters(self, recurse=True):
        return super().parameters(recurse)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'named_parameters'))
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return super().named_parameters(prefix, recurse, remove_duplicate)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'buffers'))
    def buffers(self, recurse=True):
        return super().buffers(recurse)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'named_buffers'))
    def named_buffers(self, prefix='', recurse=True, remove_duplicate=True):
        return super().named_buffers(prefix, recurse, remove_duplicate)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'children'))
    def children(self):
        return super().children()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'named_children'))
    def named_children(self):
        return super().named_children()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'modules'))
    def modules(self, remove_duplicate=True):
        return super().modules(remove_duplicate)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'named_modules'))
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        return super().named_modules(memo, prefix, remove_duplicate)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'train'))
    def train(self, mode=True):
        return super().train(mode)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'eval'))
    def eval(self):
        return super().eval()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'requires_grad_'))
    def requires_grad_(self, requires_grad=True):
        return super().requires_grad_(requires_grad)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'zero_grad'))
    def zero_grad(self, set_to_none=True):
        return super().zero_grad(set_to_none)

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'share_memory'))
    def share_memory(self):
        return super().share_memory()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_get_name'))
    def _get_name(self):
        return super()._get_name()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'extra_repr'))
    def extra_repr(self):
        return super().extra_repr()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '__repr__'))
    def __repr__(self):
        return super().__repr__()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '__dir__'))
    def __dir__(self):
        return super().__dir__()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, '_replicate_for_data_parallel'))
    def _replicate_for_data_parallel(self):
        return super()._replicate_for_data_parallel()

    # torch.nn.modules.module.Module
    @wraps(getattr_static(_PARENT, 'compile'))
    def compile(self, *args, **kwargs):
        return super().compile(*args, **kwargs)
