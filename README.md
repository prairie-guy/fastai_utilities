## fastai_utilities

### Several useful scripts for use with `fast.ai` lectures and libraries.

### Installation and Usage
`git clone https://github.com/prairie-guy/ai_utilities.git`

This is not a pip file, instead one just needs to clone the repository and include `fastai_utilities` is in the `python search path`.

Doing this from within python is easy:

```python
import sys
sys.path.append('your-parent-directory-of-fastai_utilities')
from fastai_utilities import *
```    

## Utilities

### reindex(dest, start_idx=0, ext = None)
The reason I wrote `reindex`, and the problem it solved, was related to **data cleaning**, specifically when using the widget `ImageClassifierCleaner`. Images downloaded with the fastai function `download_images` are named: **{dir1/0000001.jpg, dir1/0000002.jpg, dir1/0000002.jpg, ...}**. Accordingly, for multiple catagories, these are characterized by the name of the directory, not the pathname of the file. When using `ImageClassifierCleaner` to reclassify an image, i.e., move it from **dog**` to **cat**, `ImageClassifierCleaner` attempts to move the image from the **dog** directory to the **cat** directory. This will often result in an error as the pathname, i.e., **00001013.jpg** already exists. 

`reindex` fixes this problem by uniquely reindexing all files within first-level directories of **dest**

```
**dest** contains directories of images: dir1/{1.jpg, 2.jpg, 3.jpg}, dir2/{1.jpg, 2.jpg, 3.jpg}, dir3/{1.jpg, 2.jpg, 3.jpg}
reindex(dest, start_idx=1) -> dir1/{1.jpg, 2.jpg, 3.jpg}, dir2/{4.jpg, 5.jpg, 6.jpg}, dir3/{7.jpg, 8.jpg, 9.jpg}
```

##### From within python
```python
import sys
sys.path.append('your-parent-directory-of-fastai_utilities')
from fastai_utilities import *

reindex('dataset_folder', start_idx = 100, ext = 'jpg'
```

##### From shell
``` bash
usage: reindex.py [-h] [--start_idx START_IDX] [--ext EXT] dest

Uniquely reindexes all files within first-level directories of `dest`

positional arguments:
  dest                  Contains one or more directories

optional arguments:
  -h, --help            show this help message and exit
  --start_idx START_IDX
                        Starting index across all files
  --ext EXT             Optional file extention

```

### methods_of(obj,lr=False)
`methods_of` is a simple way to inspect the methods of a fastai object It will list the methods of a fastai object, like a learner. This is a simple tool to help me better understand how the fastai library works.

``` python
learn = vision_learner(dls, resnet34, metrics=[error_rate,accuracy])
methods_of(learn)
```

```
add_cb(cb):
add_cbs(cbs):
add_module(name: str, module: Optional[ForwardRef('Module')]) -> None:
added_cbs(cbs):
all_batches():
append(module: torch.nn.modules.module.Module) -> 'Sequential':
apply(fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T:
arch(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> torchvision.models.resnet.ResNet:
bfloat16() -> ~T:
buffers(recurse: bool = True) -> Iterator[torch.Tensor]:
children() -> Iterator[ForwardRef('Module')]:
cpu() -> ~T:
create_opt():
cuda(device: Union[int, torch.device, NoneType] = None) -> ~T:
double() -> ~T:
eval() -> ~T:
export(fname='export.pkl', pickle_module=<module 'pickle' from '/home/cdaniels/anaconda3/lib/python3.9/pickle.py'>, pickle_protocol=2):
extra_repr() -> str:
fine_tune(epochs, base_lr=0.002, freeze_epochs=1, lr_mult=100, pct_start=0.3, div=5.0, lr_max=None, div_final=100000.0, wd=None, moms=None, cbs=None, reset_opt=False):
fit(n_epoch, lr=None, wd=None, cbs=None, reset_opt=False):
fit_flat_cos(n_epoch, lr=None, div_final=100000.0, pct_start=0.75, wd=None, cbs=None, reset_opt=False):
fit_one_cycle(n_epoch, lr_max=None, div=25.0, div_final=100000.0, pct_start=0.25, wd=None, moms=None, cbs=None, reset_opt=False):
fit_sgdr(n_cycles, cycle_len, lr_max=None, cycle_mult=2, cbs=None, reset_opt=False, wd=None):
float() -> ~T:
forward(input):
freeze():
freeze_to(n):
get_buffer(target: str) -> 'Tensor':
get_extra_state() -> Any:
get_parameter(target: str) -> 'Parameter':
get_preds(ds_idx=1, dl=None, with_input=False, with_decoded=False, with_loss=False, act=None, inner=False, reorder=True, cbs=None, save_preds=None, save_targs=None, with_preds=True, with_targs=True, concat_dim=0, pickle_protocol=2):
get_submodule(target: str) -> 'Module':
half() -> ~T:
load(file, device=None, with_opt=True, strict=True):
load_state_dict(state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
loss_func(*input, **kwargs):
loss_not_reduced():
lr_find(start_lr=1e-07, end_lr=10, num_it=100, stop_div=True, show_plot=True, suggest_funcs=<function valley at 0x7f1236792d30>):
model(*input, **kwargs):
modules() -> Iterator[ForwardRef('Module')]:
named_buffers(prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
named_children() -> Iterator[Tuple[str, ForwardRef('Module')]]:
named_modules(memo: Optional[Set[ForwardRef('Module')]] = None, prefix: str = '', remove_duplicate: bool = True):
named_parameters(prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]:
no_bar():
no_logging():
no_mbar():
one_batch(i, b):
opt_func(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-05, wd=0.01, decouple_wd=True):
ordered_cbs(event):
parameters(recurse: bool = True) -> Iterator[torch.nn.parameter.Parameter]:
predict(item, rm_type_tfms=None, with_input=False):
progress(event_name):
recorder(event_name):
register_backward_hook(hook: Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Optional[torch.Tensor]]) -> torch.utils.hooks.RemovableHandle:
register_buffer(name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
register_forward_hook(hook: Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle:
register_forward_pre_hook(hook: Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle:
register_full_backward_hook(hook: Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Optional[torch.Tensor]]) -> torch.utils.hooks.RemovableHandle:
register_module(name: str, module: Optional[ForwardRef('Module')]) -> None:
register_parameter(name: str, param: Optional[torch.nn.parameter.Parameter]) -> None:
remove_cb(cb):
remove_cbs(cbs):
removed_cbs(cbs):
requires_grad_(requires_grad: bool = True) -> ~T:
save(file, with_opt=True, pickle_protocol=2):
set_extra_state(state: Any):
share_memory() -> ~T:
show_results(ds_idx=1, dl=None, max_n=9, shuffle=True, **kwargs):
show_training_loop():
splitter(m):
state_dict(destination=None, prefix='', keep_vars=False):
summary():
to(*args, **kwargs):
to_detach(b, cpu=True, gather=True):
to_empty(*, device: Union[str, torch.device]) -> ~T:
to_fp16(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
to_fp32():
to_non_native_fp16(loss_scale=512, flat_master=False, dynamic=True, max_loss_scale=16777216.0, div_factor=2.0, scale_wait=500, clip=None):
to_non_native_fp32():
train(mode: bool = True) -> ~T:
train_eval(event_name):
tta(ds_idx=1, dl=None, n=4, item_tfms=None, batch_tfms=None, beta=0.25, use_max=False):
type(dst_type: Union[torch.dtype, str]) -> ~T:
unfreeze():
validate(ds_idx=1, dl=None, cbs=None):
validation_context(cbs=None, inner=False):
xpu(device: Union[int, torch.device, NoneType] = None) -> ~T:
zero_grad(set_to_none: bool = False) -> None:
```

#### attributes_of(obj, *exclude)
`attributes_of` is similiar to `methods_of` except for that it lists the attributes for that fastai object.

``` python
learn = vision_learner(dls, resnet34, metrics=[error_rate,accuracy])
attributes_of(learn)
```

```
T_destination: ~T_destination
cbs: [TrainEvalCallback, Recorder, ProgressCallback]
create_mbar: True
dls: <fastai.data.core.DataLoaders object at 0x7f1234bd0f10>
dump_patches: False
lock: <unlocked _thread.lock object at 0x7f1234bf3780>
lr: 0.001
metrics: [<fastai.learner.AvgMetric object at 0x7f1234bef880>, <fastai.learner.AvgMetric object at 0x7f1234bef9a0>]
model_dir: models
moms: (0.95, 0.85, 0.95)
n_epoch: 1
n_out: 4
normalize: True
opt: <fastai.optimizer.Optimizer object at 0x7f1234bf3310>
path: .
pretrained: True
train_bn: True
training: False
training: False
wd: None
wd_bn_bias: False
```
                                                                                                                                                                                           
