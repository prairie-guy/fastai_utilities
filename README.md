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
The reason I wrote `reindex`, and the problem it solved, was related to **data cleaning**, specifically when using the widget `ImageClassifierCleaner`. Images downloaded with the `fastai` function `download_images` are named: `{dir1/0000001.jpg, dir1/0000002.jpg, dir1/0000002.jpg, ...}`. Accordingly, for multiple catagories, these are characterized by the name of the directory, not the pathname of the file. When using `ImageClassifierCleaner` to reclassify an image, i.e., move it from `dog` to `cat`, `ImageClassifierCleaner` attempts to move the image from the `dog` directory to the `cat` directory. This will often result in an error as the pathname, i.e., `00001013.jpg` already exists. 

`reindex` fixes this problem by uniquely reindexing all files within first-level directories of `dest`

```
`dest` contains directories of images: dir1/{1.jpg, 2.jpg, 3.jpg}, dir2/{1.jpg, 2.jpg, 3.jpg}, dir3/{1.jpg, 2.jpg, 3.jpg}
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
`methods_of` is a simple way to inspect the `methods` of a `fastai object` It will list the `methods` of a `fastai object`, like a learner. This is a simple tool to help me better understand how the `fastai` library works.

``` python
> data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
> methods_of(data.trn_dl.dataset)
denorm(arr):
get(tfm, x, y):
get_c():
get_n():
get_sz():
get_x(i):
get_y(i):
resize_imgs(targ, new_path):
transform(im, y=None):
```

#### attributes_of(obj, *exclude)
`attributes_of` is similiar to `methods_of` except for that it lists the attributes for that `fastai object`

``` python
> data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
> attributes_of(data.trn_dl.dataset,'fnames')
c: 2
fnames: ...
is_multi: False
is_reg: False
n: 23000
path: data/dogscats/
sz: 224
y: [0 0 0 ... 1 1 1]
```
                                                                                                                                                                                           
