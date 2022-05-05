## fastai_utilities

### Several useful scripts for use with `fast.ai` lectures and libraries.

### Installation and Usage
`git clone https://github.com/prairie-guy/ai_utilities.git`

```
import sys
sys.path.append('your-parent-directory-of-fastai_utilities')
from fastai_utilities import *
```    

### Utilities

####  reindex

Uniquely reindexes all files within first-level directories of `dest`

```python
From within python

import sys
sys.path.append('your-parent-directory-of-fastai_utilities')
from fastai_utilities import *

`dest` contains directories of images: dir1/{1.jpg, 2.jpg, 3.jpg}, dir2/{1.jpg, 2.jpg, 3.jpg}, dir3/{1.jpg, 2.jpg, 3.jpg}

reindex(dest, start_idx=1) -> dir1/{1.jpg, 2.jpg, 3.jpg}, dir2/{4.jpg, 5.jpg, 6.jpg}, dir3/{7.jpg, 8.jpg, 9.jpg}

```


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




#### methods_of(obj,lr=False)
List the attribues of a fastai object, like a learner

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
List the methods of a fastai object

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









- 







### Example Usage
```
import sys
sys.path.append('your-parent-directory-of-fastai_utilities')
from fastai_utilities import *


```    


