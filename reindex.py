#!/usr/bin/env python

###
# C. Bryan Daniels
# https://github.com/prairie-guy
# 05/04/2022
###

import re, argparse
from pathlib import Path


__all__ = ['reindex']

def reindex_dir(dest, start_idx=0, ext = None):
    "Reindexes files within directory `dest` starting with  `start_idx` Returns 1 + num_of_files "
    dest = Path(dest)
    rnd  = "zsk#@m"
    fns,idx  = [], None
    for idx, fn in enumerate(filter(Path.is_file, dest.iterdir()), start=start_idx):
        suf = ext if ext else fn.suffix.strip('.')
        stm = re.compile('[0-9]*$').sub("",fn.stem.strip('.')) # remove any indx from end of stem
        fn_new = dest/(f'{stm}{idx}.{suf}')
        fn_tmp = dest/(str(idx) + rnd)                         # Need unique fn to  avoid collisions
        fns.append([fn_tmp,fn_new])
        fn.rename(fn_tmp)
    for fn_tmp, fn_new in fns:
        fn_tmp.rename(fn_new)
    return idx + 1 if idx else 0


def reindex(dest, start_idx=0, ext = None):
    """
Uniquely reindexes all files within first-level directories of `dest`
Example: `dest` contains directories of images: dir1/{1.jpg, 2.jpg, 3.jpg}, dir2/{1.jpg, 2.jpg, 3.jpg}, dir3/{1.jpg, 2.jpg, 3.jpg}
reindex(dest, start_idx=1) -> dir1/{1.jpg, 2.jpg, 3.jpg}, dir2/{4.jpg, 5.jpg, 6.jpg}, dir3/{7.jpg, 8.jpg, 9.jpg}

usage: ./reindex.py image_dir --start_idx 100 --ext 'jpg'
    """
    dest = Path(dest)
    if not(dest.is_dir()): return f'{dest} is not a directory'
    idx = start_idx
    for d in filter(Path.is_dir, dest.iterdir()):
        idx = reindex_dir(d, idx, ext=ext)
    return idx + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Uniquely reindexes all files within first-level directories of `dest`")
    parser.add_argument("dest", help= "Contains one or more directories")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index across all files")
    parser.add_argument("--ext", default=None, help="Optional file extention")
    args = parser.parse_args()
    reindex(args.dest, args.start_idx, args.ext)
