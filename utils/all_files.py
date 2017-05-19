import os


def all_files(fpath, exts):
    for root, dirs, files in os.walk(fpath):
        for f in files:
            base, ext = os.path.splitext(f)
            joined = os.path.join(root, f)
            if ext.lower() in exts:
                yield joined
