import os


def all_files(folder_path: str, exts: list):
    """
    Gathers all files conforming to provided extensions.
     
    :param folder_path: Path to folder
    :type folder_path: str
    :param exts: list of file extensions to accept
    :type exts: List[str]
    """
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            base, ext = os.path.splitext(f)
            joined = os.path.join(root, f)
            if ext.lower() in exts:
                yield joined
