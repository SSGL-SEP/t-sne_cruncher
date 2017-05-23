import os
import errno


def mkdir_p(path: str):
    """
    Attempts ot create a given folder
    
    :param path: Folder path 
    :type path: str
    """
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
