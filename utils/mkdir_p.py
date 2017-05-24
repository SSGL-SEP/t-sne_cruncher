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
        v = os.path.isdir(path)
        w = ex.errno == errno.EEXIST
        if v and w:
            return "ex"
            pass
        else:
            raise
