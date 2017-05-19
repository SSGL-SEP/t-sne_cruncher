def normalize(x):
    x -= x.min(axis=0)
    x /= x.max(axis=0)
    return x
