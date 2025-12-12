def str2list(v):
    if isinstance(v, list):
        return v
    return v.split(',')