"""
Pickle Functions

@auth: Yu-Hsiang Fu
@date: 2019/06/12
@update: 2022/04/03
"""
# --------------------------------------------------------------------------------
# 1.Import modular
# --------------------------------------------------------------------------------
try:
    import _pickle as pickle  # cPickle in Python 3.x
except:
    import pickle  # normal pickle
import zlib


# --------------------------------------------------------------------------------
# 2.Define function
# --------------------------------------------------------------------------------
# copy
def deepcopy(data):
    return pickle.loads(pickle.dumps(data, -1))


# load/save
def load(file_path, is_compressed=False):
    if is_compressed:
        with open(file_path, "rb") as f:
            return pickle.loads(zlib.decompress(f.read()))
    else:
        return pickle.load(open(file_path, "rb"))


def save(file_path, data, is_compressed=False):
    if is_compressed:
        with open(file_path, "wb") as f:
            f.write(zlib.compress(pickle.dumps(data, protocol=4), zlib.Z_BEST_COMPRESSION))
    else:
        pickle.dump(data, open(file_path, "wb"), protocol=4)  # protocol=4: >4GB
