import numpy as np
from PIL import Image
from datapath import DataPath


class Loader:

    @classmethod
    def load_imgs(cls, learn_type):
        paths = DataPath.img_paths(learn_type)
        imgs = np.zeros([len(paths), 64, 64, 3])
        for i, path in enumerate(paths):
            img = np.array(Image.open(path)) / 255.0
            imgs[i] = img
        return imgs

    #TODO: Slice vecs properly
    @classmethod
    def load_vecs(cls, learn_type):
        paths = DataPath.vec_paths(learn_type)
        if learn_type == "test":
            vecs = np.zeros([len(paths), 10, 1024])
        else:
            vecs = np.zeors([len(paths), 20, 1024])
        for i, path in enumerate(paths):
            vec = np.load(path)
            vecs[i] = vec
        return vecs
