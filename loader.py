import numpy as np
from tensorflow.data import Dataset
from PIL import Image
from datapath import DataPath
    
    
class Loader:

    @classmethod
    def load_imgs(cls, learn_type):
        ds = Dataset.from_tensor_slices(DataPath.img_paths(learn_type))
        ds = ds.map(lambda path: cls.load_img(path))
        ds = ds.map(lambda img: (img, img))
        return ds

    @classmethod
    def load_vecs(cls, learn_type):
        ds = Dataset.from_tensor_slices(DataPath.vec_paths(learn_type))
        ds = ds.map(lambda path: cls.load_vec(path))
        return ds
    
    @staticmethod
    def load_img(path):
        img = np.array(Image.open(img_path)) / 255.0  # normalize
        return img

    @staticmethod
    def load_vec_seq(path):
        vec_seq = np.load(path)
        return vec_seq
    
