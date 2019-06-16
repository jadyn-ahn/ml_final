import os
import numpy as np
from PIL import Image


class DataPath:
    
    @classmethod
    def img_path(cls, learn_type, seq_number, frame_num):
        return os.path.join("Data", cls.dir_format(learn_type, seq_number), "frames{:02}.png".format(frame_num))
    
    @classmethod
    def vec_path(cls, learn_type, seq_number):
        return os.path.join("Vector", cls.dir_format(learn_type, seq_number), "vectors.npy")
    
    @staticmethod
    def dir_format(learn_type, seq_number):
        if learn_type == "train":
            dir_path =os.path.join("train_sequence", "sequence{:04}/".format(seq_number))
        elif learn_type == "val":
            dir_path = os.path.join("val_sequence", "sequence{:03}/".format(seq_number))
        else: #data_type == "test"
            dir_path = os.path.join("test_sequence", "sequence{:03}/".format(seq_number))
        return dir_path
    
    
class Loader:
    
    TRAIN_NUM = 10000
    VAL_NUM = 500
    TEST_NUM = 500
    
    @classmethod
    def load_train_imgs(cls):
        return cls.load_imgs(0, cls.TRAIN_NUM, 10, 10, "train")
    
    @classmethod
    def load_val_imgs(cls):
        return cls.load_imgs(0, cls.VAL_NUM, 10, 10, "val")
    
    @classmethod
    def load_test_imgs(cls, start, end):
        return cls.load_imgs(0, cls.TEST_NUM, 10, 0, "test")
    
    @classmethod
    def load_imgs(cls, start, end, T, K, learn_type):
        input = np.zeros([end-start, T, 64, 64, 3])
        output = np.zeros([end-start, K, 64, 64, 3])
        for i, idx in enumerate(range(start, end)):
            for t in range(T+K):
                img_path = DataPath.img_path(learn_type, idx, t)
                img = np.array(Image.open(img_path)) / 255.0  # normalize
                if t < 10:
                    input[i, t] = img
                else:
                    output[i, t-10] = img
        return input, output
    
    @classmethod
    def load_train_vecs(cls):
        return cls.load_vecs(0, cls.TRAIN_NUM, "train")
    
    @classmethod
    def load_val_vecs(cls):
        return cls.load_vecs(0, cls.VAL_NUM, "val")
    
    @classmethod
    def load_test_vecs(cls):
        return cls.load_vecs(0, cls.TEST_NUM, "test")
    
    @staticmethod
    def load_vecs(start, end, learn_type):
        if learn_type == "test":
            vec_seqs = np.zeros([end-start, 10, 1024])
        else:
            vec_seqs = np.zeros([end-start, 20, 1024])
        for i in range(start, end):
            vec_seqs[i] = np.load(DataPath.vec_path(learn_type, seq_num))
        return vec_seqs