from tensorflow.data import Dataset
import numpy as np


class Converter:
    
    @staticmethod
    def convert_imgs(imgs1, imgs2):
        stacked = np.stack((imgs1, imgs2))
        imgs = stacked.reshape([-1, 64, 64, 3])
        ds = Dataset.from_tensor_slices(imgs)
        return ds.map(lambda img: (img, img))
    
    @classmethod
    def convert_vecs(cls, vec_seqs, timesteps):
        ds = Dataset.from_tensor_slices(vec_seqs)
        ds = ds.flat_map(lambda vec_seq: cls.slice_vecs(vec_seq, timesteps))
        return ds.map(lambda vec_seq: (vec_seq[0:timesteps], vec_seq[timesteps]))
    
    @staticmethod
    def slice_vecs(vec_seq, timesteps):
        total_seq_num = (vec_seq.shape[0] - timesteps)
        sliced = np.zeros([total_seq_num, timesteps, vec_seq.shape[1]])
        for i in range(total_seq_num):
            sliced[i] = vec_seq[i:i+timesteps+1]
        return sliced