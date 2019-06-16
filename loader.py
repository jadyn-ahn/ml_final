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

    @classmethod
    def load_vecs(cls, learn_type, timesteps):
        paths = DataPath.vec_paths(learn_type)
        total_seq_num = len(paths) * (20 - timesteps)
        lstm_input = np.zeros([total_seq_num, timesteps, 1024])
        lstm_output = np.zeros([total_seq_num, 1024])
        for i, path in enumerate(paths):
            vec_seq = np.load(path)
            for j in range(0, 20 - timesteps):
                order = i * (20 - timesteps) + j
                lstm_input[order] = vec_seq[j:j+timesteps]
                lstm_output[order] = vec_seq[j+timesteps]
        return lstm_input, lstm_output
