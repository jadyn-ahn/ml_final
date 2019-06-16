import numpy as np
import os
from data_utils import *
from datapath import DataPath


def img2vec(model):
    train_in, train_out = get_train_batch(0, 10000)
    train_paths = DataPath.vec_paths("train")
    convert_and_save(model, train_in, train_out, train_paths)
    val_in, val_out = get_val_batch(0, 500)
    val_paths = DataPath.vec_paths("val")
    convert_and_save(model, val_in, val_out, val_paths)
    
        
def convert_and_save(model, in_img_seqs, out_img_seqs, vec_paths):
    for i, path in enumerate(vec_paths):
        in_seq = in_img_seqs[i]
        out_seq = out_img_seqs[i]
        stacked = np.stack((in_seq, out_seq)).reshape((20, 64, 64, 3))
        vectors = model.predict(stacked)
        os.makedirs(path, exist_ok=True)
        np.save(path, vectors)
