import numpy as np


def slice_vec_seq(vec_seq, timesteps):
    total_seq_num = (20 - timesteps)
    sliced = np.zeros([total_seq_num, timesteps + 1, vec_seq.shape[1])
    for i in range(total_seq_num):
        sliced[i] = vec_seq[i:i+timesteps+1]
    return sliced

