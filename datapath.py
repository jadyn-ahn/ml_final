import os

class DataPath:

    @classmethod
    def img_paths(cls, learn_type):
        top = "Data"
        mid = cls.mid_format(learn_type)
        bot = "frames{:02}.png"

        seq_num = cls.seq_num(learn_type)
        frame_num = cls.frame_num(learn_type)
        paths = []
        for i in range(seq_num):
            for j in range(frame_num):
                paths.append(os.path.join(top, mid.format(i), bot.format(j)))
        return paths

    @classmethod
    def vec_paths(cls, learn_type):
        top = "Vector"
        mid = cls.mid_format(learn_type)
        bot = "vecs.npy"

        seq_num = cls.seq_num(learn_type)
        paths = []
        for i in range(seq_num):
            paths.append(os.path.join(top, mid.format(i), bot))
        return paths

    @staticmethod
    def seq_num(learn_type):
        if learn_type == "train":
            return 10000
        else: #val or test
            return 500

    @staticmethod
    def frame_num(learn_type):
        if learn_type == "test":
            return 10
        else: #train or val
            return 20

    @staticmethod
    def mid_format(learn_type):
        if learn_type == "train":
            return os.path.join("train_sequence", "sequence{:04}")
        else:
            return os.path.join(learn_type + "_sequence", "sequence{:03}")
