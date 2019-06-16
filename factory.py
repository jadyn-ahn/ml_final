import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from tensorflow.keras.layers import LSTM


class ModelFactory:
    
    VECTOR_SHAPE = 1024
    TIME_STEPS = 10
    
    @classmethod
    def get_enc_dec(cls):
        enc = cls.get_enc()
        dec = cls.get_dec()
        inputs = Input((64, 64, 3))
        outputs = dec(enc(inputs))
        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    @classmethod
    def get_enc(cls):
        layers = Sequential([
            Conv2D(16, (3, 3), input_shape=(64, 64, 3), activation='relu', padding='same', name="conv_enc_1"),
            MaxPooling2D((2, 2), name="max_enc_2"),
            Conv2D(8, (3, 3), activation='relu', padding='same', name="conv_enc_3"),
            MaxPooling2D((2, 2), name="max_enc_4"),
            Conv2D(8, (3, 3), activation='relu', padding='same', name="conv_enc_5"),
            MaxPooling2D((2, 2), name="enc_max_6"),
            Flatten(name="flatten_enc_7"),
            Dense(cls.VECTOR_SHAPE, activation=None, name="dense_enc_8"),
        ])
        return layers
    
    @classmethod
    def get_dec(cls):
        layers = Sequential([
            Dense(8*8*8, input_shape=(cls.VECTOR_SHAPE,), activation=None, name="dense_dec_1"),
            Reshape((8, 8, 8), name="reshape_dec_2"),
            UpSampling2D((2, 2), name="up_dec_3"),
            Conv2D(8, (3, 3), activation='relu', padding='same', name="conv_dec_4"),
            UpSampling2D((2, 2), name="up_dec_5"),
            Conv2D(16, (3, 3), activation='relu', padding='same', name="conv_dec_6"),
            UpSampling2D((2, 2), name="up_dec_7"),
            Conv2D(3, (3, 3), activation='sigmoid', padding='same', name="conv_dec_8"),
        ])
        return layers
    
    @classmethod
    def get_lstm(cls):
        model = Sequential([LSTM(cls.VECTOR_SHAPE, input_shape=(10, 1024)),])
        return model