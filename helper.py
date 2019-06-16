import tensorflow
from datetime.datetime import now
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from factory import ModelFactory
            

class Helper:
    
    ENC_DEC = "Model/{}_enc_dec.h5"
    ENC = "Model/{}_enc.h5"
    DEC = "Model/{}_dec.h5"
    LSTM = "Model/{}_lstm.h5"
    TIME_FORM = "{:%m%d%H%M}"
    
    @staticmethod
    def train_enc_dec(model, train_data, val_data, batch_size, epochs, lr, decay, loss="binary_crossentropy"):
        opt = tensorflow.keras.optimizers.Adam(lr=lr, decay=decay)
        model.compile(optimizer=opt, loss=loss)
        checkpoint = ModelCheckpoint(filepath="Model/tmp_enc_dec.h5", verbose=1, save_best_only=True)
        model.fit(train_data[0],
                  train_data[1],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=val_data,
                  callbacks=[checkpoint],
                  verbose=1)
        return model
    
    @classmethod
    def save_enc_dec(cls, model):
        t = cls.TIME_FORM.format(now())
        model.save_weights(cls.ENC_DEC.format(t))
        model.layers[1].save_weights(cls.ENC.format(t))
        model.layers[2].save_weights(cls.DEC.format(t))
        print("SAVE DONE! ENC_DEC {}".format(t))
    
    @classmethod
    def load_enc_dec(cls, time):
        model = ModelFactory.get_enc_dec()
        model.load_weights(cls.ENC_DEC.format(time))
        return model
    
    @classmethod
    def load_enc(cls, time):
        model = ModelFactory.get_enc()
        model.load_weights(cls.ENC.format(time))
        return model
    
    @classmethod
    def load_dec(cls, time):
        model = ModelFactory.get_dec()
        model.load_weights(cls.DEC.format(time))
        return model
    
    @staticmethod
    def train_lstm(model, trainset, valset, batch_size, epochs, lr, decay, loss="mean_absolute_error"):
        opt = tensorflow.keras.optimizers.Adam(lr=lr, decay=decay)
        model.compile(optimizer=opt, loss=loss)
        checkpoint = ModelCheckpoint(filepath="Model/tmp_lstm.h5", verbose=1, save_best_only=True)
        model.fit(trainset.batch(batch_size),
                  validation_data=valset,
                  epochs=epochs,
                  steps_per_epoch=100000//batch_size,
                  callbacks=[checkpoint],
                  verbose=1)
        return model
    
    @classmethod
    def save_lstm(cls, model):
        t = cls.TIME_FORM.format(now())
        model.save_weights(cls.LSTM.format(t))
        print("SAVE DONE! LSTM {}".format(t))
    
    @classmethod
    def load_lstm(cls, time):
        model = ModelFactory.get_lstm()
        model.load_weights(cls.LSTM.format(time))
        return model