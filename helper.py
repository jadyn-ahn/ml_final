import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from factory import ModelFactory
            

class Helper:
    
    ENC_DEC = "Model/{}_enc_dec.h5"
    ENC = "Model/{}_enc.h5"
    DEC = "Model/{}_dec.h5"
    LSTM = "Model/{}_lstm.h5"
    
    @staticmethod
    def train_enc_dec(model, trainset, valset, batch_size, epochs, lr, decay, loss="binary_crossentropy"):
        opt = tensorflow.keras.optimizers.Adam(lr=lr, decay=decay)
        model.compile(optimizer=opt, loss=loss)
        checkpoint = ModelCheckpoint(filepath="Model/tmp_enc_dec.h5", verbose=1, save_best_only=True)
        model.fit(trainset.batch(batch_size),
                  validation_data=valset.batch(batch_size),
                  epochs=epochs,
                  steps_per_epoch=200000//batch_size,
                  callbacks=[checkpoint],
                  verbose=1)
        return model
    
    @classmethod
    def save_enc_dec(cls, model, name):
        model.save_weights(cls.ENC_DEC.format(name))
        model.layers[1].save_weights(cls.ENC.format(name))
        model.layers[2].save_weights(cls.DEC.format(name))
        print("SAVE DONE! ENC_DEC {}".format(name))
    
    @classmethod
    def load_enc_dec(cls, name):
        model = ModelFactory.get_enc_dec()
        model.load_weights(cls.ENC_DEC.format(name_))
        return model
    
    @classmethod
    def load_enc(cls, name):
        model = ModelFactory.get_enc()
        model.load_weights(cls.ENC.format(name))
        return model
    
    @classmethod
    def load_dec(cls, name):
        model = ModelFactory.get_dec()
        model.load_weights(cls.DEC.format(name))
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
    def save_lstm(cls, model, name):
        model.save_weights(cls.LSTM.format(name))
        print("SAVE DONE! LSTM {}".format(name))
    
    @classmethod
    def load_lstm(cls, name):
        model = ModelFactory.get_lstm()
        model.load_weights(cls.LSTM.format(name))
        return model