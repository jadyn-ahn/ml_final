from helper import Helper
from factory import ModelFactory
from loader import Loader
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=bool)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--decay", default=0.0, type=float)
    args = parser.parse_args()

    if args.model == "enc_dec":
        model = Helper.load_enc_dec(args.load) if args.load else ModelFactory.get_enc_dec()
        td = Loader.load_imgs("train")
        vd = Loader.load_imgs("val")
        Helper.train_enc_dec(model, (td, td), (vd, vd),
                             args.batch_size, args.epochs, args.lr, args.decay)
        if args.save:
            Helper.save_enc_dec(model)

    elif args.model == "lstm":
        #load or get lstm
        model = Helper.load_lstm(args.load) if args.load else ModelFactory.get_lstm()
        #load data
        td = Loader.load_vecs("train")
        vd = Loader.load_vecs("val")
        #train
        Helper.train_lstm(model, (td, td), (vd, vd), args.batch_size, args.epochs, args.lr, args.decay)
        if args.save:
            Helper.save_lstm(model)
        
