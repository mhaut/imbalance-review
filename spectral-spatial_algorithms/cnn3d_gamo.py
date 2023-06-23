import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import gc
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.layers import Activation, BatchNormalization, Conv2D, Conv3D, Dense, Flatten, MaxPooling3D, Input, Reshape, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical as keras_to_categorical
import numpy as np
import sys


def set_params(args):
    args.batch_size = 100; args.epochs = 100
    return args


def get_model_compiled(shapeinput, num_class, w_decay=0, lr=1e-3):
    ## input layer
    input_layer = Input(shapeinput)
    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=num_class, activation='softmax')(dense_layer2)
    clf = Model(inputs=input_layer, outputs=output_layer)
    clf.compile(loss=categorical_crossentropy, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return clf







def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True, \
            choices=["IP", "BW", "UP", "SV", "KSC", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, BW, UP, SV, KSC, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--spatialsize', default=11, type=int, help='windows size')
    parser.add_argument('--wdecay', default=0, type=float, help='apply penalties on layer parameters')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=None, type=int, 
                        help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--tr_percent', default=0.05, type=float, help='samples of train set')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')
    parser.add_argument('--verbosetrain', action='store_true', help='Verbose train')
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size', default=100, type=int, help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of full training cycle on the training set')
    parser.add_argument('--idtest', default=0, type=int, help='Number of test')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    if args.set_parameters: args = set_params(args)

    pixels, labels, num_class = \
                    mydata.loadData(args.dataset, num_components=args.components, preprocessing=args.preprocess)
    pixels, labels = mydata.createImageCubes(pixels, labels, windowSize=args.spatialsize, removeZeroLabels = False)

    rstate = args.random_state+pos if args.random_state != None else None
    if args.dataset in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:
        x_train, x_test, y_train, y_test = \
            mydata.load_split_data_fix(args.dataset, pixels)#, rand_state=args.random_state+pos)
    else:
        pixels = pixels[labels!=0]
        labels = labels[labels!=0] - 1
        x_train, x_test, y_train, y_test = \
            mydata.split_data(pixels, labels, args.tr_percent, rand_state=rstate)

    if args.use_val:
        x_val, x_test, y_val, y_test = \
            mydata.split_data(x_test, y_test, args.val_percent, rand_state=rstate)
        x_val   = x_val[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]
    x_train = x_train[..., np.newaxis]

    inputshape = x_train.shape[1:]
    clf = get_model_compiled(inputshape, num_class, w_decay=args.wdecay, lr=args.lr)
    valdata = (x_val, keras_to_categorical(y_val, num_class)) if args.use_val else (x_test, keras_to_categorical(y_test, num_class))
    clf.fit(x_train, keras_to_categorical(y_train, num_class),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    verbose=args.verbosetrain,
                    validation_data=valdata,
                    callbacks = [ModelCheckpoint("/tmp/best_model.h5", monitor='val_accuracy', verbose=0, save_best_only=True)])
    del clf; K.clear_session(); gc.collect()
    clf = load_model("/tmp/best_model.h5")
    results = mymetrics.reports(np.argmax(clf.predict(x_test), axis=1), y_test)[2]
    print("CNN3D", args.dataset, args.tr_percent, args.spatialsize, args.idtest, results)

if __name__ == '__main__':
    main()



























