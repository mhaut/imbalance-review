import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
import numpy as np

import gc
#import tensorflow as tf
#from tf.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical as keras_to_categorical

import sys
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, ADASYN
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def set_params(args):
    args.batch_size = 100; args.epochs = 300
    return args

def get_model_compiled(n_bands, num_class, hsize=None):
    if hsize == None: hsize = n_bands
    clf = Sequential()
    clf.add(Dense(int(hsize*2/3.)+10, activation='relu', input_shape=(n_bands,)))
    clf.add(Dense(num_class, activation='softmax'))
    clf.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    return clf

def set_params_smoteSVM(args):
    if args.dataset   in ["IP", "DIP", "DIPr"]:  args.smC = 10; args.smg = 0.01
    elif args.dataset in ["UP", "DUP", "DUPr"]:  args.smC = 1e3; args.smg = 0.01
    elif args.dataset == "KSC":  args.smC = 100000; args.smg = 0.001
    elif args.dataset == "BW":  args.smC = 100; args.smg = 0.01
    elif args.dataset == "SV":  args.smC = 100; args.smg = 0.01
    return args

def main():
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True, \
            choices=["IP", "UP", "KSC", "BW", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, KSC, BW, SV, UH, DIP, DUP, DIPr, DUPr)')

    parser.add_argument('--oversampling', type=str, required=True, \
            choices=["None", "RANDOM", "PCA", "SMOTE", "SMOTEBD1", "SMOTEBD2", "ADASYN", "KMEANSSMOTE", "SVMSMOTE"], \
            help='dataset (options: None, PCA, RANDOM, SMOTE, SMOTEBD1, SMOTEBD2,  ADASYN, KMEANSSMOTE, SVMSMOTE)')

    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size', default=100, type=int, help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs', default=300, type=int, help='Number of full training cycle on the training set')
    #########################################

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    if args.set_parameters: args = set_params(args)

    pixels, _, _ = mydata.loadData(args.dataset, preprocessing=args.preprocess)
    n_bans_orig = pixels.shape[-1]
    del pixels

    pixels, labels, num_class = \
                    mydata.loadData(args.dataset, num_components=args.components, preprocessing=args.preprocess)
    pixels = pixels.reshape(-1, pixels.shape[-1])

    if args.dataset in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:
        x_train, x_test, y_train, y_test = \
            mydata.load_split_data_fix(args.dataset, pixels)#, rand_state=args.random_state)
    else:
        labels = labels.reshape(-1)
        pixels = pixels[labels!=0]
        labels = labels[labels!=0] - 1
        rstate = args.random_state if args.random_state != None else None
        x_train, x_test, y_train, y_test = \
            mydata.split_data(pixels, labels, args.tr_percent, rand_state=rstate)
        num_class, n_pix_class = np.unique(y_train, return_counts=True)
        num_class = len(num_class)

        for a,b in zip(range(num_class+1), n_pix_class):
            if b < 5:
                pixels = pixels[labels!=a]
                labels = labels[labels!=a]
        labelsuniq = np.unique(labels, return_counts=1)[0]
        
        for idi, i in enumerate(labelsuniq):
            labels[labels==i] = idi

        x_train, x_test, y_train, y_test = \
            mydata.split_data(pixels, labels, args.tr_percent, rand_state=rstate)
        num_class, n_pix_class = np.unique(y_train, return_counts=True)
        num_class = len(num_class)


        if args.oversampling == "None":
            pass
        elif args.oversampling == "PCA":
            shapeortr = list(x_train.shape)
            shapeorte = list(x_test.shape)
            x_train = x_train.reshape(-1, x_train.shape[-1])
            x_test  = x_test.reshape(-1, x_test.shape[-1])
            num_components = int((num_class * 2) * 1.2)
            mypca = PCA(n_components=num_components)
            mypca.fit(x_train)
            mypca.fit(x_test)
            x_train = mypca.transform(x_train)
            x_test  = mypca.transform(x_test)
            shapeortr[-1] = num_components
            shapeorte[-1] = num_components
            x_train = x_train.reshape(shapeortr)
            x_test  = x_test.reshape(shapeorte)
        else:
            if args.oversampling == "RANDOM":
                sm = RandomOverSampler(random_state=rstate)
            elif args.oversampling == "SMOTE":
                sm = SMOTE(random_state=rstate, k_neighbors=4)
            elif args.oversampling == "SMOTEBD1":
                sm = BorderlineSMOTE(random_state=rstate, kind='borderline-1', k_neighbors=4)
            elif args.oversampling == "SMOTEBD2":
                sm = BorderlineSMOTE(random_state=rstate, kind='borderline-2', k_neighbors=4)
            elif args.oversampling == "ADASYN":
                sm = ADASYN(random_state=rstate, n_neighbors=4)
            elif args.oversampling == "KMEANSSMOTE":
                sm = KMeansSMOTE(random_state=rstate, k_neighbors=4)
            elif args.oversampling == "SVMSMOTE":
                args = set_params_smoteSVM(args)
                sm = SVMSMOTE(random_state=rstate, svm_estimator=SVC(gamma=args.smg, C=args.smC, tol=1e-7), k_neighbors=4)
            x_train, y_train = sm.fit_resample(x_train, y_train)
    
    
    n_bands = x_train.shape[-1]
    clf = get_model_compiled(n_bands, num_class, hsize=n_bans_orig)
    
    _, x_val, _, y_val = train_test_split(x_test, y_test, test_size=0.2, stratify=y_test)
    valdata = (x_val, keras_to_categorical(y_val, num_class))

    clf.fit(x_train, keras_to_categorical(y_train, num_class),
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=False,
                validation_data=valdata,
                callbacks = [ModelCheckpoint("/tmp/best_model.h5", monitor='val_accuracy', verbose=0, save_best_only=True)])
    clf = load_model("/tmp/best_model.h5")
    miresult = mymetrics.reports(np.argmax(clf.predict(x_test), axis=1), y_test)

    print("MLP", args.dataset, args.tr_percent, args.oversampling, args.random_state, list(miresult[2]))
    print(miresult[3])


if __name__ == '__main__':
	main()



























