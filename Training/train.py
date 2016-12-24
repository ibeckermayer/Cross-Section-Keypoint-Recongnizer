import os
import pickle
import sys
from collections import OrderedDict

import cv2
import matplotlib
#!!!matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
from skimage.util import random_noise

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print "Loaded GPU Convolutional Layers!"
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
    print "WARNING: Loaded CPU Convolutional Layers!"

np.random.seed(42)
sys.setrecursionlimit(10000)

# Name of training data text file
DATA = 'fullTrainingData.txt'

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

# function for testing purposes only
def plotSample(img,y): 
    plt.figure()
    if img.shape == (1,1,288,288):
         img = img.reshape(288,288)
    plt.imshow(img, cmap='gray')
    plt.scatter(y[0::2]*144+144, y[1::2]*144+144, c = 'r', marker='o', s=10)
    cv2.destroyAllWindows()
    plt.show()

def float32(k):
    return np.cast['float32'](k)


def fit_specialists(pretrain=False):
    # Fit specialist models running on the GPU
    a = 1
    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        flip_indices = setting['flip_indices']
        name = setting['name']
        X,y = load2d(cols=cols,flip_inds=flip_indices)
        
        # Create the model
        model = build_model()
        model.verbose = 1
        model.output_num_units = y.shape[1]
        model.max_epochs = int(4e6 / y.shape[0])
        if 'kwargs' in setting:
            # an option 'kwargs' in the settings list may be used to
            # set any other parameter of the net:
            vars(model).update(setting['kwargs'])
        
        if setting['pretrain'] == True:
            pretrain = setting['name'] + '_params.pkl'

        if pretrain:
            try:
                model.load_params_from(pretrain)
                print('loaded pretrained parameters')
            except:
                print('pretrain name error')
            

        print("Training model for columns {} for {} epochs".format(
            cols, model.max_epochs))
        model.fit(X, y)
        train_loss = np.array([i["train_loss"] for i in model.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in model.train_history_])
        plt.figure(a)
        plt.plot(train_loss, linewidth=3, label="train")
        plt.plot(valid_loss, linewidth=3, label="valid")
        plt.grid()
        plt.legend()
        tit = setting['name'] + " Loss"
        plt.title(tit)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        chart_name = setting['name'] + "Loss.png"
        plt.savefig(chart_name)
        name = setting['name'] + '_params.pkl'
        model.save_params_to(name)
        a = a+1

def load(cols=None):
    # Loads the data, with photos X and target values y
    columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    df = read_csv(DATA,usecols = columns) 
    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    # The Image column has pixel values separated by space: convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep = ' '))

    #print(df.count()) # prints the number of values for each column
    df = df.dropna() # drops all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255. # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    y = df[df.columns[:-1]].values
    y = (y - 144) / 144 # scale target coordinates to [-1, 1]
    y = y.astype(np.float32)

    return X, y



# Loads the data in 2D, flips all the data with the proper indices and shuffles the data
def load2d(cols=None, flip_inds=None, dummy_cols=None): 

    # Load and reshape the data 
    X, y = load(cols=cols)
    X = X.reshape(-1, 1, 288, 288)

    # if there are dummy_columns, load them, flip the photos vertically and flip the y values as well
    if dummy_cols:
        X, y = augWithDummy(X,y,dummy_cols)

    # augment the data with by horizontally flipping all photos and outputs
    X, y = augWithHorizontalFlip(X,y,flip_inds=flip_inds)

    # Add noise to all the data
    Xfinal, yfinal = augWithNoise(X,y)

    # Shuffle training data
    Xfinal, yfinal = shuffle(Xfinal, yfinal, random_state = 42) 

    # Make data type float 32 for use with Theano (via Lasagne framework)
    Xfinal = Xfinal.astype(np.float32)
    yfinal = yfinal.astype(np.float32)

    return Xfinal, yfinal

# augments all the data by adding noise to each photo
def augWithNoise(X,y):
    
    # get total number of samples
    len = X.shape[0]

    # You need to add 4 noisy samples, so initialize these values to insert them at the proper 
    # index in the new dataset (with a and b)
    a = 0 # beginning
    b = 4 # end
    c = 1 # to track progress
    # Pre-allocate memory
    Xfinal = np.zeros((len*5,1,288,288))
    yfinal = np.zeros((len*5,y.shape[1]))
    while X.shape[0]>0:
        X_noise = addNoise(X[0][0])
        y1 = y[0]
        y2 = y[0]
        y3 = y[0]
        y4 = y[0]
        y_noise = np.vstack((y1,y2,y3,y4))
        Xfinal[a:b] = X_noise # add augmented data
        yfinal[a:b] = y_noise # add y values
        Xfinal[b] = X[0] # add original X
        yfinal[b] = y[0] # add original y
        X = np.delete(X, (0), axis=0) # delete the sample you just copied
        y = np.delete(y, (0), axis=0) # delete the sample you just copied
        print("augmenting sample {} of {}").format(c, len)
        a = a+5 # move index forward by five 
        b = b+5 # move index forward by five
        c = c+1 # add one to progress
    
    return Xfinal, yfinal

# augments X and y by flipping them horizontally
# for some models, this means the indices of the output need to be flipped, 
# which the function accounts for if necessary
def augWithHorizontalFlip(X,y,flip_inds=False):
    X_flipped = X[:, :, :, ::-1] 
    y_flipped_t = np.copy(y) # initialize temporary copy of y
    y_flipped_t[:, ::2] = y[:, ::2] * -1 # flip all x_coordinate values of this copy

    # flipped values need to switched to correspond to left/right 
    if flip_inds:
        flip_indices = flip_inds
        y_flipped = np.copy(y_flipped_t) # y_flipped is a copy of flipped x values y
        for a,b in flip_indices:
            y_flipped[:,a] = y_flipped_t[:,b]
            y_flipped[:,b] = y_flipped_t[:,a]
    else:
        y_flipped = np.copy(y_flipped_t)
    X = np.vstack((X,X_flipped))
    y = np.vstack((y,y_flipped))

    return X, y


# augments X and y with "dummy columns" that work as similar features to the features in model we are trying to train.
# more quantitatively, it loads the data for the dummy columns and flips the photos/outputs vertically
def augWithDummy(X,y,dummy_columns):

    X_dummy, y_dummy = load(cols=dummy_columns) # load X and y again
    X_dummy = X_dummy.reshape(-1,1,288,288)
    X_dummy = X_dummy[:, :, ::-1, :] # flip X vertically
    y_dummy[:,1::2] = y_dummy[:,1::2] * -1 # flip all y values

    # augment the data
    X = np.vstack((X,X_dummy))
    y = np.vstack((y,y_dummy))

    return X, y







# Adds 4 different types of noise to a photo.
# It it adds each type of noise 4 times on each photo to make the effect more dramatic.
def addNoise(X):
    # adds different types of noise to the photo 
    X1 = random_noise(X,mode='gaussian')
    X2 = random_noise(X,mode='poisson')
    X3 = random_noise(X,mode='s&p')
    X4 = random_noise(X,mode='speckle')
    for i in xrange(3):
        X1 = random_noise(X1,mode='gaussian')
        X2 = random_noise(X2,mode='poisson')
        X3 = random_noise(X3,mode='s&p')
        X4 = random_noise(X4,mode='speckle')
    X1 = X1.reshape(1,1,288,288)
    X2 = X2.reshape(1,1,288,288)
    X3 = X3.reshape(1,1,288,288)
    X4 = X4.reshape(1,1,288,288)

    noisedUp = np.vstack((X1,X2,X3,X4))
    return noisedUp

# Define the network architecture and return the network. 
# I made this a function so I can reuse the same architecture multiple times during training
def build_model():

    net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 288, 288),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    hidden4_num_units=1000,
    dropout4_p=0.5,
    hidden5_num_units=1000,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=BatchIterator(batch_size=128),
    on_epoch_finished=[
    AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    AdjustVariable('update_momentum', start=0.9, stop=0.999),
    EarlyStopping(patience=12),
    ],
    max_epochs=3000,
    verbose=1,
    )

    return net

# Each setting in SPECIALIST_SETTINGS corresponds to model that will be trained on the DATA
# The outputs of the model being trained are represented by columns.
# dummy_columns are used to augment the last two models, which are lacking in data. They are features
# that are similar to the desired features (i.e. the top of the weld looks like an upside down bottom). 
# They are flipped vertically to augment the DATA for these models
# flip_indices are the indices of the columns that must be swapped when the data is augmented by 
# flipping each photo horizontally. 
# Pretrain determines whether the model should load certain pre-trained parameters before to training.
SPECIALIST_SETTINGS = [

    dict(
        columns = ('left_top_x', 'left_top_y', 'middle_top_x', 'middle_top_y', 'right_top_x', 'right_top_y'),
        dummy_columns = (None),
        flip_indices = ((0,4),(1,5)),
        name = 'top',
        pretrain = False
    ),

    dict(
        columns = ('middle_bottom_x', 'middle_bottom_y'),
        dummy_columns = ('middle_top_x', 'middle_top_y'),
        flip_indices = (None),
        name = 'bottom_middle',
        pretrain = False
    ),

    dict(
        columns = ('left_bottom_x', 'left_bottom_y', 'right_bottom_x', 'right_bottom_y'),
        dummy_columns = ('left_top_x', 'left_top_y', 'right_top_x', 'right_top_y'),
        flip_indices = ((0,2),(1,3)),
        name = 'bottom_outside',
        pretrain = False
    )
]


#!!!fit_specialists()
# Get all changes onto GitHub





