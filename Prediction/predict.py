import os
import pickle
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

import cv2
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator, NeuralNet

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print "Loaded GPU Convolutional Layers!"
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
    print "WARNING: Loaded CPU Convolutional Layers!"

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

def float32(k):
    return np.cast['float32'](k)

def convertToSquare(image):
    height, width = image.shape
    black_square = np.zeros((width/2*2,width/2*2), np.uint8)

    halfway_down_square = width/2
    halfway_down_image = height/2

    if height%2 == 0 and width%2 == 0:
        roi = image[:,:]
    elif height%2 != 0 and width%2 == 0:
        roi = image[0:height-1,:]
    elif height%2 == 0 and width%2 != 0:
        roi = image[:,0:width-1]
    else:
        roi = image[0:height-1,0:width-1]
    


    black_square[halfway_down_square - halfway_down_image:halfway_down_square + halfway_down_image,:] = roi

    return black_square

# Resizes image to 288 x 288    
def resize288(image):
    image = cv2.resize(image, (288,288))
    return image

# load a dictionary containing the saved models
def load_model():
    specialists = OrderedDict()

    model1 = build_model() 
    model2 = build_model() 
    model3 = build_model() 
    model1.output_num_units = 6
    model2.output_num_units = 2
    model3.output_num_units = 4
    model1.load_params_from(SPECIALIST_SETTINGS[0]['name'] + '_params.pkl')
    model2.load_params_from(SPECIALIST_SETTINGS[1]['name'] + '_params.pkl')
    model3.load_params_from(SPECIALIST_SETTINGS[2]['name'] + '_params.pkl')

    specialists['1'] = model1
    specialists['2'] = model2
    specialists['3'] = model3

    return specialists

# Auto label the image for width, height, and depth
# Input the list of models (from load_model()), image name (string) , units (string), and units per pixel (float) and get a labeled image as the output
# lab_bottom determines whether it will label the bottom of the weld (some welds don't go all the way through)
def label(models,image_name,units="pixels",units_per_pixel=1,lab_bottom=True,show=True):
    
    # find the predicted points according to the model
    x, y = predict(models,image_name,plot=False)
    
    # find the equation of the line connecting top_left and top_right
    m_width, b_width = line_between_2_pts((x[0],y[0]),(x[2],y[2]))

    # find the equation of the line perpendicular to the line connecting top_left and top_right
    # and passing through top middle
    m_height, b_height = perp_line(m_width,(x[1],y[1]))
    
    # find the intersection of the width line and the height line
    inter = intersection(m_width,b_width,m_height,b_height)

    # find the width (pixel distance between top left and top right * units_per_pixel)
    width = dist((x[0],y[0]),(x[2],y[2])) * units_per_pixel

    # find the height (pixel distance between top middle and inter * units_per_pixel)
    height = dist(inter,(x[1],y[1])) * units_per_pixel
    
    # find the "drawn width line" that will be drawn on the photo. It should be parallel to the line
    # connecting top left and top right, and n pixels "up" in the perpendicular direction from the top middle
    n = 5
    m_width_draw, b_width_draw = find_drawn_width_line((x[1],y[1]),m_height,b_height,m_width,n) #!!!

    #!!! plot this

    if lab_bottom:
        # find equation of the line connecting bottom left and bottom right
        m_bot, b_bot = line_between_2_pts((x[4],y[4]),(x[5],y[5]))

        # find the equation of the line perpendicular to the line connecting bottom left and bottom right
        # and passing through bottom middle
        m_depth, b_depth = perp_line(m_bot,(x[3],y[3]))

        # find the intersection of the bottom line and the depth line
        inter = intersection(m_bot,b_bot,m_depth,b_depth)

        # find the depth (pixel distance between bottom middle and inter * units_per_pixel)
        depth = dist(inter,(x[3],y[3])) * units_per_pixel

        #!!! plot THIS TOO!!!!!!

    if show:
        plt.show()

# find the "drawn width line" that will be drawn on the photo. It should be parallel to the line
# connecting top left and top right (m_width), and n pixels "up" in the perpendicular direction (m_height) from the top middle
def find_drawn_width_line(top_middle,m_height,b_height,m_width,n): #!!!!!!!!!!!!!!!!!! Formula's are wrong
    if m_height == np.infty: # special case
        m = 0 # slope is 0
        b = top_middle[1] + n # b becomes y value + n and line is given by y = b
    else:
        # Do lots of algebra
        x = top_middle[0]
        y = top_middle[1]
        a = 1. + m_height**2.
        b = 2.*m_height*b_height - 2.*x - 2.*y*m_height
        c = n**2.-x**2.-y**2.+2.*y*b_height-b_height**2.

        # use pythagorean theorem to find the 2 potential x values for the point n units away from top_middle and on the height line
        x_plus = (-b+np.sqrt(b**2.-4.*a*c))/(2.*a)
        x_minus = (-b-np.sqrt(b**2.-4.*a*c))/(2.*a)
        y_plus = m_height*x_plus + b_height
        y_minus = m_height*x_minus + b_height
        
        print x_plus,y_plus
        print x_minus,y_minus

        # take the x value that gives the higher y value (since we want the line to be "above")
        if y_plus>y_minus: #!!!
            xx = x_plus
            yy = y_plus
        else:
            xx = x_minus
            yy = y_minus

        # now we know we want our drawn line to have the slope of m_width, and we can use the xx,yy we
        # just found to find our intercept bb
        bb = yy - m_width*xx
        m = m_width
        b = bb

    return m,b


# finds the intersection of two lines given their slope and intercepts
# returns as a tuple (x,y)
def intersection(m1,b1,m2,b2):
    if m2 == np.infty:
        x = b2 # this is the correct answer as dealt with by the special case in perp_line()
    else:
        x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return (x,y)

# finds line perpendicular to slope passing through pt
# returns m and b where m and b correspond to y = mx + b
# pt is a tuple
def perp_line(slope,pt):
    if slope == 0: # special case
        m = np.infty # make m infinity to "alert" other functions
        b = pt[0] # b becomes the x value and line is given by x = b
    else:
        m = -slope
        b = pt[1]-pt[0]*m
    return m, b

# takes 2 points and finds the distance between them
# points are tuples in the form (x,y)
def dist(pt1,pt2):
    return np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

# takes 2 pts and returns m and b where m and b correspond to y = mx + b
# points are tuples of the form (x,y)
def line_between_2_pts(pt1,pt2):
    if (pt2[0]-pt1[0]) == 0: # special case
        m = np.infty # make m infinity
        b = pt1[0] # b becomes the x value
    else:
        m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
        b = pt1[1]-pt1[0]*m
    return m, b


# input image name and models, get out the model's prediction **on the original sized image**
def predict(models,image_name,plot=True):

    # Process the image for model
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if plot==True:
        img_for_plot = cv2.imread(image_name)
    original_width = img.shape[1]
    original_height = img.shape[0]
    img = convertToSquare(img)
    img = resize288(img)
    
    # Get the outline of the cross section
    outline = find_outline(img,original_width,original_height)
    
    # Make prediction with the models
    x_vals_predicted, y_vals_predicted = make_prediction(img, models)

    # Move the predictions to the closest point on the outline
    edgeX, edgeY = find_closest_edge(x_vals_predicted,y_vals_predicted,outline)

    # Rescale the predictions to the original size image
    scaledX, scaledY = scale_to_orig(edgeX,edgeY,original_width,original_height)

    if plot == True:
        plotSample(img_for_plot,scaledX,scaledY)

    return scaledX,scaledY

# takes the arrays of predicted x and y values and rescales (from 288x288) them so they will be plotted on the proper place on the
# original image
def scale_to_orig(x,y,original_width,original_height):
    # scaling x is a simple proportion
    scaledX = (original_width/288.) * x
    # y is more complicated since the photos are "pasted" onto middle of black square the size of the width
    # first scale down from (288 x 288) to (288 x scaled_orig_height), where scaled_orig_height (called h)
    # is proportionate to the original height:width ratio and width is 288
    h = (float(original_height)/float(original_width)) * 288.
    # next, find the gap between the top of the "pasted" (288 x 288) picture and the actual photo
    c = (288.-h)/2.
    # now translate the y values to compensate for that gap
    ys = y-c
    # now scale same as for x but with the scaled_orig_height (h)
    scaledY = ys*(original_height/h)
    return scaledX, scaledY

# takes the arrays of predicted x and y values, and for each pair finds the point on the edge of the cross section
# it is closest to
def find_closest_edge(xvals,yvals,outline): #!!!

    # these must be (y,x) because the possible edge pts later will be returned y,x
    top_left = np.transpose(np.array([yvals[0],xvals[0]]))
    top_middle = np.transpose(np.array([yvals[1],xvals[1]]))
    top_right = np.transpose(np.array([yvals[2],xvals[2]]))
    bottom_middle = np.transpose(np.array([yvals[3],xvals[3]]))
    bottom_left = np.transpose(np.array([yvals[4],xvals[4]]))
    bottom_right = np.transpose(np.array([yvals[5],xvals[5]]))

    pred_pts = np.array([top_left,top_middle,top_right,bottom_middle,bottom_left,bottom_right]).reshape((6,2))# (6x2)
    
    possible_edge_pts = np.transpose(np.nonzero(outline>0)) # indices of the possible edge points (nx2)

    # initialize array for closest pts
    closest_pts = np.empty((6,2))

    # for each predicted point
    # get the closest pt and put in the array
    for i in xrange(pred_pts.shape[0]):
        pt = np.array([pred_pts[i,:]])
        closest_pts[i,:] = possible_edge_pts[cdist(pt,possible_edge_pts,'euclidean').argmin()] # after the equals sign is correct

    #remember x and y are flipped
    edgeX = closest_pts[:,1]
    edgeY = closest_pts[:,0]


    return edgeX, edgeY

# takes in 288x288 image and makes a prediction using the models
# returns array xvals & yvals, with columns corresponding to which point
# 0 - top left
# 1 - top middle
# 2 - top right
# 3 - bottom middle
# 4 - bottom left
# 5 - bottom right
def make_prediction(img,models):
    img_for_model = img.reshape(1,1,288,288) 

    y_pred = np.empty((1,0))
    for model in models.values():
        y_pred1 = model.predict(img_for_model/255.)
        y_pred = np.hstack([y_pred, y_pred1])

    y_pred = np.transpose(y_pred)

    xvals = y_pred[0::2]*144+144
    yvals = y_pred[1::2]*144+144

    return xvals, yvals

# Finds the outline of the cross section
def find_outline(img,original_width,original_height):
    # Create binary image using an adaptive threshold
    edged = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,3)
    
    # The outline will now be obvious but there will be a nasty mess in the middle of white and black
    # This will find the contours but not the ones we are looking for, due to the mess
    cnts,_=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # By drawing all the countours in as white, we begin to "fuse" the mess in the middle (which many of the contours follow)
    # This is useful for ensuring the next threshold doesn't destroy relevant pieces of the outline
    cv2.drawContours(img,cnts,-1,(255,255,255),1)

    # Now take apply a regular threshold function over the entire image. This will (with a few errors)
    # make the cross section white and the background black
    ret,th =cv2.threshold(img,50,255,cv2.THRESH_BINARY)

    # Because of how we reshape the photos, there will often now be an extra white line between the original image and the 
    # boundary of where it was "placed" onto the 288/288 black background. Delete this line (and other noise)
    h = 288.0*original_height/original_width
    gap = int(288-h)/2

    th[0:gap+12,:] = 0
    th[287-gap-12:287,:] = 0

    outline = np.zeros((288,288)) # initialize outline

    # find the top edge of the outline
    for j in xrange(287): # for each column:
        # find the top edge of the outline
        for i in xrange(gap+12,287-gap-12): # go down the column:
            if th[i,j]==255: # if the pixel is white:
                outline[i,j] = 255 # make the pixel in the outline array white
                break
        # find the bottom edge of the outline
        for i in reversed(xrange(gap+12,287-gap-12)):
            if th[i,j]==255: # if the pixel is white:
                outline[i,j] = 255 # make the pixel in the outline array white
                break
    return outline
      
def plotSample(img,x,y):
    plt.figure()
    if img.shape == (1,1,288,288):
        img = img.reshape(288,288)
    plt.imshow(img, cmap='gray')
    plt.scatter(x, y, c = 'r', marker='o', s=10)
    cv2.destroyAllWindows()
    plt.show()

SPECIALIST_SETTINGS = [

    dict(
        columns = ('left_top_x', 'left_top_y', 'middle_top_x', 'middle_top_y', 'right_top_x', 'right_top_y'),
        flip_indices = ((0,4),(1,5)),
        name = 'top',
        pretrain = True
    ),

    dict(
        columns = ('middle_bottom_x', 'middle_bottom_y'),
        flip_indices = (None),
        name = 'bottom_middle',
        pretrain = True
    ),

    dict(
        columns = ('left_bottom_x', 'left_bottom_y', 'right_bottom_x', 'right_bottom_y'),
        flip_indices = ((0,2),(1,3)),
        name = 'bottom_outside',
        pretrain = False
    )
]

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

mod = load_model()
