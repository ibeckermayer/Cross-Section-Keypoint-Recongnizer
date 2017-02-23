import os
import pickle
import sys
import csv
from collections import OrderedDict

import matplotlib as mlp
#!!!mlp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import xml.etree.ElementTree as et

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

# GLOBAL VARIABLES
LABEL_SIZE = 26
LABEL_SIZE_ADJUSTMENT = 10 # (If label's are overlapping too much with the lines, make this bigger)
LABEL_COLOR = 'red'

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

def convert_to_square(image):
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
def resize_288(image):
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

# Label all the photos in the ToBeLabeled directory. Each photo should begin with the photo number (i.e. 90_5x.jpg)
# and have a corresponding XML file (i.e. 90_5x.jpg_meta.xml)
# The photos will be given a folder name (based on their number) that contains the original photo, xml file, and labeled photo
def auto_label(mod):
    for file in os.listdir('ToBeLabeled/'):
        name, ext = os.path.splitext(file)
        if not(ext=='.jpg' or ext=='.png' or ext=='.bmp'):
            continue
        number = get_number_from_name(name)
        target_dir =os.path.join('Labeled/',number)
        if not(os.path.isdir(target_dir)):
            os.mkdir(target_dir)
        XML_file_name = 'ToBeLabeled/' + file + '_meta.xml'
        scaling_factor = find_scaling_factor_from_XML(XML_file_name)
        image_name = 'ToBeLabeled/' + name + ext
        fig, ax = label(mod,image_name,units_per_pixel=scaling_factor,show=False)
        fileName = os.path.join(target_dir,name+'_labeled.png')
        fig.savefig(fileName,transparent=True,bboxinches='tight',pad_inches=0)
        os.rename('ToBeLabeled/'+file,os.path.join(target_dir,file))
        os.rename(XML_file_name,os.path.join(target_dir,XML_file_name[12:len(XML_file_name)]))
        plt.close()



# String -> String
# Get the number (as a string) from the name of a photo file
def get_number_from_name(name):
    name = list(name)
    for i in xrange(len(name)):
        if not(is_number(name[i])):
            return ''.join(name[0:i])

# String -> Boolean
# Check is a string is a number or not
def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False

# Label the image for width, height, and depth
# Input the list of models (from load_model()), image name (string) , units (string), and units per pixel (float) and get a labeled image as the output
# lab_bottom determines whether it will label the bottom of the weld (some welds don't go all the way through)
def label(models,image_name,units=r'$\mu m$',units_per_pixel=1,lab_bottom=True,show=True):
    # find the predicted points according to the model
    x, y, original_width, original_height, top_left_edgepoint, top_right_edgepoint = predict(models,image_name,plot=False)
    # separate points into tuples
    top_left   = (x[0],y[0])
    top_middle = (x[1],y[1])
    top_right  = (x[2],y[2])
    bottom_middle = (x[3],y[3])
    bottom_left = (x[4],y[4])
    bottom_right = (x[5],y[5])
    # find all the boundaries and lengths for labeling
    left_width_boundary,right_width_boundary,top_height_boundary,bottom_height_boundary,top_depth_boundary,bottom_depth_boundary,width,height,depth,inter,inter2 = find_all_boundaries_and_lengths(top_left,top_middle,top_right,bottom_middle,bottom_left,bottom_right,units_per_pixel,top_left_edgepoint,top_right_edgepoint)
    # plot
    fig,ax = plot_labels(image_name,top_left,top_middle,top_right,bottom_middle,bottom_left,bottom_right,left_width_boundary,right_width_boundary,top_height_boundary,bottom_height_boundary,top_depth_boundary,bottom_depth_boundary,lab_bottom,width,height,depth,units,units_per_pixel,inter,inter2,original_width,original_height,top_left_edgepoint,top_right_edgepoint)
    if show:
        plt.axis('off')
        plt.show()
    return fig, ax

# Parse the XML file that comes with each cross section in order to find the proper scaling factor for labeling
def find_scaling_factor_from_XML(file_name):
    e = et.parse(file_name).getroot() # parse the file
    val = e.find('Scaling/Factor_0') # find the scaling factor
    fact_text = val.text # get the scaling factor (as a string)
    fact_list = list(fact_text) # turn the string into a list (to allow editing)
    for i in xrange(len(fact_list)): # change the comma to a decimal point (damn Commies!)
        if fact_list[i]==',':
            fact_list[i] = '.'
    return float(''.join(fact_list))



# plot all the labels
def plot_labels(image_name,top_left,top_middle,top_right,bottom_middle,bottom_left,bottom_right,left_width_boundary,right_width_boundary,top_height_boundary,bottom_height_boundary,top_depth_boundary,bottom_depth_boundary,lab_bottom,width,height,depth,units,units_per_pixel,inter,inter2,original_width,original_height,top_left_edgepoint,top_right_edgepoint):
    line_color=LABEL_COLOR
    fig = plt.figure()
    ax = plt.Axes(fig,[0., 0., 1., 1.])
    fig.set_size_inches(original_width/fig.dpi,original_height/fig.dpi)
    fig.add_axes(ax)
    img = plt.imread(image_name)
    if len(img.shape) == 2:
        ax.imshow(img,aspect='auto',cmap='gray')
    else:
        ax.imshow(img,aspect='auto')
    ax.plot(np.array([top_left[0],left_width_boundary[0]]),np.array([top_left[1],left_width_boundary[1]]),color=line_color)
    ax.plot(np.array([top_right[0],right_width_boundary[0]]),np.array([top_right[1],right_width_boundary[1]]),color=line_color)
    # save width line as a variable for labeling
    width_line, = ax.plot(np.array([left_width_boundary[0],right_width_boundary[0]]),np.array([left_width_boundary[1],right_width_boundary[1]]),color=line_color)
    ax.plot(np.array([top_left_edgepoint[0],top_right_edgepoint[0]]),np.array([top_left_edgepoint[1],top_right_edgepoint[1]]),color=line_color,linestyle='--')
    # find midpoint for text
    width_text_point = midpoint(left_width_boundary,right_width_boundary)
    ax.plot(np.array([top_middle[0],top_height_boundary[0]]),np.array([top_middle[1],top_height_boundary[1]]),color=line_color)
    ax.plot(np.array([inter[0],bottom_height_boundary[0]]),np.array([inter[1],bottom_height_boundary[1]]),color=line_color)
    # save height line as a variable for labeling
    height_line, = ax.plot(np.array([top_height_boundary[0],bottom_height_boundary[0]]),np.array([top_height_boundary[1],bottom_height_boundary[1]]),color=line_color)
    # find midpoint, angle for text
    height_text_point = midpoint(bottom_height_boundary,top_height_boundary)
    if lab_bottom:
        # plot
        ax.plot(np.array([bottom_left[0],bottom_right[0]]),np.array([bottom_left[1],bottom_right[1]]),color=line_color,linestyle='--')
        ax.plot(np.array([inter2[0],top_depth_boundary[0]]),np.array([inter2[1],top_depth_boundary[1]]),color=line_color)
        ax.plot(np.array([bottom_middle[0],bottom_depth_boundary[0]]),np.array([bottom_middle[1],bottom_depth_boundary[1]]),color=line_color)
        depth_line = ax.plot(np.array([top_depth_boundary[0],bottom_depth_boundary[0]]),np.array([top_depth_boundary[1],bottom_depth_boundary[1]]),color=line_color)
        # label
        depth_text_point = midpoint(bottom_depth_boundary,top_depth_boundary)
        label_line(height_line,str('%.2f'%depth)+' '+units,depth_text_point[0],depth_text_point[1],'depth')
    # Label all text at the end to avoid messing up aspect ratios:
    label_line(width_line,str('%.2f'%width)+' '+units,width_text_point[0],width_text_point[1],'width')
    label_line(height_line,str('%.2f'%height)+' '+units,height_text_point[0],height_text_point[1],'height')
    # Setup figure properly for saving as image
    a = fig.gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,original_width)
    plt.ylim(original_height,0)
    return fig, ax


# find all the boundaries for labeling
def find_all_boundaries_and_lengths(top_left,top_middle,top_right,bottom_middle,bottom_left,bottom_right,units_per_pixel,top_left_edgepoint,top_right_edgepoint):
    # find the equation of the line connecting top_left and top_right
    m_width, b_width = line_between_2_pts(top_left,top_right)
    # find the equation of the line connecting the edge points
    m_edge, b_edge = line_between_2_pts(top_left_edgepoint,top_right_edgepoint)
    # find the equation of the line perpendicular to the line connecting top_left_edgepoint and top_right_edgepoint
    # and passing through top middle
    m_height, b_height = perp_line(m_edge,top_middle)
    # find the intersection of the width line and the height line
    inter = intersection(m_edge,b_edge,m_height,b_height)
    # find the width (pixel distance between top left and top right * units_per_pixel)
    width = dist_between_two_pts(top_left,top_right) * units_per_pixel
    # find the height (pixel distance between top middle and inter * units_per_pixel)
    height = dist_between_two_pts(inter,top_middle) * units_per_pixel
    # find the "drawn width line" that will be drawn on the photo. It should be parallel to the line
    # connecting top left and top right, and n pixels "up" in the perpendicular direction from the top middle
    n = 40
    m_width_draw, b_width_draw = find_drawn_width_line(top_middle,m_height,b_height,m_width,n)
    # find the boundary points of the draw line. These will be the intersection of the lines going through the left_top/right_top points with
    # the slope of m_height, and the draw line. They will be tuples (x,y)
    left_width_boundary = find_width_bound(m_width_draw,b_width_draw,m_height,top_left)
    right_width_boundary = find_width_bound(m_width_draw,b_width_draw,m_height,top_right)
    # take midpoint of the midpoint between the intersection point and the top right (projected onto top edge) as the bottom height label boundary
    bottom_height_boundary = midpoint(midpoint(inter,(top_right[0],m_edge*top_right[0]+b_edge)),(top_right[0],m_edge*top_right[0]+b_edge))
    # find the line going through this point with slope m_height
    m_height_draw = m_height
    b_height_draw = bottom_height_boundary[1] - m_height_draw*bottom_height_boundary[0]
    if m_height_draw == np.infty:
        b_height_draw = bottom_height_boundary[0]
    # find the line going through top_middle with slope m_edge
    mm = m_edge
    bb = top_middle[1] - top_middle[0]*mm
    # take the intersection between the last 2 lines as the top height boundary
    top_height_boundary = intersection(mm,bb,m_height_draw,b_height_draw)
    # find equation of the line connecting bottom left and bottom right
    m_bot, b_bot = line_between_2_pts(bottom_left,bottom_right)
    # find the equation of the line perpendicular to the line connecting bottom left and bottom right
    # and passing through bottom middle
    m_depth, b_depth = perp_line(m_bot,bottom_middle)
    # find the intersection of the bottom line and the depth line
    inter2 = intersection(m_bot,b_bot,m_depth,b_depth)
    # find the depth (pixel distance between bottom middle and inter * units_per_pixel)
    depth = dist_between_two_pts(inter2,bottom_middle) * units_per_pixel
    # find the equation of the line with slope m_depth going through bottom_height_boundary
    m_depth_draw = m_depth
    b_depth_draw = bottom_height_boundary[1] - m_depth_draw*bottom_height_boundary[0]
    if m_depth_draw == np.infty:
        b_depth_draw = bottom_height_boundary[0]
    # top depth boundary is intersection of this line and the line connecting bottom left and bottom right (m_bot,b_bot)
    top_depth_boundary = intersection(m_bot,b_bot,m_depth_draw,b_depth_draw)
    # find equation going through bottom_middle with slope m_bot
    mm2 = m_bot
    bb2 = bottom_middle[1] - mm2*bottom_middle[0]
    # bottom depth boundary is intersection of depth draw line and this line
    bottom_depth_boundary = intersection(mm2,bb2,m_depth_draw,b_depth_draw)
    return left_width_boundary,right_width_boundary,top_height_boundary,bottom_height_boundary,top_depth_boundary,bottom_depth_boundary,width,height,depth,inter,inter2



def label_line(line, label, x, y, hor_or_vert, color=LABEL_COLOR, size=LABEL_SIZE):
    """Add a label to a line, at the proper angle.
        
        Arguments
        ---------
        line : matplotlib.lines.Line2D object,
        label : str
        x : float
        x-position to place center of text (in data coordinated
        y : float
        y-position to place center of text (in data coordinates)
        color : str
        size : float
        """
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]
    
    ax = line.get_axes()
    if hor_or_vert == 'width':
        text = ax.annotate(label, xy=(x, y), xytext=(0, LABEL_SIZE_ADJUSTMENT),
                           textcoords='offset points',
                           size=size, color=color,
                           horizontalalignment='center',
                           verticalalignment='center')
    if hor_or_vert == 'height' or hor_or_vert == 'depth':
        text = ax.annotate(label, xy=(x, y), xytext=(LABEL_SIZE_ADJUSTMENT, 0),
                           textcoords='offset points',
                           size=size, color=color,
                           horizontalalignment='center',
                           verticalalignment='center')

    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])
    
    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees)
    return text

def midpoint(p1,p2):
    return ((p1[0]+p2[0])/2.,(p1[1]+p2[1])/2.)

# find the boundary points of the draw line. These will be the intersection of the lines going through the left_top/right_top points with
# the slope of m_height, and the draw line. They will be tuples (x,y)
def find_width_bound(m_width_draw,b_width_draw,m,pt):
    x = pt[0]
    y = pt[1]
    b = y-m*x
    if m == np.infty:
        b = x
    return intersection(m_width_draw,b_width_draw,m,b)

# find the "drawn width line" that will be drawn on the photo. It should be parallel to the line
# connecting top left and top right (m_width), and n pixels "up" in the perpendicular direction (m_height) from the top middle
def find_drawn_width_line(top_middle,m_height,b_height,m_width,n):
    if m_height == np.infty: # special case
        m = 0 # slope is 0
        b = top_middle[1] - n # b becomes y value + n and line is given by y = b
    else:
        # Do lots of algebra: find the pts n units from
        x = top_middle[0]
        y = top_middle[1]
        a = m_height**2. + 1
        b = 2.*m_height*b_height - 2.*x - 2.*y*m_height
        c = b_height**2. - 2*y*b_height + y**2 + x**2 - n**2
        
        # use the quadratic formula to find the 2 potential x values for the point n units away from top_middle and on the height line
        x_plus = (-b+np.sqrt(b**2.-4.*a*c))/(2.*a)
        x_minus = (-b-np.sqrt(b**2.-4.*a*c))/(2.*a)
        y_plus = m_height*x_plus + b_height
        y_minus = m_height*x_minus + b_height
        
        # take the x value that gives the lower y value (since we want the line to be "above", but the pixel order changes the intuition)
        if y_plus<y_minus:
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
        m = -1./slope
        b = pt[1]-pt[0]*m
    return m, b

# takes 2 points and finds the distance between them
# points are tuples in the form (x,y)
def dist_between_two_pts(pt1,pt2):
    return np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

# takes 2 pts and returns m and b where m and b correspond to y = mx + b
# points are tuples of the form (x,y)
def line_between_2_pts(pt1,pt2):
    if (float(pt2[0])-float(pt1[0])) == 0: # special case
        m = np.infty # make m infinity
        b = pt1[0] # b becomes the x value
    else:
        m = (float(pt2[1])-float(pt1[1]))/(float(pt2[0])-float(pt1[0]))
        b = float(pt1[1])-float(pt1[0])*m
    return m, b


# input image name and models, get out the model's prediction **on the original sized image**
def predict(models,image_name,plot=True):
    # Process the image for model
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    imgp = convert_to_square(img)
    imgp = resize_288(imgp)
    if plot==True:
        img_for_plot = cv2.imread(image_name)
    original_width = img.shape[1]
    original_height = img.shape[0]
    # Get the outline of the cross section
    outline, top_outline = find_outline(img,original_width,original_height)

    # Make prediction with the models
    x_vals_predicted, y_vals_predicted = make_prediction(imgp, models)

    # Rescale the predictions to the original size image
    scaledX, scaledY = scale_to_orig(x_vals_predicted,y_vals_predicted,original_width,original_height)
    
    # Move the predictions to the closest point on the outline
    edgeX, edgeY = find_closest_edge(scaledX,scaledY,outline)
    
    # find the points defining the top edge
    top_left_edgepoint, top_right_edgepoint, mleft, bleft, mright, bright = find_edge_points(outline,top_outline,edgeX,edgeY)

    # revise top_left and top_right predictions to be more accurate
    #!!!Xfinal,Yfinal = revise_top_left_and_top_right(top_outline,top_left_edgepoint,top_right_edgepoint,edgeX,edgeY)

    #!!! testing FarthestDistanceFromTopLine


    if plot == True:
        plot_sample(img_for_plot,scaledX,scaledY)
        plot_sample(imgp,x_vals_predicted,y_vals_predicted)
        plot_sample(img_for_plot,edgeX,edgeY)
        plot_sample(outline,edgeX,edgeY)
        plt.show()

    return edgeX,edgeY,original_width,original_height,top_left_edgepoint,top_right_edgepoint, mleft, bleft, mright, bright # edgeY and edgeX should become Xfinal and Yfinal !!!

# Takes outline, top_outline arrays and full X and Y prediction arrays (all 6 points) and returns top_left_edgepoint and top_right_edgepoint as (x,y) pairs
def find_edge_points(outline,top_outline,X,Y):
    top_left   = (X[0],Y[0])
    top_right  = (X[2],Y[2])
    top_outline_left = np.hstack((top_outline[:,0:top_left[0]],np.zeros((outline.shape[0],top_outline.shape[1]-top_left[0]))))
    top_outline_right= np.hstack((np.zeros((outline.shape[0],top_right[0])),top_outline[:,top_right[0]:top_outline.shape[1]]))
    mleft,bleft = find_top_line(outline,top_outline_left)
    mright,bright = find_top_line(outline,top_outline_right)
    top_left_edgepoint = (0,bleft)
    top_right_edgepoint= (outline.shape[1]-1,mright*(outline.shape[1]-1)+bright)
    return top_left_edgepoint, top_right_edgepoint, mleft, bleft, mright, bright

# Takes top_outline, top_left_edgepoint, top_right_edgepoint and full X and Y prediction arrays (all 6 points) and revises top_left and top_right
# (X/Y[0] and X/Y[2]) via empirically derived algorithm that searches for said points in relation to the top edge line
def revise_top_left_and_top_right(top_outline,top_left_edgepoint,top_right_edgepoint,X,Y):
    return None #!!!

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
def find_closest_edge(xvals,yvals,outline):
    
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
        closest_pts[i,:] = possible_edge_pts[cdist(pt,possible_edge_pts,'euclidean').argmin()]

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
    
    rows = original_height
    columns = original_width
    
    outline = np.zeros((rows,columns)) # initialize outline
    top_outline = np.zeros((rows,columns)) # initialize top outline
    
    # find the top edge of the outline
    for j in xrange(columns): # for each column:
        # find the top edge of the outline
        for i in xrange(rows): # go down the column:
            if th[i,j]==255: # if the pixel is white:
                outline[i,j] = 255 # make the pixel in the outline array white
                top_outline[i,j]=255
                break
        # find the bottom edge of the outline
        for i in reversed(xrange(rows)):
            if th[i,j]==255: # if the pixel is white:
                outline[i,j] = 255 # make the pixel in the outline array white
                break

    # delete standalone pixels (these can throw off predictions)
    for j in xrange(1,columns-1):
        for i in xrange(1,rows-1):
            if outline[i,j]==255 and outline[i-1,j-1]==0 and outline[i-1,j]==0 and outline[i-1,j+1]==0 and outline[i,j-1]==0 and outline[i,j+1]==0 and outline[i+1,j-1]==0 and outline[i+1,j]==0 and outline[i+1,j+1]==0:
                outline[i,j] = 0
                top_outline[i,j] = 0


    return outline, top_outline


def plot_sample(img,x,y):
    plt.figure()
    if img.shape == (1,1,288,288):
        img = img.reshape(288,288)
    plt.imshow(img, cmap='gray')
    plt.scatter(x, y, c = 'r', marker='o', s=10)
    cv2.destroyAllWindows()
    plt.show()

def find_top_line(outline,top_outline):
    max_m, min_m = find_max_min_slopes(outline)
    
    maxnumpts = 0 # holds the value for the maximum number of points
    m_final = 0 # initialize final slope value
    b_final = 0 # initialize final intercept value
    edge_pts = np.transpose(np.nonzero(top_outline>0)) # indices of the possible edge points (nx2) (y,x)
    edge_pts[:,[0,1]] = edge_pts[:,[1,0]] # make (x,y)
    for i in xrange(edge_pts.shape[0]): # for each point of the top outline
        d = {} # initialize dictionary: key = slope (m), entry = list of intercepts for that slope (should be identical, but useful to keep as a list to have both the value and the number of total pts on said line)
        for j in xrange(edge_pts.shape[0]): # for eacch point of the top outline
            if i!=j: # if it's not the current point (i)
                m,b = line_between_2_pts(edge_pts[i,:],edge_pts[j,:]) # find the line connecting i and j
                if d.has_key(m): # if another pt has this slope, add the intercept to its list
                    d[m].append(b)
                else:
                    d[m] = [b] # else, create key of slope m and add a list with intercept b
        # Now we have dictionary d with all slope values to other pts and a list of intercepts the length of
        # however many pts reside on the given line.
        
        # Determine the line with the most # of points through it within the confines of our slope limits
        max = 0
        key = 0
        for keyy in d:
            maxx = len(d[keyy])
            if maxx >= max and keyy > min_m and keyy < max_m:
                max = maxx
                key = keyy
        if max > maxnumpts: # if the max pts through a line from this round is greater than the max we already have, update final values
            maxnumpts = max
            m_final = key
            b_final = d[key][0]

    return m_final, b_final

# takes the outline and finds the maximum and minimum slope for the top line,
# the heuristic being that the slope can't be greater or less than
# the slope from one corner of the outline to the other divided by 2
def find_max_min_slopes(outline):
    top_l    = np.infty
    top_r    = np.infty
    bottom_l = np.infty
    bottom_r = np.infty
    for i in xrange(outline.shape[0]):
        if outline[i,0]==255:
            top_l = (0,i)
            break
    for i in xrange(outline.shape[0]):
        if outline[i,outline.shape[1]-1]==255:
            top_r = (outline.shape[1]-1,i)
            break
    for i in reversed(xrange(outline.shape[0])):
        if outline[i,0]==255:
            bottom_l = (0,i)
            break
    for i in reversed(xrange(outline.shape[0])):
        if outline[i,outline.shape[1]-1]==255:
            bottom_r = (outline.shape[1]-1,i)
            break

    min_m, _ = line_between_2_pts(bottom_l,top_r)
    max_m, _ = line_between_2_pts(bottom_r,top_l)
    min_m = min_m/2.
    max_m = max_m/2.
    
    return max_m,min_m


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

#!!!mod = load_model()
#auto_label(mod)

# takes the top line and finds the point on the top outline that's farthest below (or above, according to the photo coordinate system) that line.
def farthest_distance_below_top_line(mleft,bleft,mright,bright,top_outline):
    top_left  = (0,0)
    left_max_dist = -1
    top_right = (0,0)
    right_max_dist = -1
    top_outline_left  = np.hstack((top_outline[:,0:top_outline.shape[1]/2],np.zeros((top_outline.shape[0],top_outline.shape[1]-top_outline.shape[1]/2))))
    top_outline_right = np.hstack((np.zeros((top_outline.shape[0],top_outline.shape[1]/2)),top_outline[:,top_outline.shape[1]/2:top_outline.shape[1]]))
    left_edge_pts = np.transpose(np.nonzero(top_outline_left>0))
    left_edge_pts[:,[0,1]] = left_edge_pts[:,[1,0]] # make (x,y)
    right_edge_pts = np.transpose(np.nonzero(top_outline_right>0))
    right_edge_pts[:,[0,1]] = right_edge_pts[:,[1,0]] # make (x,y)

    for i in xrange(left_edge_pts.shape[0]):
        pt = (left_edge_pts[i,0],left_edge_pts[i,1])
        dist = distance_from_point_to_line(pt,mleft,bleft)
        if dist>left_max_dist and mleft*pt[0]+bleft < pt[1]:
            left_max_dist=dist
            top_left=pt

    for i in xrange(right_edge_pts.shape[0]):
        pt = (right_edge_pts[i,0],right_edge_pts[i,1])
        dist = distance_from_point_to_line(pt,mright,bright)
        if dist>right_max_dist and mright*pt[0]+bright < pt[1]:
            right_max_dist=dist
            top_right=pt

    return top_left, top_right


def distance_from_point_to_line(pt,m,intercept):
    x = pt[0]
    y = pt[1]
    a = float(-m)
    b = 1.
    c = float(-intercept)
    dist = (np.abs(a*x+b*y+c))/(np.sqrt(a**2.+b**2.))
    return dist

# Takes the top_outline and the top line equations for the left side and the right side, and finds the first point in the top outline
# that intersects each line, to produce an estimate for top_left and top_right
def find_top_line_intersection_points(top_outline,mleft,bleft,mright,bright):
    # initialize top_left and top_right
    top_left = None
    top_right = None
    
    # find (x,y) coordinates of the top_outline and sort them in ascending order from x=0 -> x=num_columns
    edge_pts = np.transpose(np.nonzero(top_outline>0))
    edge_pts[:,[0,1]] = edge_pts[:,[1,0]] # make (x,y)
    edge_pts = edge_pts[edge_pts[:,0].argsort()]
    
    # separate into left and right edge points
    num_pts = edge_pts.shape[0]
    halfway = num_pts/2
    left_edge_pts = edge_pts[0:halfway,:]
    right_edge_pts = edge_pts[halfway:num_pts,:]
    
    # starting from x=halfway-1 and moving towards x=0, find the equation of each point in the top_outline and its neigbor to
    # the left. If the x value of the intersection of this line and the top line left (mleft, bleft) is between the x values
    # of the point and it's neighbor, then this intersection point is top_left
    for i in reversed(xrange(halfway)):
        p1 = left_edge_pts[i,:]
        if mleft*p1[0]+bleft == p1[1]:
            top_left = p1
            break
        p2 = left_edge_pts[i-1,:]
        m_neighbor,b_neighbor = line_between_2_pts(p1,p2)
        if m_neighbor == mleft:
            continue
        inter = intersection(m_neighbor,b_neighbor,mleft,bleft)
        if inter[0] >= p2[0] and inter[0] <= p1[0]:
            top_left = inter
            break

    # starting from x=halfway and moving towards x=num_pts, find the equation of each point in the top_outline and its neigbor to
    # the right. If the x value of the intersection of this line and the top line left (mleft, bleft) is between the x values
    # of the point and it's neighbor, then this intersection point is top_right
    for i in xrange(right_edge_pts.shape[0]):
        p1 = right_edge_pts[i,:]
        if mright*p1[0]+bright == p1[1]:
            top_right = p1
            break
        p2 = right_edge_pts[i+1,:]
        m_neighbor,b_neighbor = line_between_2_pts(p1,p2)
        if m_neighbor == mright:
            continue
        inter = intersection(m_neighbor,b_neighbor,mright,bright)
        if inter[0] >= p1[0] and inter[0] <= p2[0]:
            top_right = inter
            break

    return top_left, top_right

# Takes the predictions from farthest_distance_below_top_line() and find_top_line_intersection_points, finds the midpoint
# between them and then finds the closest point on the top_outline to that point. This new point is sometimes the best prediction.
def project_from_midpoint_of_2_predictions(pt1,pt2,top_outline):
    midpt = midpoint(pt1,pt2)
    # need a length 6 vector for x and y for use with find_closest_edge
    xx = [midpt[0],0.,0.,0.,0.,0.,0.]
    yy = [midpt[1],0.,0.,0.,0.,0.,0.]
    X,Y = find_closest_edge(xx,yy,top_outline)
    return (X[0],Y[0])




###!!! everything above this line should be copied to full program
import pandas as pd
"""
# This script is used to generate a data file that contains the distance (and relative based on photo size) between the find_top_line_intersection_points() 
# and farthest_distance_below_top_line(), the distance (and relative based on photo size) between farthest_distance_below_top_line() and the top line 
# (on each respective side), the abs() of the slope between find_top_line_intersection_points() and farthest_distance_below_top_line(), and the distance 
# between find_top_line_intersection_points() and top_middle() for a test set of photos, and attempts to use those as a heuristic for deciding which of the 
# predictions to choose (intersection, farthest, midpoint, any, none). This script was used before the predict() function was modified.

data = ['NAME', 'distance_left', 'distance_right', 'relative_dist_left', 'relative_dist_right', 'depth_left', 'depth_right', 'relative_depth_left', 'relative_depth_right', 'abs_slope_left', 'abs_slope_right', 'dist_to_middle_left', 'dist_to_middle_right']
with open('experiments/experimentData.csv', 'wb') as f:
    a = csv.writer(f,delimiter=",")
    a.writerow(data)
    f.close()
for file in os.listdir("experiments"):
    if file.endswith(".jpg"):
        image_name = "experiments/" + file
        #plt.close("all")
        #ax = plt.subplot(111)

        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        original_width = img.shape[1]
        original_height = img.shape[0]
        # Get the outline of the cross section
        outline, top_outline = find_outline(img,original_width,original_height)

        edgeX,edgeY,original_width,original_height,top_left_edgepoint,top_right_edgepoint, mleft, bleft, mright, bright = predict(mod,image_name,plot=False)
        top_middle = (edgeX[1],edgeY[1])
        top_left,top_right = farthest_distance_below_top_line(mleft,bleft,mright,bright,top_outline)
        top_left2,top_right2 = find_top_line_intersection_points(top_outline,mleft,bleft,mright,bright)
        top_left3 = project_from_midpoint_of_2_predictions(top_left,top_left2,top_outline)
        top_right3 = project_from_midpoint_of_2_predictions(top_right,top_right2,top_outline)
        #img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img, cmap='gray')
        MARKER_SIZE = 12
        #plt.scatter([top_left[0],top_right[0]],[top_left[1],top_right[1]],s=MARKER_SIZE,c='y',label='farthest')
        #plt.scatter([top_left2[0],top_right2[0]],[top_left2[1],top_right2[1]],s=MARKER_SIZE,c='r',label='intersection')
        #plt.scatter([top_left3[0],top_right3[0]],[top_left3[1],top_right3[1]],s=MARKER_SIZE,c='b',label='midpoint')
        #plt.legend()
        #plt.title(image_name[12:len(image_name)])
        distance_left = dist_between_two_pts(top_left,top_left2)
        distance_right = dist_between_two_pts(top_right,top_right2)
        relative_dist_left = distance_left/original_width
        relative_dist_right = distance_right/original_width
        depth_left = distance_from_point_to_line(top_left,mleft,bleft)
        depth_right = distance_from_point_to_line(top_right,mright,bright)
        relative_depth_left = depth_left/original_height
        relative_depth_right = depth_right/original_height
        abs_slope_left, _ = line_between_2_pts(top_left,top_left2)
        abs_slope_left = np.abs(abs_slope_left)
        abs_slope_right, _ = line_between_2_pts(top_right,top_right2)
        abs_slope_right = np.abs(abs_slope_right)
        dist_to_middle_left = dist_between_two_pts(top_left2,top_middle)
        dist_to_middle_right = dist_between_two_pts(top_right2,top_middle)

        NAME = image_name[0:len(image_name)-3]+'pickle'
        #pickle.dump(ax, open(NAME, 'w'))
        data = [NAME, "%.15f" % distance_left, "%.15f" % distance_right, "%.15f" % relative_dist_left, "%.15f" % relative_dist_right, "%.15f" % depth_left, "%.15f" % depth_right, "%.15f" % relative_depth_left, "%.15f" % relative_depth_right, "%.15f" % abs_slope_left, "%.15f" % abs_slope_right, "%.15f" % dist_to_middle_left, "%.15f" % dist_to_middle_right]
        with open('experiments/experimentData.csv', 'ab') as f:
            a = csv.writer(f,delimiter=",")
            a.writerow(data)
            f.close()
"""
"""
# After creating the data, this script is used to go through each of the test photos and look at the diagram, so I can enter which point to choose manually

import pandas as pd
df = pd.read_csv('experiments/experimentData.csv')
for index, row in df.iterrows():
    ax = pickle.load(open(row['NAME']))
    plt.show()

"""

"""
# plot the data
df = pd.read_csv('experiments/experimentData.csv')
df.loc[:,'dist_to_middle_left_over_dist_to_middle_right']=df['dist_to_middle_left']/df['dist_to_middle_right']
df.loc[:,'dist_to_middle_right_over_dist_to_middle_left']=df['dist_to_middle_right']/df['dist_to_middle_left']

f1 = plt.figure()
ax = f1.add_subplot(111)

left_f = df[df.left_farthest_inter_mid_any_or_neither=='f']
right_f = df[df.right_farthest_inter_mid_any_or_neither=='f']
depth_left_f = left_f['depth_left'].values
distance_left_f = left_f['distance_left'].values
depth_right_f = right_f['depth_right'].values
distance_right_f = right_f['distance_right'].values
abs_slope_left_f = left_f['abs_slope_left'].values
abs_slope_right_f = right_f['abs_slope_right'].values
right_over_left_f = right_f['dist_to_middle_right_over_dist_to_middle_left'].values
left_over_right_f = left_f['dist_to_middle_left_over_dist_to_middle_right'].values
ratio_f = np.hstack((left_over_right_f,right_over_left_f))
slope_f = np.hstack((abs_slope_left_f,abs_slope_right_f))
depth_f = np.hstack((depth_left_f,depth_right_f))
distance_f = np.hstack((distance_left_f,distance_right_f))
ax.scatter(distance_f,ratio_f,c='y',label='farthest')

left_m = df[df.left_farthest_inter_mid_any_or_neither=='m']
right_m = df[df.right_farthest_inter_mid_any_or_neither=='m']
depth_left_m = left_m['depth_left'].values
distance_left_m = left_m['distance_left'].values
depth_right_m = right_m['depth_right'].values
distance_right_m = right_m['distance_right'].values
abs_slope_left_m = left_m['abs_slope_left'].values
abs_slope_right_m = right_m['abs_slope_right'].values
right_over_left_m = right_m['dist_to_middle_right_over_dist_to_middle_left'].values
left_over_right_m = left_m['dist_to_middle_left_over_dist_to_middle_right'].values
ratio_m = np.hstack((left_over_right_m,right_over_left_m))
slope_m = np.hstack((abs_slope_left_m,abs_slope_right_m))
depth_m = np.hstack((depth_left_m,depth_right_m))
distance_m = np.hstack((distance_left_m,distance_right_m))
ax.scatter(distance_m,ratio_m,c='b',label='midpoint')

left_i = df[df.left_farthest_inter_mid_any_or_neither=='i']
right_i = df[df.right_farthest_inter_mid_any_or_neither=='i']
depth_left_i = left_i['depth_left'].values
distance_left_i = left_i['distance_left'].values
depth_right_i = right_i['depth_right'].values
distance_right_i = right_i['distance_right'].values
abs_slope_left_i = left_i['abs_slope_left'].values
abs_slope_right_i = right_i['abs_slope_right'].values
right_over_left_i = right_i['dist_to_middle_right_over_dist_to_middle_left'].values
left_over_right_i = left_i['dist_to_middle_left_over_dist_to_middle_right'].values
ratio_i = np.hstack((left_over_right_i,right_over_left_i))
slope_i = np.hstack((abs_slope_left_i,abs_slope_right_i))
depth_i = np.hstack((depth_left_i,depth_right_i))
distance_i = np.hstack((distance_left_i,distance_right_i))
ax.scatter(distance_i,ratio_i,c='r',label='intersection')


left_a = df[df.left_farthest_inter_mid_any_or_neither=='a']
right_a = df[df.right_farthest_inter_mid_any_or_neither=='a']
depth_left_a = left_a['depth_left'].values
distance_left_a = left_a['distance_left'].values
depth_right_a = right_a['depth_right'].values
distance_right_a = right_a['distance_right'].values
abs_slope_left_a = left_a['abs_slope_left'].values
abs_slope_right_a = right_a['abs_slope_right'].values
right_over_left_a = right_a['dist_to_middle_right_over_dist_to_middle_left'].values
left_over_right_a = left_a['dist_to_middle_left_over_dist_to_middle_right'].values
ratio_a = np.hstack((left_over_right_a,right_over_left_a))
slope_a = np.hstack((abs_slope_left_a,abs_slope_right_a))
depth_a = np.hstack((depth_left_a,depth_right_a))
distance_a = np.hstack((distance_left_a,distance_right_a))
ax.scatter(distance_a,ratio_a,c='b',label='any')

left_n = df[df.left_farthest_inter_mid_any_or_neither=='n']
right_n = df[df.right_farthest_inter_mid_any_or_neither=='n']
depth_left_n = left_n['depth_left'].values
distance_left_n = left_n['distance_left'].values
depth_right_n = right_n['depth_right'].values
distance_right_n = right_n['distance_right'].values
abs_slope_left_n = left_n['abs_slope_left'].values
abs_slope_right_n = right_n['abs_slope_right'].values
right_over_left_n = right_n['dist_to_middle_right_over_dist_to_middle_left'].values
left_over_right_n = left_n['dist_to_middle_left_over_dist_to_middle_right'].values
ratio_n = np.hstack((left_over_right_n,right_over_left_n))
slope_n = np.hstack((abs_slope_left_n,abs_slope_right_n))
depth_n = np.hstack((depth_left_n,depth_right_n))
distance_n = np.hstack((distance_left_n,distance_right_n))
ax.scatter(distance_n,ratio_n,c='k',label='none')


plt.xlabel('distance')
plt.ylabel('ratio')

plt.legend()
plt.show()
"""

# Create a classifer using sklearn's One-vs-Rest Logistic Regression
# data is in CSV with the following columns:
# NAME, distance_left, distance_right, relative_dist_left, relative_dist_right, depth_left, depth_right, relative_depth_left, relative_depth_right, abs_slope_left, abs_slope_right, dist_to_middle_left, dist_to_middle_right, left_farthest_inter_mid_any_or_neither, right_farthest_inter_mid_any_or_neither
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('experiments/experimentData.csv')
df.loc[:,'dist_to_middle_left_over_dist_to_middle_right']=df['dist_to_middle_left']/df['dist_to_middle_right']
df.loc[:,'dist_to_middle_right_over_dist_to_middle_left']=df['dist_to_middle_right']/df['dist_to_middle_left']
# remove crazy outliers

# Classifier will be the same for left and right points, so engineer numpy arrays so that left is simply
# on top of right 

depth_left      = df['depth_left'].values
depth_right     = df['depth_right'].values
num_rows = depth_left.shape[0]+depth_right.shape[0]
depth = np.transpose(np.hstack((depth_left,depth_right))).reshape(num_rows,1)

distance_left   = df['distance_left'].values
distance_right  = df['distance_right'].values
distance = np.transpose(np.hstack((distance_left,distance_right))).reshape(num_rows,1)

abs_slope_left  = df['abs_slope_left'].values
abs_slope_right = df['abs_slope_right'].values
abs_slope = np.transpose(np.hstack((abs_slope_left,abs_slope_right))).reshape(num_rows,1)

left_over_right = df['dist_to_middle_left_over_dist_to_middle_right'].values
right_over_left = df['dist_to_middle_right_over_dist_to_middle_left'].values
ratio_to_center = np.transpose(np.hstack((left_over_right,right_over_left))).reshape(num_rows,1)

classification_left = df['left_farthest_inter_mid_any_or_neither']
classification_right = df['right_farthest_inter_mid_any_or_neither']
classification_letters = np.transpose(np.hstack((classification_left,classification_right))).reshape(num_rows,1)

Xy = np.concatenate((depth,distance,abs_slope,ratio_to_center,classification_letters),axis=1)

# turn back into DataFrame to do logical masking and clean data of outliers
df2 = pd.DataFrame(Xy, columns = ['depth','distance','abs_slope','ratio_to_center','classification_letters'])
df2=df2[df2.depth<=np.mean(df2['depth'].values)+np.std(df2['depth'].values)]
df2=df2[df2.distance<=np.mean(df2['distance'].values)+2*np.std(df2['distance'].values)]
df2=df2[df2.abs_slope<=np.mean(df2['abs_slope'].values)+np.std(df2['abs_slope'].values)]
df2=df2[df2.classification_letters != 'n']
df2['depth_squared'] = df2['depth']**2. 
df2['distance_squared'] = df2['distance']**2.
df2['abs_slope_squared'] = df2['abs_slope']**2.
df2['depth_cubed'] = df2['depth']**3.
df2['distance_cubed'] = df2['distance']**3.
df2['abs_slope_cubed'] = df2['abs_slope']**3.
df2['depth_sqrt'] = df2['depth']**0.5
df2['distance_sqrt'] = df2['distance']**0.5
df2['abs_slope_sqrt'] = df2['abs_slope']**0.5
df2['depth_distance'] = df2['depth']*df2['distance']
df2['depth_abs_slope'] = df2['depth']*df2['abs_slope']
df2['distance_abs_slope'] = df2['distance']*df2['abs_slope']




X = df2.values[:,[0,1,2,5,6,7,8,9,10,11,12,13,14,15,16]] # dropped the none, so decided not to use ratio_to_center
classification_letters = df2.values[:,4]


# classification will be held in variable y and map as follows
# farthest_distance_below_top_line       ('f') - 0
# find_top_line_intersection_points      ('i') - 1
# project_from_midpoint_of_2_predictions ('m') - 2
# any                                    ('a') - 2
y = np.zeros((classification_letters.shape[0],))
for i in xrange(classification_letters.shape[0]):
    letter = classification_letters[i]
    if letter=='f':
        y[i]=0
    if letter=='i':
        y[i]=1
    if letter=='m':
        y[i]=2
    if letter=='a':
        y[i]=2

# train the classifer
multi_class='multinomial'
clf = LogisticRegression(solver='newton-cg', max_iter=10000, random_state=42, multi_class=multi_class).fit(X, y)
print("training score : %.3f (%s)" % (clf.score(X, y), multi_class)) # ~80.2%

# pickle the classifier
with open('which_top_point_classifier.pickle','wb') as f:
    pickle.dump(clf,f)



#!!! This is the full program, just used as a testing file
# modify original program so it has all the engineered features,
# make predictions and take the TWO most likely predictions on either side and
# save 4 pictures with all the combinations