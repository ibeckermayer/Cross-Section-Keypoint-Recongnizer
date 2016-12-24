import cv2
import numpy as np
import os
# This Script Takes all images from the RawImages directory, converts them to grayscale format,
# makes them square sized, and then resizes them to 288 x 288 and saves them in the PreProcessedImages directory


# Take a cv2 read grayscale image and make it square size by adding extra black

def convertToSquare(image):
    height, width = image.shape

    # Make a black square with an even number of rows and columns, within 1 row of image width
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

directory = "RawImages/"
for filename in os.listdir(directory):
    if filename.endswith(".bmp") or filename.endswith(".jpg"):
        print filename
        file = os.path.join(directory, filename)
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = convertToSquare(image)
        image = resize288(image)
        filename_length = len(filename)
        new_filename = filename[0:filename_length-3] + "jpg"
        cv2.imwrite('PreProcessedImages/'+new_filename,image)
        os.rename(file,'RawImagesAlreadyPreProcessed/' + filename)