image_name = "1194_2x(1).jpg"
img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
original_width = img.shape[1]
original_height = img.shape[0]
img = convertToSquare(img)
img = resize288(img)
cv2.imshow("original", img)
edged = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,3)
cv2.imshow("edged",edged)

# Delete all of the loner white pixels
edged2 = edged.copy()
#for i in xrange(18,270):
 #   for j in xrange(287):
  #      if not(edged[i+1,j]==255 or edged[i-1,j]==255 or edged[i,j+1]==255 or edged[i,j-1]==255):
   #         edged2[i,j]=0
#cv2.imshow("edged2",edged2)

    
# The outline will now be obvious but there will be a nasty mess in the middle of white and black
# This will find the contours but not the ones we are looking for, due to the mess
cnts,_=cv2.findContours(edged2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# By drawing all the countours in as white, we begin to "fuse" the mess in the middle (which many of the contours follow)
cv2.drawContours(img,cnts,-1,(255,255,255),1)
cv2.imshow("contours",img)

# Now take apply a regular threshold function over the entire image. This will (with a few errors)
# make the cross section white and the background black
ret,th =cv2.threshold(img,50,255,cv2.THRESH_BINARY)
cv2.imshow("th",th)

# Because of how we reshape the photos, there will often now be an extra white line between the original image and the 
# boundary of where it was "placed" onto the 288/288 black background. Delete this line (and other noise)
h = 288*original_height/original_width
gap = int(288-h)/2

th[0:gap+25,:] = 0
th[287-gap-25:287,:] = 0

cv2.imshow("th2",th)

# clean up some outliers if they exist
th2 = th.copy()
#for i in xrange(gap+12,287-gap-12):
#    for j in xrange(287):
#        if th[i,j]==255:
#            if th[i+1,j]==255 and th[i-1,j]==0 or th[i+1,j]==0 and th[i-1,j]==255: 
#                th2[i,j]=255
#            else:
#                th2[i,j]=0

outline = np.zeros((288,288)) # initialize outline

# find the top edge of the outline
for j in xrange(287): # for each column:
    # find the top edge of the outline
    for i in xrange(287): # go down the column:
        if th2[i,j]==255: # if the pixel is white:
            outline[i,j] = 255 # make the pixel in the outline array white
            break
    # find the bottom edge of the outline
    for i in reversed(xrange(287)):
        if th2[i,j]==255: # if the pixel is white:
            outline[i,j] = 255 # make the pixel in the outline array white
            break

cv2.imshow("outline",outline)