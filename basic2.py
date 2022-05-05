from turtle import shape
import numpy as np
import cv2
import matplotlib.pyplot as plt

####
# Access pixel values
#####

img_path = "./imgs/cat_img_2.jpg"
cv_img = cv2.imread(img_path)
print(cv_img.shape)
x = 100
y = 100
single_pixel = cv_img[x,y]
print(f'Pixel at x: {x} and y: {y} has the value {single_pixel}')

# # Show the region of the pixel
# center_point = (x,y)
# radius = 5
# color = (0,0,255)
# thickness = 1
# img_with_circle = cv2.circle(cv_img, center_point, radius, color, thickness)
# cv2.imshow('Resized_img with a circle', img_with_circle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# blue_part = cv_img[:,:,1]
# cv2.imshow("Blue image", blue_part)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Remove the blue part
# cv_img[:,:,0] = 0
# cv2.imshow("Without Blue image", cv_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########
#Accessing image properties
#########

print(f'The shape of the image {cv_img.shape}')
print(f'The type used for saving  {cv_img.dtype}')
print(f'The size of the image {cv_img.size} Bytes')

##############
## Image Region of interest (ROI)
#############

# # part_of_img = cv_img[100: 350, 200: 550]
# # cv2.imshow('part_of_img', part_of_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # select a ROI
# roi = cv2.selectROI(cv_img) # Choose the regions
# print(roi)

# # Cropp the selectedc ROI from the region
# roi_cropped = cv_img[int(roi[1]): int(roi[1]) + int(roi[3]), int(roi[0]): int(roi[0]) + int(roi[2]),:]

# cv2.imshow("Cropped image", roi_cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


######
# Splitting and Merging image Channels
#########
# b,g,r  = cv2.split(cv_img) # This is a costly operation, better use numpy slicings
# cv2.imshow('blue_img', b)
# cv2.imshow('red_img', r)
# cv2.imshow('green_img', g)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#########
#Merging images
########
img_path_2 = "./imgs/cat_test_img.jpg"
cv_img_2 = cv2.imread(img_path_2)

# # Resize the images
# heigth = 500
# width = 400
# cv_img_1_resized = cv2.resize(cv_img,(heigth, width))
# cv_img_2_resized = cv2.resize(cv_img_2, (heigth, width))

# dst = cv2.addWeighted(cv_img_1_resized, 0.5, cv_img_2_resized, 0.5, 0)
# cv2.imshow('Blended_img', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########
#Applying filters to the image
#########
# Resize the images
heigth = 500
width = 400

cv_img_1_resized = cv2.resize(cv_img,(heigth, width))
#transfrom to grayscale
cv_img_1_resized = cv2.cvtColor(cv_img_1_resized, cv2.COLOR_BGR2GRAY)


canny_img = cv2.Canny(cv_img_1_resized, 20, 255 )
cv_img_2_resized = cv2.resize(cv_img_2, (heigth, width))
filt = np.array([[-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]])
sharp_img = cv2.filter2D(cv_img_1_resized, -1, filt)
# cv2.imshow("Original image", cv_img_1_resized)
# cv2.imshow("Image after filtering", sharp_img)


# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######
#Thresholding
#########
lower_limit = 100
upper_limit = 255
ret, mask = cv2.threshold(cv_img_1_resized, lower_limit, upper_limit, cv2.THRESH_BINARY)
#cv2.imshow("Thresholded image", mask)

#######
#Edge detection
########
canny_img = cv2.Canny(cv_img_1_resized, 20, 255 )
#cv2.imshow("Canny image", canny_img)



#######
#Contour detection
######
# shape_img_path = "./imgs/different_shapes.png"
# shape_img = cv2.imread(shape_img_path)
# shape_img = cv2.resize(shape_img, (800,700))
# shape_img_gray = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
# # setting threshold 
# lower_thr = 10
# upper_thr = 255

# ret,mask = cv2.threshold(shape_img_gray, lower_thr, upper_thr, cv2.THRESH_BINARY)

# # contour using findcountours function

# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# i = 0
# for contour in contours:
#     if i == 0:
#         i += 1
#         continue
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True )
#     cv2.drawContours(shape_img, [contour], 0, (255,0,255), 5)

#     # finding the center of different shapes
#     center = cv2. moments(contour)
#     if center['m00'] != 0.0:
#         x = int(center['m10']/center['m00'])
#         y = int(center['m01']/center['m00'])

#     #Placing names of the shape

#     if len(approx) == 3:
#         cv2.putText(shape_img, "Triangle", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
#         print(f"Detected Shape with {len(approx)} edges")
#     elif len(approx) == 4:
#         cv2.putText(shape_img, "Quadrilateral", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
#         print(f"Detected Shape with {len(approx)} edges")
#     elif len(approx) == 5:
#         cv2.putText(shape_img, "Pentagon", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
#         print(f"Detected Shape with {len(approx)} edges")
#     elif len(approx) == 6:
#         cv2.putText(shape_img, "Hexagon", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
#         print(f"Detected Shape with {len(approx)} edges")
#     else:
#         cv2.putText(shape_img, "Circle", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
#         print(f"Detected Shape with {len(approx)} edges")
# cv2.imshow("shape_image_thresholded", mask)
# cv2.imshow("Original image", shape_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

########
#Color detection
########
shape_img_path = "./imgs/different_shapes.png"
shape_img = cv2.imread(shape_img_path)

# Change color schema

#hsv = cv2.cvtColor(shape_img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0,0,0])
upper_blue = np.array([255,0,0])

# threhshold the HSV img to get only blue colors

mask_blue = cv2.inRange(shape_img, lower_blue, upper_blue)
res = cv2.bitwise_and(shape_img, shape_img, mask=mask_blue)

cv2.imshow("Mask for blue", mask_blue)
cv2.imshow('Thresholded blue only image', res)
cv2.imshow('Original image', shape_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
