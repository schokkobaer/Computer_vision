import numpy as np
import cv2 
###########
# Loading an image
############
img_path = "./imgs/cat_img_2.jpg"
cv_img = cv2.imread(img_path)

#Displaying an image
#cv2.imshow('Simple cat image',cv_img)
# Wait for an input to close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#############
# Saving an image
#############
safe_path_dir = "./created_data/"
filename = "saved_img.jpg"
cv2.imwrite(safe_path_dir + filename, cv_img)
print("Image was succesfully saved")

#Get the property of an image
print(cv_img.shape)

#########
# Changel color scheme
#########
gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray scale image', gray_img)
#cv2.imshow('Original Image', cv_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#############
# Change size of an image
############
resized_img = cv2.resize(cv_img,(400,400))

#############
# Displaying text on an img
############
img_with_text = cv2.putText(resized_img, "My Cat", (0, 200), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0,0), 2)

###########
# Draw a line on a image
############
start_point = (0,0)
end_point = (400,200)
color = (0,255,0)
thickness = 4
img_with_line = cv2.line(resized_img, start_point, end_point, color, thickness)
# cv2.imshow('Resized_img with line', img_with_line)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##############
# Draw a circle on an image
##############
center_point = (200,200)
radius = 50
color = (0,255,0)
thickness = 4
img_with_circle = cv2.circle(resized_img, center_point, radius, color, thickness)
# cv2.imshow('Resized_img with a circle', img_with_circle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##############
# Draw a rectangle on an image
##############
pt1 = (100,100)
pt2 = (150,250)
color = (0,255,0)
thickness = 4
img_with_rectangle = cv2.rectangle(resized_img,pt1, pt2, color, thickness)
# cv2.imshow('Resized_img with a rectangel', img_with_rectangle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


################
# Draw an ellipse on the image
################
center_point = (300,300)
main_axis_size = (int(100/2), int(50/2))
rot_angle_dgr = 0
start_angle = 0
end_angle = 360
color = (0,255,0)
thickness = -1 # Fill the ellipses
# img_with_ellipse = cv2.ellipse(resized_img,center_point, main_axis_size, rot_angle_dgr, start_angle, end_angle, color, thickness)
# cv2.imshow('Resized_img with ellipse', img_with_ellipse)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######
#Display the image in multiple mode (other option)
#######
cv_img_mode = cv2.imread(img_path, 0) # Greyscale

#Displaying an image
#cv2.imshow('Simple cat image',cv_img_mode)
# Wait for an input to close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()


########
# Videocamera
########

####
# Start the camera
######

# cap = cv2.VideoCapture(0) # Will initiate the webcam
# # while(True): # Start a loop to get every fram
# #     ret, frame = cap.read()
# #     cv2.imshow("Video Capture", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         cap.release()
# #         break
# # cv2.destroyAllWindows()     

# # Check if the video works
# if (cap.isOpened() == False):
#     print("Camera could not be found")
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))


# # video coded .... encoder and decoders

# video_cod = cv2.VideoWriter_fourcc(*'XVID')
# video_output = cv2.VideoWriter('Captured_video.mp4', video_cod, 30, (frame_width, frame_height))

# while(True): # Start a loop to get every fram
#     ret, frame = cap.read()
#     if ret == False:
#         cap.release()
#         break
#     video_output.write(frame)
#     cv2.imshow("Video Capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         break
# cv2.destroyAllWindows()
# print("The video was saved") 

####
# Play a saved video
######
cap = cv2.VideoCapture('Captured_video.mp4')
while(True): # Start a loop to get every fram
    ret, frame = cap.read()
    cv2.imshow("Video Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
cv2.destroyAllWindows()     