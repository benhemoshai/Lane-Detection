import numpy as np 
import cv2

# Global variables to store the last detected lines
last_left_line = None
last_right_line = None

#Find the slope and intercept of the left and right lanes of each image.
def average_slope_intercept(lines):
    
    if lines is None:
        return None, None
    
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

#Converts the slope and intercept of each line into pixel points.   
def pixel_points(y1, y2, line):
   
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

#Create full length lines from pixel points.   
def lane_lines(image, lines):

    global last_left_line, last_right_line
    
    left_lane, right_lane = average_slope_intercept(lines)
    
    # If current lines are not detected, use the last detected lines
    if left_lane is None and last_left_line is not None:
        left_lane = last_left_line
    if right_lane is None and last_right_line is not None:
        right_lane = last_right_line
        
    last_left_line = left_lane
    last_right_line = right_lane
    
    y1 = image.shape[0]
    y2 = y1 * 0.58
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)


    return left_line, right_line

#Draws lines and a trapezoid polygon on the input image.     
def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=12):
    
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    
    #Adds trapezoid polygon
    global last_left_line, last_right_line
    
    left_lane, right_lane = last_left_line, last_right_line
    
    if left_lane is not None and right_lane is not None:
        y1 = image.shape[0]
        y2 = y1 * 0.58
        left_line  = pixel_points(y1, y2, left_lane)
        right_line = pixel_points(y1, y2, right_lane)
        
        # Defines trapezoid polygon based on the points of the lines
        left_top, left_bottom = left_line
        right_top, right_bottom = right_line
        points = np.array([[left_top, right_top, right_bottom, left_bottom]], dtype=np.int32)
        cv2.fillPoly(line_image, points, (0, 255, 0))
    
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

#Converts the image to gray 
def gray_im(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#Uses Gaussian Blur in order to reduce noise, using (5,5) kernel
def gaus_blur(frame):
    return cv2.GaussianBlur(gray_im(frame), (5, 5), 0)


#Thresholding the frame and crops it to a trapezoid shape in order to minimize the noise and to focus on the desired part of the road.
def crop_im(frame):
   
    _, white_mask = cv2.threshold(gaus_blur(frame), 240, 255, cv2.THRESH_BINARY)

    height, width = frame.shape[:2]
    roi_vertices = np.array([[(width * 0.05, height), (width * 0.48, height * 0.4),
                              (width * 0.355, height * 0.4), (width * 0.9, height)]], dtype=np.int32)

    roi_mask = np.zeros_like(white_mask)
    cv2.fillPoly(roi_mask, roi_vertices, 255)

    masked_lane = cv2.bitwise_and(white_mask, roi_mask)

    return masked_lane

#Uses Canny in order to detect the edges of the lane
def canny(frame):
    return cv2.Canny(crop_im(frame), 50, 150)

#Uses Hough Lines Transform to detect the lines
def hough_lines(frame):
    return cv2.HoughLinesP(canny(frame), 1, np.pi / 180, threshold=40, minLineLength=20, maxLineGap=200)


#Main
vid = cv2.VideoCapture('project1/night-drive.mp4')

if not vid.isOpened():
    print("Could not open video")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('night-drive-output.avi', fourcc, 20.0, (int(vid.get(3)), int(vid.get(4))))

while True:
    ret, frame = vid.read()
    if not ret:
        break

    #cv2.imshow("gray",canny(frame))
    lines = hough_lines(frame)

    left_line, right_line = lane_lines(frame, lines)
    result = draw_lane_lines(frame, [left_line, right_line])
    out.write(result)
    cv2.imshow('Lane Detection', result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
out.release()
cv2.destroyAllWindows()
