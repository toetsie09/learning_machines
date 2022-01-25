import os
import pickle
from typing import Dict
import numpy as np
import cv2
from matplotlib import pyplot as plt
from robot_interface import RoboboEnv
#(hMin = 24 , sMin = 65, vMin = 3), (hMax = 149 , sMax = 255, vMax = 163)

#exit()

def nothing(x):
    pass

colorThresholdFile= './src/calibration/camera/colorThreshold.pickle'
blobParametersFile= './src/calibration/camera/blobParameters.pickle'

# Load image
#image = cv2.imread("./src/view/food/test_photo_0.png")
#image = cv2.imread("./src/view/food/test_photo_1.png")
#image = cv2.imread("./src/view/food/test_photo_3.png")
image = cv2.imread("./src/view/reference photos/test_photo.png")

windowTitle ="HSV color calibration: Q to quit"
window_2_Title ="Blob detection calibration: Q to quit"
# Create a window
cv2.namedWindow(windowTitle) #HSV
cv2.namedWindow(window_2_Title) #blob

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', windowTitle, 0, 179, nothing)
cv2.createTrackbar('SMin', windowTitle, 0, 255, nothing)
cv2.createTrackbar('VMin', windowTitle, 0, 255, nothing)
cv2.createTrackbar('HMax', windowTitle, 0, 179, nothing)
cv2.createTrackbar('SMax', windowTitle, 0, 255, nothing)
cv2.createTrackbar('VMax', windowTitle, 0, 255, nothing)

# Set default value for Max HSV trackbars
if os.path.isfile(colorThresholdFile):
    loadedCT = pickle.load( open( colorThresholdFile, "rb" ) )

    cv2.setTrackbarPos('HMin', windowTitle, loadedCT["lowerBound"][0] )
    cv2.setTrackbarPos('SMin', windowTitle, loadedCT["lowerBound"][1] )
    cv2.setTrackbarPos('VMin', windowTitle, loadedCT["lowerBound"][2] )

    cv2.setTrackbarPos('HMax', windowTitle, loadedCT["upperBound"][0] )
    cv2.setTrackbarPos('SMax', windowTitle, loadedCT["upperBound"][1] )
    cv2.setTrackbarPos('VMax', windowTitle, loadedCT["upperBound"][2] )
    #exit()
else:
    cv2.setTrackbarPos('HMax', windowTitle, 179)
    cv2.setTrackbarPos('SMax', windowTitle, 255)
    cv2.setTrackbarPos('VMax', windowTitle, 255)

# Create trackbars for blob change
cv2.createTrackbar('-Threshold', window_2_Title, 0, 255, nothing)
cv2.createTrackbar('+Threshold', window_2_Title, 0, 255, nothing)
cv2.createTrackbar('-Area', window_2_Title, 1, 23_040, nothing)
cv2.createTrackbar('+Area', window_2_Title, 1, 23_040, nothing)
cv2.createTrackbar('-Circularity', window_2_Title, 0, 1000, nothing)
cv2.createTrackbar('+Circularity', window_2_Title, 0, 1000, nothing)
cv2.createTrackbar('-Convexity', window_2_Title, 0, 1000, nothing)
cv2.createTrackbar('+Convexity', window_2_Title, 0, 1000, nothing)
cv2.createTrackbar('-InertiaRatio', window_2_Title, 0, 1000, nothing)
cv2.createTrackbar('+InertiaRatio', window_2_Title, 0, 1000, nothing)


parNames= ['minThreshold', 'maxThreshold', 
'minArea', 'maxArea',
'minCircularity', 'maxCircularity',
'minConvexity', 'maxConvexity',
'minInertiaRatio', 'maxInertiaRatio']

trackbarNames=['-Threshold', '+Threshold', 
'-Area', '+Area',
'-Circularity', '+Circularity',
'-Convexity', '+Convexity',
'-InertiaRatio', '+InertiaRatio']

# Set default value for Blob trackbars
if os.path.isfile(blobParametersFile):
    loadedBlobT = pickle.load( open( blobParametersFile, "rb" ) )

    for pname,tbarName in zip(parNames[:4],trackbarNames[:4]):
        cv2.setTrackbarPos(tbarName, window_2_Title, loadedBlobT[pname])

    for pname,tbarName in zip(parNames[4:],trackbarNames[4:]):
        # float vars to int times 1000
        cv2.setTrackbarPos(tbarName, window_2_Title, int(loadedBlobT[pname]*1000))
else:
    cv2.setTrackbarPos('+Threshold', window_2_Title, 255)
    cv2.setTrackbarPos('+Area', window_2_Title, 230_400)
    cv2.setTrackbarPos('+Circularity', window_2_Title, 1000)
    cv2.setTrackbarPos('+Convexity', window_2_Title, 1000)
    cv2.setTrackbarPos('+InertiaRatio', window_2_Title, 1000)

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', windowTitle)
    sMin = cv2.getTrackbarPos('SMin', windowTitle)
    vMin = cv2.getTrackbarPos('VMin', windowTitle)
    hMax = cv2.getTrackbarPos('HMax', windowTitle)
    sMax = cv2.getTrackbarPos('SMax', windowTitle)
    vMax = cv2.getTrackbarPos('VMax', windowTitle)

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display result image
    #cv2.imshow(windowTitle, result)
    cv2.imshow("color threshold", result)
    #cv2.imshow("color mask", mask)
    #---------------------------------------------

    params= {pname:cv2.getTrackbarPos(tbarName, window_2_Title) for pname,tbarName in zip(parNames[:4],trackbarNames[:4])}
    params= {**params,**{pname:cv2.getTrackbarPos(tbarName, window_2_Title)/1000.0 for pname,tbarName in zip(parNames[4:],trackbarNames[4:])}}

    inverseMask = cv2.bitwise_not(mask)
    keypoints, im_with_keypoints= RoboboEnv.BlobDetect(inverseMask, **params)
    cv2.imshow("Blob detection", im_with_keypoints)


    #---------------------------------------------
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print(f"Saving: color thresholds\n\tlowerBound= {(hMin,sMin,vMin)}\n\tupperBound= {(hMax,sMax,vMax)}\n")
        with open(colorThresholdFile, 'wb') as handle:
            pickle.dump({"lowerBound":(hMin,sMin,vMin),"upperBound":(hMax,sMax,vMax)}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saving: blob tresholds\n\t{params}\n")
        with open(blobParametersFile, 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        break #dont remove plz!

#closing all open windows 
cv2.destroyAllWindows() 