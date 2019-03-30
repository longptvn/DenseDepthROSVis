import numpy as np
import cv2
import glob

ROW = 7
COL = 5
SQUARE = 30

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((COL*ROW,3), np.float32)
objp[:,:2] = np.mgrid[0:ROW,0:COL].T.reshape(-1,2)
objp *= SQUARE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('*.jpg')

cap = cv2.VideoCapture(0)

cnt = 0

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('img',img)

    key = cv2.waitKey(10)

    if key == 27 or cnt == 16:
      break

    elif key == ord('t'):

      # Find the chess board corners
      ret, corners = cv2.findChessboardCorners(gray, (ROW,COL),None)

      # If found, add object points, image points (after refining them)
      if ret == True:

          print("found objects")

          objpoints.append(objp)

          corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
          imgpoints.append(corners2)

          # Draw and display the corners
          img2 = cv2.drawChessboardCorners(img, (ROW,COL), corners2,ret)
          cv2.imshow('img',img2)

          cnt = cnt + 1

          cv2.waitKey(200)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(mtx)

while True:
  _, img = cap.read()
  h,  w = img.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
  # undistort
  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  # crop the image
  x,y,w,h = roi
  dst = dst[y:y+h, x:x+w]

  cv2.imshow('undistort', dst)
  cv2.imshow('org',img)

  key = cv2.waitKey(10)

  if key == 27:
    break

cv2.destroyAllWindows()
