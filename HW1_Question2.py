from tkinter import W
import cv2
import numpy as np 
import matplotlib.pyplot as plt


x_data = []
y_data = []


def standard_LS(x_data,y_data):
    x = [i**2 for i in x_data]
    test = np.vstack([np.ones])
    a = np.vstack([np.ones(len(x_data)), x_data, x]).T
    b = np.dot(np.linalg.inv(np.dot(a.T, a)),(a.T))
    return np.dot(b,y_data)
    #return np.linalg.lstsq(a,y_data)

def data(image):

    #h,w,c = image.shape()
    coordinates = np.where(image==0)
    index_max = np.argmax(coordinates[0])
    index_min = np.argmin(coordinates[0])
    X = coordinates[0]
    Y = coordinates[1]
    return X[index_min], X[index_max], Y[index_min], Y[index_max]
    

cap = cv2.VideoCapture('ball_video1.mp4')
#cap = cv2.VideoCapture('ball_video2.mp4')
if (cap.isOpened()== False):
  print("Error opening video stream or file")
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    red_channel = frame[:,:,2]
    (thresh, im_bw) = cv2.threshold(red_channel, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    values = data(im_bw)
    x_data.append((values[0]+values[1])/2)
  # x_data.append(values[1])
    y_data.append((values[2]+values[3])/2)
   #y_data.append(values[3])
    display = cv2.resize(im_bw, (960,540))  
    cv2.imshow('Frame',display)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break


cap.release()
cv2.destroyAllWindows()
x_data = [(1676-j) for j in x_data]
w = standard_LS(y_data,x_data)
x = np.linspace(0,3500,100)
y = w[0]+w[1]*x+w[2]*x*x


print(x_data)
print(y_data)
plt.scatter(y_data,x_data)
plt.plot(x,y)
plt.show()