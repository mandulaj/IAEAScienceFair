import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation

cap = cv2.VideoCapture(0)
flop = 0
while True:
  flop += 1
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow("color", frame)
  gray = cv2.flip(gray,flop%4)
  cv2.imshow("gray", gray)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break



#def animate(i):
    #ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.flip(gray,0)
    #im.set_data(frame)
    #return im
    #cv2.imshow("frame1", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

#cap.release()


#anim = animation.FuncAnimation(fig, animate, interval=50, frames=100000)
