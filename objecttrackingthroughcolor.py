import cv2
import numpy as np

def main():
    Windowname = "preview"
    cv2.namedWindow(Windowname)
    cap = cv2.VideoCapture(0)


    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    while ret:
      ret,frame = cap.read()
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      #blue
      #low = np.array([100,50,50])
      #high = np.array([140,255,255])
      #green
      low = np.array([40,50,50])
      high = np.array([80,255,255])
      #red
      #low = np.array([140,150,0])
      #high = np.array([180,255,255])
      image_mask = cv2.inRange(hsv, low, high)
      output = cv2.bitwise_and(frame,frame, mask=image_mask)
      cv2.imshow("imagemask",image_mask)
      cv2.imshow("orignal webcam feed", frame)
      cv2.imshow("color tracking", output)


      if cv2.waitKey(1) == 27:
        break
    cv2.destroyAllWindows()
    cap.release()
if __name__ == "__main__":
    main()

