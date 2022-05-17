import cv2
import numpy as np
import os

Camera = cv2.VideoCapture(1)
kernel = np.ones((12, 12), np.uint8)
name = "one"
while True:
    ret,Square = Camera.read()
    Cut_Square = Square[0:200,0:250]
    Cut_Square_Gri = cv2.cvtColor(Cut_Square, cv2.COLOR_BGR2GRAY)
    Cut_Square_HSV = cv2.cvtColor(Cut_Square, cv2.COLOR_BGR2HSV)
    Min_Values = np.array([0, 20, 40])
    Max_Values = np.array([40, 255, 255])

    Color_Filter_Result = cv2.inRange(Cut_Square_HSV, Min_Values, Max_Values)
    Color_Filter_Result = cv2.morphologyEx(Color_Filter_Result, cv2.MORPH_CLOSE, kernel)
    Color_Filter_Result = cv2.dilate(Color_Filter_Result, kernel, iterations=1)

    Result = Cut_Square.copy()

    cnts,_= cv2.findContours(Color_Filter_Result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Max_Width = 0
    Max_Length = 0
    Max_Index = -1

    for t in range(len(cnts)):
        if (len(cnts) > 0):
            print(t)
            print("selam1")
            cnt = cnts[t]
            x, y, w, h = cv2.boundingRect(cnt)
            if (w > Max_Width and h > Max_Length):
                print("selam2")
                Max_Length = h
                Max_Width = w
                Max_Index = t
                if (len(cnts) > 0):
                    print("selam3")
                    x, y, w, h = cv2.boundingRect(cnts[Max_Index])
                    cv2.rectangle(Result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    Hand_Picture = Color_Filter_Result[y:y + h, x:x + w]
                    cv2.imshow("Hand Picture", Hand_Picture)
                    cv2.imshow('Square', Square)
                    cv2.imshow("Cut Square", Cut_Square)
                    cv2.imshow("Color Filter Result", Color_Filter_Result)
                    cv2.imshow("Result", Result)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.imwrite("data/" + name + ".jpg", Hand_Picture)
                        Camera.release()
                        cv2.destroyAllWindows()
                        break

