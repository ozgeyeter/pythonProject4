import cv2
import numpy as np
import os

Camera = cv2.VideoCapture(1)
kernel = np.ones((12, 12), np.uint8) #tarama ,filtreleme


def PictureFindDifferent(Picture1, Picture2):
    Picture2 = cv2.resize(Picture2, (Picture1.shape[1], Picture1.shape[0])) #2.resım hacmını,boyutunu degıstırıyo
    Different_Picture = cv2.absdiff(Picture1, Picture2) #2 resım arasındakı farkı cıkarıp
    Different_Number = cv2.countNonZero(Different_Picture) #siyah olmayan pıkselleri sayıyo
    return Different_Number #saydıktan sonra donduruyo


def UploadData(): # kaydedilen datayı alıp resimleri ve isimleri liste olarak kaydediyor
    Data_Names = []
    Data_Pictures = []

    Folders = os.listdir("data/") #datanın altındakı tum klasorlerı dızıyo
    for Folder in Folders: #son klasorlerı gezıyo
        Data_Names.append(Folder.replace(".jpg", "")) #resimlerin isimlerini alıyo ve listeye atıyo
        Data_Pictures.append(cv2.imread("data/" + Folder, 0)) #resmın kendısını alıp klasorune atıyo


    return Data_Names, Data_Pictures #daha sonra 2 listeyı donduruyo


def Classification(Picture, Data_Names, Data_Pictures):#sınıflandırma yapıyo
    Min_Index = 0
    Min_Value = PictureFindDifferent(Picture, Data_Pictures[0])#2 resım arasındakı en az farkı alıyo
    for t in range(len(Data_Names)): #klasordekı resımlerın ısımlerını gezıyo
        Different_Value = PictureFindDifferent(Picture, Data_Pictures[t])
        if (Different_Value < Min_Value): #klasordekı ılk resmı alıyo, 2 resmı karsılastırıyo aralarındakı en kucuk değeri koyuyo

            Min_Value = Different_Value #burada tanımlandı

            Min_Index = t #en kucuk deger atandı

            return Data_Names[Min_Index] #en kucuk degerı donduruyo



Data_Names, Data_Pictures = UploadData()  #datayı yukluyo

while True:
    ret, Square = Camera.read() #kamerayı okuyo
    Cut_Square = Square[0:200,0:250]  #cerceveyı ayarlıyo
    Cut_Square_Gri = cv2.cvtColor(Cut_Square, cv2.COLOR_BGR2GRAY) #gri oluyor
    Cut_Square_HSV = cv2.cvtColor(Cut_Square, cv2.COLOR_BGR2HSV) #renk uzayı

    Min_Values = np.array([0, 20, 40])
    Max_Values = np.array([40, 255, 255])

    Color_Filter_Result = cv2.inRange(Cut_Square_HSV, Min_Values, Max_Values)
    Color_Filter_Result = cv2.morphologyEx(Color_Filter_Result, cv2.MORPH_CLOSE, kernel) #goruntu genıslemesı
    Color_Filter_Result = cv2.dilate(Color_Filter_Result, kernel, iterations=1) #beyaz bolgeyı artırıyo ve on plana cıkartıro

    Result = Cut_Square.copy()
    cnts,_ = cv2.findContours(Color_Filter_Result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #goruntunun dıs hatlarını gosterıyo
    Max_Width = 0
    Max_Length = 0
    Max_Index  = -1

    for t in range(len(cnts)): #contorların uzerınde gezıyo
        if (len(cnts) > 0): #konturlar doluysa
            cnt=cnts[t]  #buraya atanıyo
            x,y,w,h = cv2.boundingRect(cnt)
            if(w>Max_Width and h>Max_Length): # 0 dan buyukse max w ve max l nın gercek boyutlarını ele alıo

                Max_Length=h
                Max_Width =w
                Max_Index=t
        if (len(cnts)>0): #boyutlara gore kucuk pencereyı gosterıyo

            x,y,w,h=cv2.boundingRect(cnts[Max_Index])
            cv2.rectangle(Result,(x,y),(x+w,y+h),(0,255,0),2)
            Hand_Picture = Color_Filter_Result[y:y+h,x:x+w]
            cv2.imshow("Hand Picture", Hand_Picture)
            print(Classification(Hand_Picture,Data_Names,Data_Pictures)) # yazdırıyo


        cv2.imshow('Square', Square)
        cv2.imshow("Cut Square", Cut_Square)
        cv2.imshow("Color Filter Result", Color_Filter_Result)
        cv2.imshow("Result", Result)

        if cv2.waitKey(1) & 0xFF== ord('q'):
            Camera.release()
            cv2.destroyAllWindows()
            break

