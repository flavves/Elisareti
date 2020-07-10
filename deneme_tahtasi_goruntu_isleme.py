# -*- coding: utf-8 -*-
"""
Deneme tahtası burası
"""


import cv2
import numpy as np
import os




kamera = cv2.VideoCapture(0)
kernel = np.ones((12,12),np.uint8)







def resimFarkBul(resim1,resim2):
    resim2= cv2.resize(resim2,(resim1.shape[1],resim1.shape[0]))
    fark_resim= cv2.absdiff(resim1,resim2)
    fark_sayi = cv2.countNonZero(fark_resim)
    return fark_sayi

veri_resim1= cv2.imread("veri/bir.jpg",0)



def VeriYukle():
    veri_isimler = []
    veri_resimler = []
    
    Dosyalar = os.listdir("veri/")
    for Dosya in Dosyalar:
        veri_isimler.append(Dosya.replace(".jpg",""))
        veri_resimler.append(cv2.imread("veri/"+Dosya,0))
        
    return veri_isimler,veri_resimler

veri_isimler,veri_resimler=VeriYukle()


def sınıflandır(resim,veri_isimler,veri_resimler):
    min_Index = 0
    min_deger = resimFarkBul(resim,veri_resimler[0])
    for t in range(len(veri_isimler)):
        fark_deger= resimFarkBul(resim,veri_resimler[t])
        if(fark_deger<min_deger):
            min_deger=fark_deger
            min_Index=t
    return veri_isimler[min_Index]



while True:
    ret, kare=kamera.read()
    kesilmis_kare = kare[0:200,0:250]
    kesilmis_kare_gri= cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2GRAY)
    kesilmis_kare_HSV= cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2HSV)
    
    alt_degerler = np.array([0,20,40])
    ust_dgerler = np.array([40,255,255])
    
    renk_filtresi_sonuc=cv2.inRange(kesilmis_kare_HSV,alt_degerler,ust_dgerler)
    renk_filtresi_sonuc = cv2.morphologyEx(renk_filtresi_sonuc, cv2.MORPH_CLOSE, kernel)
    
    sonuc = kesilmis_kare.copy()
    (cnts, _)= cv2.findContours(renk_filtresi_sonuc,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    max_genislik=0
    max_uzunluk=0
    max_Index=-1
    
    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h=cv2.boundingRect(cnt)
        if (w>max_genislik and h>max_uzunluk):
            max_uzunluk = h
            max_genislik =w
            max_Index=t
        
    
    
    if (len(cnts)>0):
        x,y,w,h=cv2.boundingRect(cnts[max_Index])
        cv2.rectangle(sonuc,(x,y),(x+w,y+h),(0,255,0),2)
        el_resim = renk_filtresi_sonuc[y:y+h,x:x+w]
        cv2.imshow("el resim",el_resim)
        print(sınıflandır(el_resim,veri_isimler,veri_resimler))

    
    
    cv2.imshow("kare",kare)
    cv2.imshow("kesme", kesilmis_kare)
    cv2.imshow("renk filtresi sonuc",renk_filtresi_sonuc)
    cv2.imshow("sonuc", sonuc)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
    



"""

Yaptıklarımızı kaydeder
bursası



"""





import cv2
import numpy as np
import os

isim="ben"


kamera = cv2.VideoCapture(0)
kernel = np.ones((12,12),np.uint8)












while True:
    ret, kare=kamera.read()
    kesilmis_kare = kare[0:200,0:250]
    kesilmis_kare_gri= cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2GRAY)
    kesilmis_kare_HSV= cv2.cvtColor(kesilmis_kare,cv2.COLOR_BGR2HSV)
    
    alt_degerler = np.array([0,20,40])
    ust_dgerler = np.array([40,255,255])
    
    renk_filtresi_sonuc=cv2.inRange(kesilmis_kare_HSV,alt_degerler,ust_dgerler)
    renk_filtresi_sonuc = cv2.morphologyEx(renk_filtresi_sonuc, cv2.MORPH_CLOSE, kernel)
    renk_filtresi_sonuc = cv2.dilate(renk_filtresi_sonuc,kernel,iterations=1)

    
    sonuc = kesilmis_kare.copy()
    (cnts, _)= cv2.findContours(renk_filtresi_sonuc,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    max_genislik=0
    max_uzunluk=0
    max_Index=-1
    
    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h=cv2.boundingRect(cnt)
        if (w>max_genislik and h>max_uzunluk):
            max_uzunluk = h
            max_genislik =w
            max_Index=t
        
    
    
    if (len(cnts)>0):
        x,y,w,h=cv2.boundingRect(cnts[max_Index])
        cv2.rectangle(sonuc,(x,y),(x+w,y+h),(0,255,0),2)
        el_resim = renk_filtresi_sonuc[y:y+h,x:x+w]
        cv2.imshow("el resim",el_resim)

    
    
    cv2.imshow("akare",kare)
    cv2.imshow("kesme", kesilmis_kare)
    cv2.imshow("renk filtresi sonuc",renk_filtresi_sonuc)
    cv2.imshow("sonuc", sonuc)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.imwrite("veri/"+isim+".jpg",el_resim)
kamera.release()
cv2.destroyAllWindows()
    












