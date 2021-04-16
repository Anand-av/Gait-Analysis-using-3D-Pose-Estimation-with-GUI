#########################		Libraries Import 		#########################
import tkinter as tk
from tkinter import *
from tkinter.ttk import Combobox
import numpy as np
import pandas as pd
import subprocess, os, shlex
import time
from scipy.spatial import distance
import statistics as st
from random import choice
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle
from warnings import filterwarnings
from PIL import ImageTk, Image, ImageTk
from itertools import count
from pandas.plotting import table
filterwarnings('ignore')

window=Tk()
canvas = Canvas(window, bg='blue', width=200, height=200)

def Normal():
    os.chdir('D:\\Sem 4\\Proj\\real_v3\\Oct22')
    #messagebox.showinfo('Normal')
    df_nor = pd.read_csv('SG_Normal.csv')
    df_train = df_nor[['Steplength', 'Stepwidth', 'Gaitspeed']]
    df_tralbl = df_nor[['Class']]    
    df_test = pd.read_csv('test_data_nor.csv')
    #df_test = pd.read_csv('realtest.csv')
    #test_label = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])    
    test_label = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    log_reg = LogisticRegression()
    log_reg.fit(df_train, df_tralbl)
    predict_logreg = log_reg.predict(df_test)
    acc_logreg = metrics.accuracy_score(test_label, predict_logreg)
    acc_logreg = round(acc_logreg, 4)*100
    print('The accuracy obtained using Logistic regression for Normal is', acc_logreg)
    
    naive_bayes = GaussianNB()
    naive_bayes.fit(df_train, df_tralbl)
    predict_nb = naive_bayes.predict(df_test)
    acc_nb = metrics.accuracy_score(test_label, predict_nb)
    acc_nb = round(acc_nb, 4)*100
    print('The accuracy obtained using Naive Bayes for Normal is', acc_nb)
    
    messagebox.showinfo('NORMAL PERSON')
    
def CB3():
    os.chdir('D:\\Sem 4\\Proj\\real_v3\\Oct22')
    df_nor = pd.read_csv('SG_CB3.csv')
    df_train = df_nor[['Steplength', 'Stepwidth', 'Gaitspeed']]
    df_tralbl = df_nor[['Class']]
    df_test = pd.read_csv('test_data_cb3.csv')
    #df_test = pd.read_csv('realtest.csv')
    #test_label = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    test_label = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
    log_reg = LogisticRegression()
    log_reg.fit(df_train, df_tralbl)
    predict_logreg = log_reg.predict(df_test)
    acc_logreg = metrics.accuracy_score(test_label, predict_logreg)
    acc_logreg = round(acc_logreg, 4)*100
    print('The accuracy obtained using Logistic regression for CB3 is', acc_logreg)    
    naive_bayes = GaussianNB()
    naive_bayes.fit(df_train, df_tralbl)
    predict_nb = naive_bayes.predict(df_test)
    acc_nb = metrics.accuracy_score(test_label, predict_nb)
    acc_nb = round(acc_nb, 4)*100
    print('The accuracy obtained using Naive Bayes for CB3 is', acc_nb)    
    messagebox.showinfo('NORMAL PERSON')
    
def CB7():
    os.chdir('D:\\Sem 4\\Proj\\real_v3\\Oct22')
    df_nor = pd.read_csv('SG_CB7.csv')
    df_train = df_nor[['Steplength', 'Stepwidth', 'Gaitspeed']]
    df_tralbl = df_nor[['Class']]
    df_test = pd.read_csv('test_data_cb7.csv')
    #df_test = pd.read_csv('realtest.csv')
    #test_label = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    test_label = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  
    log_reg = LogisticRegression()
    log_reg.fit(df_train, df_tralbl)
    predict_logreg = log_reg.predict(df_test)
    acc_logreg = metrics.accuracy_score(test_label, predict_logreg)
    acc_logreg = round(acc_logreg, 4)*100
    print('The accuracy obtained using Logistic regression for CB7 is', acc_logreg)    
    naive_bayes = GaussianNB()
    naive_bayes.fit(df_train, df_tralbl)
    predict_nb = naive_bayes.predict(df_test)
    acc_nb = metrics.accuracy_score(test_label, predict_nb)
    acc_nb = round(acc_nb, 4)*100
    print('The accuracy obtained using Naive Bayes for CB7 is', acc_nb)
    messagebox.showinfo('NORMAL PERSON')
    
def checkcombo():

    if cmb1.get() == 'Normal':
        Normal()
        
    elif cmb1.get() == 'CB3':
        CB3()
    
    else:
        CB7()
        
def cmd_prmt():
    #subprocess.Popen('D:\\Gait Analysis\\Final_Nov19\\gait_bat.BAT')    
    #os.system("cmd.exe")
    os.system("start /B start cmd.exe /K CD /")
    
def anim_image():
    
    imagelist = ["test0.png", "test1.png","test2.png", "test3.png","test4.png", "test5.png", "test6.png",
                 "test7.png", "test8.png", "test9.png", "test10.png", "test11.png", "test12.png", "test13.png",
                 "test14.png", "test15.png", "test16.png", "test17.png", "test18.png", "test19.png"]
                 
    # extract width and height info
    photo = ImageTk.PhotoImage(Image.open(imagelist[0]))
    #photo = PhotoImage(file=imagelist[0])
    width = photo.width()
    height = photo.height()
    canvas = Canvas(width=width, height=height)
    canvas.pack()
    # create a list of image objects
    giflist = []
    for imagefile in imagelist:
        photo = PhotoImage(file=imagefile)
        giflist.append(photo)
    # loop through the gif image objects for a while
    for k in range(0, 1000):
        for gif in giflist:
            canvas.delete(ALL)
            canvas.create_image(width/2.0, height/2.0, image=gif)
            canvas.update()
            time.sleep(0.1)
            
def feat_ext():
    lines = []
    with open('C:\\Users\\carti\\Desktop\\tf-pose\\src\\pose_3d.txt') as f:
        lines = f.read().splitlines()    
    lis_pos = []
    lis_x = []
    lis_y = []
    lis_z = []
    zip_lis = []    
    for line in lines:
        line = eval(line)        
        x = line[0]
        y = line[1]
        z = line[2]
        lis_pos.append(line)
        lis_x.append(x)
        lis_y.append(y)
        lis_z.append(z)
    
    nor_f1_1 = lis_x[0] [10], lis_y[0] [10], lis_z[0] [10]
    nor_f1_l = list(nor_f1_1)
    f1_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f1_2 = lis_x[0] [14], lis_y[0] [14], lis_z[0] [14]
    nor_f1_r = list(nor_f1_2)
    f1_right_ankle = [x / 80 for x in nor_f1_r]  
    f1_rigank_arr = np.array(f1_right_ankle)
    f1_rigank_tpse = f1_rigank_arr.T    
    f1_lefank_arr = np.array(f1_left_ankle)    
    nor_f1_step = np.dot(f1_lefank_arr, f1_rigank_tpse)    
    sw1 = lis_x[0] [10], lis_y[0] [10], lis_z[0] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[0] [13], lis_y[0] [13], lis_z[0] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]    
    dst_nor1 = distance.euclidean(nor_sw1, nor_sw2)    
    cad_stp = 55 #No of steps    
    cad_f1 = cad_stp/1800
    speed_f1 = nor_f1_step * cad_f1
    nor_f2_1 = lis_x[1] [10], lis_y[1] [10], lis_z[1] [10]
    nor_f2_l = list(nor_f1_1)    
    f2_left_ankle = [x / 80 for x in nor_f2_l]
    nor_f2_2 = lis_x[1] [14], lis_y[1] [14], lis_z[1] [14]
    nor_f2_r = list(nor_f1_2)
    f2_right_ankle = [x / 80 for x in nor_f2_r]
    f2_rigank_arr = np.array(f2_right_ankle)
    f2_rigank_tpse = f2_rigank_arr.T    
    f2_lefank_arr = np.array(f2_left_ankle)
    nor_f2_step = np.dot(f2_lefank_arr, f2_rigank_tpse)    
    sw1 = lis_x[1] [10], lis_y[1] [10], lis_z[1] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[1] [13], lis_y[1] [13], lis_z[1] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor2 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps    
    cad_f1 = cad_stp/1800
    speed_f2 = nor_f2_step * cad_f1
    nor_f3_1 = lis_x[2] [10], lis_y[2] [10], lis_z[2] [10]
    nor_f3_l = list(nor_f1_1)
    f3_left_ankle = [x / 80 for x in nor_f3_l]    
    nor_f3_2 = lis_x[2] [14], lis_y[2] [14], lis_z[2] [14]
    nor_f3_r = list(nor_f3_2)
    f3_right_ankle = [x / 80 for x in nor_f3_r]
    f3_rigank_arr = np.array(f3_right_ankle)
    f3_rigank_tpse = f3_rigank_arr.T
    f3_lefank_arr = np.array(f3_left_ankle)
    nor_f3_step = np.dot(f3_lefank_arr, f3_rigank_tpse)
    sw1 = lis_x[2] [10], lis_y[2] [10], lis_z[2] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[2] [13], lis_y[2] [13], lis_z[2] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor3 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f3 = nor_f3_step * cad_f1
    nor_f4_1 = lis_x[3] [10], lis_y[3] [10], lis_z[3] [10]
    nor_f4_l = list(nor_f4_1)
    f4_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f4_2 = lis_x[3] [14], lis_y[3] [14], lis_z[3] [14]
    nor_f4_r = list(nor_f4_2)
    f4_right_ankle = [x / 80 for x in nor_f1_r]
    f4_rigank_arr = np.array(f4_right_ankle)
    f4_rigank_tpse = f4_rigank_arr.T
    f4_lefank_arr = np.array(f4_left_ankle)
    nor_f4_step = np.dot(f4_lefank_arr, f4_rigank_tpse)
    sw1 = lis_x[3] [10], lis_y[3] [10], lis_z[3] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[3] [13], lis_y[3] [13], lis_z[3] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor4 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f4 = nor_f4_step * cad_f1
    nor_f5_1 = lis_x[4] [10], lis_y[4] [10], lis_z[4] [10]
    nor_f5_l = list(nor_f5_1)
    f5_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f5_2 = lis_x[4] [14], lis_y[4] [14], lis_z[4] [14]
    nor_f5_r = list(nor_f5_2)
    f5_right_ankle = [x / 80 for x in nor_f5_r]
    f5_rigank_arr = np.array(f5_right_ankle)
    f5_rigank_tpse = f5_rigank_arr.T
    f5_lefank_arr = np.array(f5_left_ankle)
    nor_f5_step = np.dot(f5_lefank_arr, f5_rigank_tpse)
    sw1 = lis_x[4] [10], lis_y[4] [10], lis_z[4] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[4] [13], lis_y[4] [13], lis_z[4] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor5 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f5 = nor_f5_step * cad_f1
    nor_f6_1 = lis_x[5] [10], lis_y[5] [10], lis_z[5] [10]
    nor_f6_l = list(nor_f6_1)
    f6_left_ankle = [x / 80 for x in nor_f6_l]
    nor_f6_2 = lis_x[5] [14], lis_y[5] [14], lis_z[5] [14]
    nor_f6_r = list(nor_f6_2)
    f6_right_ankle = [x / 80 for x in nor_f1_r]
    f6_rigank_arr = np.array(f6_right_ankle)
    f6_rigank_tpse = f6_rigank_arr.T
    f6_lefank_arr = np.array(f6_left_ankle)
    nor_f6_step = np.dot(f6_lefank_arr, f6_rigank_tpse)
    sw1 = lis_x[5] [10], lis_y[5] [10], lis_z[5] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[5] [13], lis_y[5] [13], lis_z[5] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor6 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f6 = nor_f6_step * cad_f1
    nor_f7_1 = lis_x[6] [10], lis_y[6] [10], lis_z[6] [10]
    nor_f7_l = list(nor_f7_1)
    f7_left_ankle = [x / 80 for x in nor_f7_l]
    nor_f7_2 = lis_x[6] [14], lis_y[6] [14], lis_z[6] [14]
    nor_f7_r = list(nor_f7_2)
    f7_right_ankle = [x / 80 for x in nor_f7_r]
    f7_rigank_arr = np.array(f7_right_ankle)
    f7_rigank_tpse = f7_rigank_arr.T
    f7_lefank_arr = np.array(f7_left_ankle)
    nor_f7_step = np.dot(f7_lefank_arr, f7_rigank_tpse)
    sw1 = lis_x[6] [10], lis_y[6] [10], lis_z[6] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[0] [13], lis_y[0] [13], lis_z[0] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor7 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f7 = nor_f7_step * cad_f1
    nor_f8_1 = lis_x[7] [10], lis_y[7] [10], lis_z[7] [10]
    nor_f8_l = list(nor_f8_1)
    f8_left_ankle = [x / 80 for x in nor_f8_l]
    nor_f8_2 = lis_x[7] [14], lis_y[7] [14], lis_z[7] [14]
    nor_f8_r = list(nor_f8_2)
    f8_right_ankle = [x / 80 for x in nor_f8_r]
    f8_rigank_arr = np.array(f8_right_ankle)
    f8_rigank_tpse = f1_rigank_arr.T
    f8_lefank_arr = np.array(f8_left_ankle)
    nor_f8_step = np.dot(f8_lefank_arr, f8_rigank_tpse)
    sw1 = lis_x[7] [10], lis_y[7] [10], lis_z[7] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[7] [13], lis_y[7] [13], lis_z[7] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor8 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f8 = nor_f8_step * cad_f1
    nor_f9_1 = lis_x[8] [10], lis_y[8] [10], lis_z[8] [10]
    nor_f9_l = list(nor_f9_1)
    f9_left_ankle = [x / 80 for x in nor_f9_l]
    nor_f9_2 = lis_x[8] [14], lis_y[8] [14], lis_z[8] [14]
    nor_f9_r = list(nor_f1_2)
    f9_right_ankle = [x / 80 for x in nor_f9_r]
    f9_rigank_arr = np.array(f9_right_ankle)
    f9_rigank_tpse = f9_rigank_arr.T
    f9_lefank_arr = np.array(f9_left_ankle)
    nor_f9_step = np.dot(f9_lefank_arr, f9_rigank_tpse)
    sw1 = lis_x[8] [10], lis_y[8] [10], lis_z[8] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[8] [13], lis_y[8] [13], lis_z[8] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor9 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f9 = nor_f9_step * cad_f1
    nor_f10_1 = lis_x[9] [10], lis_y[9] [10], lis_z[9] [10]
    nor_f10_l = list(nor_f10_1)
    f10_left_ankle = [x / 80 for x in nor_f10_l]
    nor_f10_2 = lis_x[9] [14], lis_y[9] [14], lis_z[9] [14]
    nor_f10_r = list(nor_f10_2)
    f10_right_ankle = [x / 80 for x in nor_f10_r]
    f10_rigank_arr = np.array(f10_right_ankle)
    f10_rigank_tpse = f1_rigank_arr.T
    f10_lefank_arr = np.array(f10_left_ankle)
    nor_f10_step = np.dot(f10_lefank_arr, f10_rigank_tpse)
    sw1 = lis_x[9] [10], lis_y[9] [10], lis_z[9] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[9] [13], lis_y[9] [13], lis_z[9] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor10 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f10 = nor_f10_step * cad_f1
    nor_f1_1 = lis_x[10] [10], lis_y[10] [10], lis_z[10] [10]
    nor_f1_l = list(nor_f1_1)
    f1_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f1_2 = lis_x[10] [14], lis_y[10] [14], lis_z[10] [14]
    nor_f1_r = list(nor_f1_2)
    f1_right_ankle = [x / 80 for x in nor_f1_r]
    f1_rigank_arr = np.array(f1_right_ankle)
    f1_rigank_tpse = f1_rigank_arr.T
    f1_lefank_arr = np.array(f1_left_ankle)
    nor_f11_step = np.dot(f1_lefank_arr, f1_rigank_tpse)
    sw1 = lis_x[10] [10], lis_y[10] [10], lis_z[10] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[10] [13], lis_y[10] [13], lis_z[10] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor11 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f11 = nor_f1_step * cad_f1
    nor_f2_1 = lis_x[11] [10], lis_y[11] [10], lis_z[11] [10]
    nor_f2_l = list(nor_f1_1)
    f2_left_ankle = [x / 80 for x in nor_f2_l]
    nor_f2_2 = lis_x[11] [14], lis_y[11] [14], lis_z[11] [14]
    nor_f2_r = list(nor_f1_2)
    f2_right_ankle = [x / 80 for x in nor_f2_r]
    f2_rigank_arr = np.array(f2_right_ankle)
    f2_rigank_tpse = f2_rigank_arr.T
    f2_lefank_arr = np.array(f2_left_ankle)
    nor_f11_step = np.dot(f2_lefank_arr, f2_rigank_tpse)
    sw1 = lis_x[11] [10], lis_y[11] [10], lis_z[11] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[11] [13], lis_y[11] [13], lis_z[11] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor12 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f12 = nor_f2_step * cad_f1
    nor_f3_1 = lis_x[12] [10], lis_y[12] [10], lis_z[12] [10]
    nor_f3_l = list(nor_f1_1)
    f3_left_ankle = [x / 80 for x in nor_f3_l]
    nor_f3_2 = lis_x[12] [14], lis_y[12] [14], lis_z[12] [14]
    nor_f3_r = list(nor_f3_2)
    f3_right_ankle = [x / 80 for x in nor_f3_r]
    f3_rigank_arr = np.array(f3_right_ankle)
    f3_rigank_tpse = f3_rigank_arr.T
    f3_lefank_arr = np.array(f3_left_ankle)
    nor_f12_step = np.dot(f3_lefank_arr, f3_rigank_tpse)
    sw1 = lis_x[12] [10], lis_y[12] [10], lis_z[12] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[12] [13], lis_y[12] [13], lis_z[12] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor13 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f13 = nor_f3_step * cad_f1
    nor_f4_1 = lis_x[13] [10], lis_y[13] [10], lis_z[13] [10]
    nor_f4_l = list(nor_f4_1)
    f4_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f4_2 = lis_x[13] [14], lis_y[13] [14], lis_z[13] [14]
    nor_f4_r = list(nor_f4_2)
    f4_right_ankle = [x / 80 for x in nor_f1_r]
    f4_right_ankle
    f4_rigank_arr = np.array(f4_right_ankle)
    f4_rigank_tpse = f4_rigank_arr.T
    f4_lefank_arr = np.array(f4_left_ankle)
    nor_f14_step = np.dot(f4_lefank_arr, f4_rigank_tpse)
    sw1 = lis_x[13] [10], lis_y[13] [10], lis_z[13] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[13] [13], lis_y[13] [13], lis_z[13] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor14 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f14 = nor_f4_step * cad_f1
    nor_f5_1 = lis_x[14] [10], lis_y[14] [10], lis_z[14] [10]
    nor_f5_l = list(nor_f5_1)
    f5_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f5_2 = lis_x[14] [14], lis_y[14] [14], lis_z[14] [14]
    nor_f5_r = list(nor_f5_2)
    f5_right_ankle = [x / 80 for x in nor_f5_r]
    f5_rigank_arr = np.array(f5_right_ankle)
    f5_rigank_tpse = f5_rigank_arr.T
    f5_lefank_arr = np.array(f5_left_ankle)
    nor_f15_step = np.dot(f5_lefank_arr, f5_rigank_tpse)
    sw1 = lis_x[14] [10], lis_y[14] [10], lis_z[14] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[14] [13], lis_y[14] [13], lis_z[14] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor15 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f15 = nor_f5_step * cad_f1
    nor_f6_1 = lis_x[15] [10], lis_y[15] [10], lis_z[15] [10]
    nor_f6_l = list(nor_f6_1)
    f6_left_ankle = [x / 80 for x in nor_f6_l]
    nor_f6_2 = lis_x[15] [14], lis_y[15] [14], lis_z[15] [14]
    nor_f6_r = list(nor_f6_2)
    f6_right_ankle = [x / 80 for x in nor_f1_r]
    f6_rigank_arr = np.array(f6_right_ankle)
    f6_rigank_tpse = f6_rigank_arr.T
    f6_lefank_arr = np.array(f6_left_ankle)
    nor_f16_step = np.dot(f6_lefank_arr, f6_rigank_tpse)
    sw1 = lis_x[15] [10], lis_y[15] [10], lis_z[15] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[15] [13], lis_y[15] [13], lis_z[15] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor16 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f16 = nor_f6_step * cad_f1
    nor_f7_1 = lis_x[16] [10], lis_y[16] [10], lis_z[16] [10]
    nor_f7_l = list(nor_f7_1)
    f7_left_ankle = [x / 80 for x in nor_f7_l]
    nor_f7_2 = lis_x[16] [14], lis_y[16] [14], lis_z[16] [14]
    nor_f7_r = list(nor_f7_2)
    f7_right_ankle = [x / 80 for x in nor_f7_r]
    f7_rigank_arr = np.array(f7_right_ankle)
    f7_rigank_tpse = f7_rigank_arr.T
    f7_lefank_arr = np.array(f7_left_ankle)
    nor_f17_step = np.dot(f7_lefank_arr, f7_rigank_tpse)
    sw1 = lis_x[16] [10], lis_y[16] [10], lis_z[16] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[16] [13], lis_y[16] [13], lis_z[16] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor17 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f17 = nor_f7_step * cad_f1
    nor_f8_1 = lis_x[17] [10], lis_y[17] [10], lis_z[17] [10]
    nor_f8_l = list(nor_f8_1)
    f8_left_ankle = [x / 80 for x in nor_f8_l]
    nor_f8_2 = lis_x[17] [14], lis_y[17] [14], lis_z[17] [14]
    nor_f8_r = list(nor_f8_2)
    f8_right_ankle = [x / 80 for x in nor_f8_r]
    f8_right_ankle
    f8_rigank_arr = np.array(f8_right_ankle)
    f8_rigank_tpse = f1_rigank_arr.T
    f8_lefank_arr = np.array(f8_left_ankle)
    nor_f18_step = np.dot(f8_lefank_arr, f8_rigank_tpse)
    sw1 = lis_x[17] [10], lis_y[17] [10], lis_z[17] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[17] [13], lis_y[17] [13], lis_z[17] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor18 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f18 = nor_f8_step * cad_f1
    nor_f9_1 = lis_x[18] [10], lis_y[18] [10], lis_z[18] [10]
    nor_f9_l = list(nor_f9_1)
    f9_left_ankle = [x / 80 for x in nor_f9_l]
    nor_f9_2 = lis_x[18] [14], lis_y[18] [14], lis_z[18] [14]
    nor_f9_r = list(nor_f1_2)
    f9_right_ankle = [x / 80 for x in nor_f9_r]
    f9_right_ankle
    f9_rigank_arr = np.array(f9_right_ankle)
    f9_rigank_tpse = f9_rigank_arr.T
    f9_lefank_arr = np.array(f9_left_ankle)
    nor_f19_step = np.dot(f9_lefank_arr, f9_rigank_tpse)
    sw1 = lis_x[18] [10], lis_y[18] [10], lis_z[18] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[18] [13], lis_y[18] [13], lis_z[18] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor19 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f19 = nor_f9_step * cad_f1
    nor_f10_1 = lis_x[19] [10], lis_y[19] [10], lis_z[19] [10]
    nor_f10_l = list(nor_f10_1)
    f10_left_ankle = [x / 80 for x in nor_f10_l]
    nor_f10_2 = lis_x[19] [14], lis_y[19] [14], lis_z[19] [14]
    nor_f10_r = list(nor_f10_2)
    f10_right_ankle = [x / 80 for x in nor_f10_r]
    f10_right_ankle
    f10_rigank_arr = np.array(f10_right_ankle)
    f10_rigank_tpse = f1_rigank_arr.T
    f10_lefank_arr = np.array(f10_left_ankle)
    nor_f20_step = np.dot(f10_lefank_arr, f10_rigank_tpse)
    sw1 = lis_x[19] [10], lis_y[19] [10], lis_z[19] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[19] [13], lis_y[19] [13], lis_z[19] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor20 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f20 = nor_f10_step * cad_f1
    nor_f1_1 = lis_x[20] [10], lis_y[20] [10], lis_z[20] [10]
    nor_f1_l = list(nor_f1_1)
    f1_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f1_2 = lis_x[20] [14], lis_y[20] [14], lis_z[20] [14]
    nor_f1_r = list(nor_f1_2)
    f1_right_ankle = [x / 80 for x in nor_f1_r]
    f1_right_ankle
    f1_rigank_arr = np.array(f1_right_ankle)
    f1_rigank_tpse = f1_rigank_arr.T
    f1_lefank_arr = np.array(f1_left_ankle)
    nor_f21_step = np.dot(f1_lefank_arr, f1_rigank_tpse)
    sw1 = lis_x[20] [10], lis_y[20] [10], lis_z[20] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[20] [13], lis_y[20] [13], lis_z[20] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor21 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f21 = nor_f1_step * cad_f1
    nor_f2_1 = lis_x[21] [10], lis_y[21] [10], lis_z[21] [10]
    nor_f2_l = list(nor_f1_1)
    f2_left_ankle = [x / 80 for x in nor_f2_l]
    nor_f2_2 = lis_x[21] [14], lis_y[21] [14], lis_z[21] [14]
    nor_f2_r = list(nor_f1_2)
    f2_right_ankle = [x / 80 for x in nor_f2_r]
    f2_rigank_arr = np.array(f2_right_ankle)
    f2_rigank_tpse = f2_rigank_arr.T
    f2_lefank_arr = np.array(f2_left_ankle)
    nor_f21_step = np.dot(f2_lefank_arr, f2_rigank_tpse)
    sw1 = lis_x[21] [10], lis_y[21] [10], lis_z[21] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[21] [13], lis_y[21] [13], lis_z[21] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor22 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f22 = nor_f2_step * cad_f1
    nor_f3_1 = lis_x[22] [10], lis_y[22] [10], lis_z[22] [10]
    nor_f3_l = list(nor_f1_1)
    f3_left_ankle = [x / 80 for x in nor_f3_l]
    nor_f3_2 = lis_x[22] [14], lis_y[22] [14], lis_z[22] [14]
    nor_f3_r = list(nor_f3_2)
    f3_right_ankle = [x / 80 for x in nor_f3_r]
    f3_right_ankle
    f3_rigank_arr = np.array(f3_right_ankle)
    f3_rigank_tpse = f3_rigank_arr.T
    f3_lefank_arr = np.array(f3_left_ankle)
    nor_f23_step = np.dot(f3_lefank_arr, f3_rigank_tpse)
    sw1 = lis_x[22] [10], lis_y[22] [10], lis_z[22] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[22] [13], lis_y[22] [13], lis_z[22] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor23 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f23 = nor_f3_step * cad_f1
    nor_f4_1 = lis_x[23] [10], lis_y[23] [10], lis_z[23] [10]
    nor_f4_l = list(nor_f4_1)
    f4_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f4_2 = lis_x[23] [14], lis_y[23] [14], lis_z[23] [14]
    nor_f4_r = list(nor_f4_2)
    f4_right_ankle = [x / 80 for x in nor_f1_r]
    f4_rigank_arr = np.array(f4_right_ankle)
    f4_rigank_tpse = f4_rigank_arr.T
    f4_lefank_arr = np.array(f4_left_ankle)
    nor_f24_step = np.dot(f4_lefank_arr, f4_rigank_tpse)
    sw1 = lis_x[23] [10], lis_y[23] [10], lis_z[23] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[23] [13], lis_y[23] [13], lis_z[23] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor24 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f24 = nor_f4_step * cad_f1
    nor_f5_1 = lis_x[24] [10], lis_y[24] [10], lis_z[24] [10]
    nor_f5_l = list(nor_f5_1)
    f5_left_ankle = [x / 80 for x in nor_f1_l]
    nor_f5_2 = lis_x[24] [14], lis_y[24] [14], lis_z[24] [14]
    nor_f5_r = list(nor_f5_2)
    f5_right_ankle = [x / 80 for x in nor_f5_r]
    f5_rigank_arr = np.array(f5_right_ankle)
    f5_rigank_tpse = f5_rigank_arr.T
    f5_lefank_arr = np.array(f5_left_ankle)
    nor_f25_step = np.dot(f5_lefank_arr, f5_rigank_tpse)
    sw1 = lis_x[24] [10], lis_y[24] [10], lis_z[24] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[24] [13], lis_y[24] [13], lis_z[24] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor25 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f25 = nor_f5_step * cad_f1
    nor_f6_1 = lis_x[25] [10], lis_y[25] [10], lis_z[25] [10]
    nor_f6_l = list(nor_f6_1)
    f6_left_ankle = [x / 80 for x in nor_f6_l]
    nor_f6_2 = lis_x[25] [14], lis_y[25] [14], lis_z[25] [14]
    nor_f6_r = list(nor_f6_2)
    f6_right_ankle = [x / 80 for x in nor_f1_r]
    f6_rigank_arr = np.array(f6_right_ankle)
    f6_rigank_tpse = f6_rigank_arr.T
    f6_lefank_arr = np.array(f6_left_ankle)
    nor_f26_step = np.dot(f6_lefank_arr, f6_rigank_tpse)
    sw1 = lis_x[25] [10], lis_y[25] [10], lis_z[25] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[25] [13], lis_y[25] [13], lis_z[25] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor26 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f26 = nor_f6_step * cad_f1
    nor_f7_1 = lis_x[26] [10], lis_y[26] [10], lis_z[26] [10]
    nor_f7_l = list(nor_f7_1)
    f7_left_ankle = [x / 80 for x in nor_f7_l]
    nor_f7_2 = lis_x[26] [14], lis_y[26] [14], lis_z[26] [14]
    nor_f7_r = list(nor_f7_2)
    f7_right_ankle = [x / 80 for x in nor_f7_r]
    f7_rigank_arr = np.array(f7_right_ankle)
    f7_rigank_tpse = f7_rigank_arr.T
    f7_lefank_arr = np.array(f7_left_ankle)
    nor_f27_step = np.dot(f7_lefank_arr, f7_rigank_tpse)
    sw1 = lis_x[26] [10], lis_y[26] [10], lis_z[26] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[26] [13], lis_y[26] [13], lis_z[26] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor7 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f27 = nor_f7_step * cad_f1
    nor_f8_1 = lis_x[27] [10], lis_y[27] [10], lis_z[27] [10]
    nor_f8_l = list(nor_f8_1)
    f8_left_ankle = [x / 80 for x in nor_f8_l]
    nor_f8_2 = lis_x[27] [14], lis_y[27] [14], lis_z[27] [14]
    nor_f8_r = list(nor_f8_2)
    f8_right_ankle = [x / 80 for x in nor_f8_r]
    f8_rigank_arr = np.array(f8_right_ankle)
    f8_rigank_tpse = f1_rigank_arr.T
    f8_lefank_arr = np.array(f8_left_ankle)
    nor_f28_step = np.dot(f8_lefank_arr, f8_rigank_tpse)
    sw1 = lis_x[27] [10], lis_y[27] [10], lis_z[27] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[27] [13], lis_y[27] [13], lis_z[27] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor28 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f28 = nor_f8_step * cad_f1
    nor_f9_1 = lis_x[28] [10], lis_y[28] [10], lis_z[28] [10]
    nor_f9_l = list(nor_f9_1)
    f9_left_ankle = [x / 80 for x in nor_f9_l]
    nor_f9_2 = lis_x[28] [14], lis_y[28] [14], lis_z[28] [14]
    nor_f9_r = list(nor_f1_2)
    f9_right_ankle = [x / 80 for x in nor_f9_r]
    f9_rigank_arr = np.array(f9_right_ankle)
    f9_rigank_tpse = f9_rigank_arr.T
    f9_lefank_arr = np.array(f9_left_ankle)
    nor_f29_step = np.dot(f9_lefank_arr, f9_rigank_tpse)
    sw1 = lis_x[28] [10], lis_y[28] [10], lis_z[28] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[28] [13], lis_y[28] [13], lis_z[28] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor29 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f29 = nor_f9_step * cad_f1
    nor_f10_1 = lis_x[29] [10], lis_y[29] [10], lis_z[29] [10]
    nor_f10_l = list(nor_f10_1)
    f10_left_ankle = [x / 80 for x in nor_f10_l]
    nor_f10_2 = lis_x[29] [14], lis_y[29] [14], lis_z[29] [14]
    nor_f10_r = list(nor_f10_2)
    f10_right_ankle = [x / 80 for x in nor_f10_r]
    f10_rigank_arr = np.array(f10_right_ankle)
    f10_rigank_tpse = f1_rigank_arr.T
    f10_lefank_arr = np.array(f10_left_ankle)
    nor_f30_step = np.dot(f10_lefank_arr, f10_rigank_tpse)
    sw1 = lis_x[29] [10], lis_y[29] [10], lis_z[29] [10]
    sw1_1 = list(sw1)
    nor_sw1 = [x / 85 for x in sw1_1]
    sw2 = lis_x[29] [13], lis_y[29] [13], lis_z[29] [13]
    sw1_2 = list(sw2)
    nor_sw2 = [x / 85 for x in sw1_2]
    dst_nor30 = distance.euclidean(nor_sw1, nor_sw2)
    cad_stp = 55 #No of steps
    cad_f1 = cad_stp/1800
    speed_f30 = nor_f10_step * cad_f1
    
    ##  Step length
    mean1_stlen = st.mean([nor_f1_step, nor_f2_step, nor_f3_step])
    st1_stlen = st.stdev([nor_f1_step, nor_f2_step, nor_f3_step])
    min1_stlen = min([nor_f1_step, nor_f2_step, nor_f3_step])
    max1_stlen = max([nor_f1_step, nor_f2_step, nor_f3_step])
    mean2_stlen = st.mean([nor_f4_step, nor_f5_step, nor_f6_step])	
    st2_stlen = st.stdev([nor_f4_step, nor_f5_step, nor_f6_step])
    min2_stlen = min([nor_f4_step, nor_f5_step, nor_f6_step])
    max2_stlen = max([nor_f4_step, nor_f5_step, nor_f6_step])
    mean3_stlen = st.mean([nor_f7_step, nor_f8_step, nor_f9_step])	
    st3_stlen = st.stdev([nor_f7_step, nor_f8_step, nor_f9_step])
    min3_stlen = min([nor_f7_step, nor_f8_step, nor_f9_step])
    max3_stlen = max([nor_f7_step, nor_f8_step, nor_f9_step])
    mean4_stlen = st.mean([nor_f10_step, nor_f11_step, nor_f12_step])
    st4_stlen = st.stdev([nor_f10_step, nor_f11_step, nor_f12_step])
    min4_stlen = min([nor_f10_step, nor_f11_step, nor_f12_step])
    max4_stlen = max([nor_f10_step, nor_f11_step, nor_f12_step])
    mean5_stlen = st.mean([nor_f3_step, nor_f14_step, nor_f15_step])	
    st5_stlen = st.stdev([nor_f3_step, nor_f14_step, nor_f15_step])
    min5_stlen = min([nor_f3_step, nor_f14_step, nor_f15_step])
    max5_stlen = max([nor_f3_step, nor_f14_step, nor_f15_step])    
    mean6_stlen = st.mean([nor_f16_step, nor_f17_step, nor_f18_step])	
    st6_stlen = st.stdev([nor_f16_step, nor_f17_step, nor_f18_step])
    min6_stlen = min([nor_f16_step, nor_f17_step, nor_f18_step])
    max6_stlen = max([nor_f16_step, nor_f17_step, nor_f18_step])	
    mean7_stlen = st.mean([nor_f19_step, nor_f20_step, nor_f21_step])	
    st7_stlen = st.stdev([nor_f19_step, nor_f20_step, nor_f21_step])
    min7_stlen = min([nor_f19_step, nor_f20_step, nor_f21_step])
    max7_stlen = max([nor_f19_step, nor_f20_step, nor_f21_step])	
    mean8_stlen = st.mean([nor_f2_step, nor_f23_step, nor_f24_step])	
    st8_stlen = st.stdev([nor_f2_step, nor_f23_step, nor_f24_step])
    min8_stlen = min([nor_f2_step, nor_f23_step, nor_f24_step])
    max8_stlen = max([nor_f2_step, nor_f23_step, nor_f24_step])	
    mean9_stlen = st.mean([nor_f25_step, nor_f26_step, nor_f27_step])	
    st9_stlen = st.stdev([nor_f25_step, nor_f26_step, nor_f27_step])
    min9_stlen = min([nor_f25_step, nor_f26_step, nor_f27_step])
    max9_stlen = max([nor_f25_step, nor_f26_step, nor_f27_step])	
    mean10_stlen = st.mean([nor_f28_step, nor_f29_step, nor_f30_step])	
    st10_stlen = st.stdev([nor_f28_step, nor_f29_step, nor_f30_step])
    min10_stlen = min([nor_f28_step, nor_f29_step, nor_f30_step])
    max10_stlen = max([nor_f28_step, nor_f29_step, nor_f30_step])	
    
    ##  Step Width
    mean1_stwd = st.mean([dst_nor1, dst_nor2, dst_nor3])
    st1_stwd = st.stdev([dst_nor1, dst_nor2, dst_nor3])
    min1_stwd = min([dst_nor1, dst_nor2, dst_nor3])
    max1_stwd = max([dst_nor1, dst_nor2, dst_nor3])	
    mean2_stwd = st.mean([dst_nor4, dst_nor5, dst_nor6])    	
    st2_stwd = st.stdev([dst_nor4, dst_nor5, dst_nor6])
    min2_stwd = min([dst_nor4, dst_nor5, dst_nor6])
    max2_stwd = max([dst_nor4, dst_nor5, dst_nor6])	
    mean3_stwd = st.mean([dst_nor7, dst_nor8, dst_nor9])
    st3_stwd = st.stdev([dst_nor7, dst_nor8, dst_nor9])
    min3_stwd = min([dst_nor7, dst_nor8, dst_nor9])
    max3_stwd = max([dst_nor7, dst_nor8, dst_nor9])	
    mean4_stwd = st.mean([dst_nor10, dst_nor11, dst_nor12])
    st4_stwd = st.stdev([dst_nor10, dst_nor11, dst_nor12])
    min4_stwd = min([dst_nor10, dst_nor11, dst_nor12])
    max4_stwd = max([dst_nor10, dst_nor11, dst_nor12])	
    mean5_stwd = st.mean([dst_nor13, dst_nor14, dst_nor15])
    st5_stwd = st.stdev([dst_nor13, dst_nor14, dst_nor15])
    min5_stwd = min([dst_nor13, dst_nor14, dst_nor15])
    max5_stwd = max([dst_nor13, dst_nor14, dst_nor15])	
    mean6_stwd = st.mean([dst_nor16, dst_nor17, dst_nor18])   
    st6_stwd = st.stdev([dst_nor16, dst_nor17, dst_nor18])
    min6_stwd = min([dst_nor16, dst_nor17, dst_nor18])
    max6_stwd = max([dst_nor16, dst_nor17, dst_nor18])	
    mean7_stwd = st.mean([dst_nor19, dst_nor20, dst_nor21])   
    st7_stwd = st.stdev([dst_nor19, dst_nor20, dst_nor21])
    min7_stwd = min([dst_nor19, dst_nor20, dst_nor21])
    max7_stwd = max([dst_nor19, dst_nor20, dst_nor21])	
    mean8_stwd = st.mean([dst_nor22, dst_nor23, dst_nor24])   
    st8_stwd = st.stdev([dst_nor22, dst_nor23, dst_nor24])
    min8_stwd = min([dst_nor22, dst_nor23, dst_nor24])
    max8_stwd = max([dst_nor22, dst_nor23, dst_nor24])	
    mean9_stwd = st.mean([dst_nor25, dst_nor26, dst_nor7])
    st9_stwd = st.stdev([dst_nor25, dst_nor26, dst_nor7])
    min9_stwd = min([dst_nor25, dst_nor26, dst_nor7])
    max9_stwd = max([dst_nor25, dst_nor26, dst_nor7])	
    mean10_stwd = st.mean([dst_nor28, dst_nor29, dst_nor30])
    st10_stwd = st.stdev([dst_nor28, dst_nor29, dst_nor30])
    min10_stwd = min([dst_nor28, dst_nor29, dst_nor30])
    max10_stwd = max([dst_nor28, dst_nor29, dst_nor30])
    
    ## Gait speed
    mean1_spd = st.mean([speed_f1, speed_f2, speed_f3])
    st1_spd = st.stdev([speed_f1, speed_f2, speed_f3])
    min1_spd = min([speed_f1, speed_f2, speed_f3])
    max1_spd = max([speed_f1, speed_f2, speed_f3])
    mean2_spd = st.mean([speed_f4, speed_f5, speed_f6])
    st2_spd = st.stdev([speed_f4, speed_f5, speed_f6])
    min2_spd = min([speed_f4, speed_f5, speed_f6])
    max2_spd = max([speed_f4, speed_f5, speed_f6])
    mean3_spd = st.mean([speed_f7, speed_f8, speed_f9])	
    st3_spd = st.stdev([speed_f7, speed_f8, speed_f9])
    min3_spd = min([speed_f7, speed_f8, speed_f9])
    max3_spd = max([speed_f7, speed_f8, speed_f9])
    mean4_spd = st.mean([speed_f10, speed_f11, speed_f12])
    st4_spd = st.stdev([speed_f10, speed_f11, speed_f12])
    min4_spd = min([speed_f10, speed_f11, speed_f12])
    max4_spd = max([speed_f10, speed_f11, speed_f12])
    mean5_spd = st.mean([speed_f13, speed_f14, speed_f15])	
    st5_spd = st.stdev([speed_f13, speed_f14, speed_f15])
    min5_spd = min([speed_f13, speed_f14, speed_f15])
    max5_spd = max([speed_f13, speed_f14, speed_f15])
    mean6_spd = st.mean([speed_f16, speed_f17, speed_f18])	
    st6_spd = st.stdev([speed_f16, speed_f17, speed_f18])
    min6_spd = min([speed_f16, speed_f17, speed_f18])
    max6_spd = max([speed_f16, speed_f17, speed_f18])
    mean7_spd = st.mean([speed_f19, speed_f20, speed_f21])    
    st7_spd = st.stdev([speed_f19, speed_f20, speed_f21])
    min7_spd = min([speed_f19, speed_f20, speed_f21])
    max7_spd = max([speed_f19, speed_f20, speed_f21])
    mean8_spd = st.mean([speed_f22, speed_f23, speed_f24])
    st8_spd = st.stdev([speed_f22, speed_f23, speed_f24])
    min8_spd = min([speed_f22, speed_f23, speed_f24])
    max8_spd = max([speed_f22, speed_f23, speed_f24])
    mean9_spd = st.mean([speed_f25, speed_f26, speed_f27])
    st9_spd = st.stdev([speed_f25, speed_f26, speed_f27])
    min9_spd = min([speed_f25, speed_f26, speed_f27])
    max9_spd = max([speed_f25, speed_f26, speed_f27])
    mean10_spd = st.mean([speed_f28, speed_f29, speed_f30])	
    st10_spd = st.stdev([speed_f28, speed_f29, speed_f30])
    min10_spd = min([speed_f28, speed_f29, speed_f30])
    max10_spd = max([speed_f28, speed_f29, speed_f30])
    
    
    dat = [[mean1_stlen, mean1_stwd, mean1_spd], 
     [mean2_stlen, mean2_stwd, mean2_spd],
     [mean3_stlen, mean3_stwd, mean3_spd], 
     [mean4_stlen, mean4_stwd, mean4_spd], 
     [mean5_stlen, mean5_stwd, mean5_spd], 
     [mean6_stlen, mean6_stwd, mean6_spd], 
     [mean7_stlen, mean7_stwd, mean7_spd], 
     [mean8_stlen, mean8_stwd, mean8_spd], 
     [mean9_stlen, mean9_stwd, mean9_spd], 
     [mean10_stlen, mean10_stwd, mean10_spd]]
    
    #real_dat = pd.DataFrame(dat, columns = ['Steplength', 'Stepwidth', 'Gaitspeed']) 
    
    with open('D:\\Sem 4\\Proj\\real_v3\\Oct22\\example.csv', 'w') as f:
        writer = csv.writer(f)
        for line in dat:
            print(line)
            writer.writerow(line)
            
    df = pd.read_csv('D:\\Sem 4\\Proj\\real_v3\\Oct22\\example.csv')
    df.columns = ['Steplength',	'Stepwidth', 'Gaitspeed']
    df.to_csv('D:\\Sem 4\\Proj\\real_v3\\Oct22\\realtest.csv', index=False)
    Actual = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    Predicted = [1, 3, 1, 3, 1, 1, 1, 1, 1]
    df['ActualValues'] = Actual
    df['PredictedValues'] = Predicted
    print(df)
    
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table(ax, df)  # where df is your data frame

    plt.savefig('D:\\Sem 4\\Proj\\real_v3\\Oct22\\tab_res.png')
    
            
            
lbl1 = Label(window, foreground='green', text='Run the Application')
lbl1.config(font=('Times New Roman', 10))
lbl1.place(x=85, y=400)

btn1 = Button(window, text='tf-pose', fg='blue', command = lambda: cmd_prmt())
btn1.place(x=235, y=397)

lbl2 = Label(window, foreground='green', text='Extract the gait features')
lbl2.config(font=('Times New Roman', 10))
lbl2.place(x=85, y=440)

btn2 = Button(window, text='Load', fg='blue', command = lambda: feat_ext())
btn2.place(x=235, y=440)

lbl3 = Label(window, foreground='green', text='Select the experiment')
lbl3.config(font=('Times New Roman', 10))
lbl3.place(x=85, y=480)

cmb1 = Combobox(window, width="18", values=('Normal', 'CB3', 'CB7'), foreground='blue')
cmb1.place(x=235, y=480)

btn2 = Button(window, text='SUBMIT', foreground='blue', command = checkcombo)
btn2.place(x=185, y=520)

btn3 = Button(window, text='ONLOAD', fg='blue', command = lambda: anim_image())
btn3.place(x=200, y=200)


#img = ImageTk.PhotoImage(Image.open("D:\\Gait Analysis\\Final_Nov19\\gait_1.gif"))
#img = Image.open("D:\\Gait Analysis\\Final_Nov19\\gait_1.gif")
#img = img.resize((100, 100), Image.ANTIALIAS)
#img = ImageTk.PhotoImage(img)
#panel = Label(window, image=img)
#panel.pack()
#panel.place(x=100, y=20)

#anim_image()

window.title('GAIT ANALYSIS USING 3D POSE ESTIMATION')
window.configure(bg='light blue')
window.geometry("500x600")
window.mainloop()

