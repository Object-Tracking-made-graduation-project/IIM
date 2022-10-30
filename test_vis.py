import os
import sys
import numpy as np
from scipy import spatial as ss

import cv2
from misc.utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter

from one_pic_inference import *



def main():

    #pred_data, gt_data = read_pred_and_gt(pred_file,gt_file)

    net = create_model(model_path)

    img = Image.open("./frames/images/60.jpg")

    pred_data = predict_people_in_frame(img, net)

     
    gt_p,pred_p,fn_gt_index,tp_pred_index,fp_pred_index,ap,ar= [],[],[],[],[],[],[]

    if pred_data['num'] !=0:            
        pred_p =  pred_data['points']
        fp_pred_index = np.array(range(pred_p.shape[0]))
        ap = 0
        ar = 0

    if pred_data['num'] ==0:
        ap = 0
        ar = 0 
        

    img = cv2.imread("./frames/images/30.jpg")#bgr
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    
    point_r_value = 5
    thickness = 3

    if pred_data['num'] !=0:
        for i in range(pred_p.shape[0]):
            if i in tp_pred_index:
                cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value,(0,255,0),-1)# tp: green
            else:                
                cv2.circle(img,(int(pred_p[i][0]),int(pred_p[i][1])),point_r_value*2,(255,0,255),-1) # fp: Magenta
                #cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value,(0,255,0),-1)# tp: green

    #cv2.imwrite(exp_name+'/'+str(i_sample)+ '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] + '.jpg', img)
    cv2.imwrite("marked_pic"+ '.jpg', img)


if __name__ == '__main__':
    main()
