from tqdm import tqdm
import numpy as np
import os
import cv2
 
object_path = "/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/object_det/"
imgs = np.sort(os.listdir( object_path )) 
# for img_name in tqdm(np.sort(imgs)):
#     print(img_name)

def LBP(src):
    '''
    :param src:灰度图像
    :return:
    '''
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    lbp_value = np.zeros((1,8), dtype=np.uint8)
    #print(lbp_value)
    neighbours = np.zeros((1,8), dtype=np.uint8)
    #print(neighbours)
    for x in range(1, width-1):
        for y in range(1, height-1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]
            center = src[y, x]
            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 7] * 128
            
            #print(lbp)
            dst[y, x] = lbp

    return dst

for img_name in tqdm(imgs):
    if img_name == "0155_002.jpg"  or img_name == "0155_001.jpg" :
        red_car = cv2.imread(object_path + img_name)
        gray = cv2.cvtColor(red_car , cv2.COLOR_BGR2GRAY)
        
        # Harris角点检测
        gray32 = np.float32(gray)
        dst = cv2.cornerHarris(gray32,2,3,0.06)
        dst = cv2.dilate(dst,None)
        red_car_harris=red_car.copy()
        red_car_harris[dst>0.01*dst.max()]=[0,0,255]
        cv2.imwrite(  "./object_process_output/red_car_harris.png",red_car_harris)

        # # SIFT算法
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp = sift.detect(gray,None)#找到关键点
        # red_car_SIFT = red_car.copy()
        # red_car_SIFT = cv2.drawKeypoints(red_car_SIFT,kp,red_car_SIFT)#绘制关键点
        # cv2.imwrite(  "./object_process_output/red_car_SIFT.png",red_car_SIFT )

        # LBP
        red_car_LBP = LBP(gray)
        cv2.imwrite(  "./object_process_output/red_car_LBP.png",red_car_LBP)

        # 颜色矩阵
        hsv = cv2.cvtColor(red_car,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        color_feature = []
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)
        color_feature.extend([h_mean,s_mean,v_mean])
        h_std = np.std(h)
        s_std = np.std(s)
        v_std = np.std(v)
        color_feature.extend([h_std,s_std,v_std])
        h_skewness = np.mean(abs(h-h.mean())**3)
        s_skewness = np.mean(abs(s-s.mean())**3)
        v_skewness = np.mean(abs(v-v.mean())**3)
        h_thirdMoment = h_skewness**(1./3)
        s_thirdMoment = s_skewness**(1./3)
        v_thirdMoment = v_skewness**(1./3)
        color_feature.extend([h_thirdMoment,s_thirdMoment,v_thirdMoment])
        print(color_feature)

     

        



   

    frame_num , object_name = img_name.split("_")
    # print(frame_num)
    # print(object_name)
    