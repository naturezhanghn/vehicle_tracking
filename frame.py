import cv2
import os
import numpy as np
import tqdm
def save_img():
    # video_path = r'/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/UA-DETRAC-TRACKING-TEST-VIDEOS/'
    video_path = r'/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/'
    save_path = r"/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/data/"
    videos = os.listdir(video_path)
    for video_name in np.sort(videos):
        if video_name != 'parkcarerror.mp4':
            print(video_name)
            continue
        file_name = video_name.split('.')[0]
        
        folder_name = save_path + file_name
        os.makedirs(folder_name,exist_ok=True)
        vc = cv2.VideoCapture(video_path+video_name) #读入视频文件
        c=0
        rval=vc.isOpened()

        while (rval):   #循环读取视频帧
            rval, frame = vc.read()
            c = c + 1
            if c > 0:
                # rval, frame = vc.read()
                pic_path = folder_name+'/'
                if rval:
                    cv2.imwrite(pic_path  + str(c).zfill(5) + '.png', frame) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                    cv2.waitKey(1)
                else:
                    break
                print(c)
            if c == 5000:
                rval = 0
            

        vc.release()
        print('save_success')
        print(folder_name)
save_img()