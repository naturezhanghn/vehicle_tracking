import cv2
import libs.pyalgorithm as pyalgorithm
from tqdm import tqdm
import numpy as np
import os
import argparse
from sort.sort2 import Sort

parser = argparse.ArgumentParser()

parser.add_argument("--img_path",
                    # default = '/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/data/parkcarerror/',
                    default = '/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/data/MVI_40172/',
                    # default = r"/workspace/mnt/storage/zhangzirna/zhangzr4/data/test_video/",
                    help = "data path")
parser.add_argument('--video_path',
                    default='/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/UA-DETRAC-TRACKING-TEST-VIDEOS/MVI_40172.mp4',
                    # default='/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/cover/a4.mp4',
                    help='video path')
parser.add_argument('--tronmodel',
                    default=
                    'models/PeleeNet_ATSSv2_4class_det_20210311_e29cd266_OPENCV_960cut8-v2.13.4.tronmodel',
                    help='frame_save_path')
parser.add_argument('--frame_save_path',
                    default=
                    '/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/a1_output/',
                    help='save path')
args = parser.parse_args()

MIN_WIDTH = 10
colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [0, 69, 255],
          [203, 192, 255], [255, 255, 255], [139, 139, 0], [30, 105, 210],
          [169, 169, 169], [0, 0, 139], [10, 215, 255], [0, 128, 128],
          [144, 238, 144], [230, 216, 173], [130, 0, 75], [128, 0, 128],
          [147, 20, 255], [238, 130, 238]]

def detector_init(tronmodel,
                  btype='Native',
                  fp16=False,
                  device=0,
                  batchsize=1):
    """
    detector init
    Args:
        tronmodel: tronmodel path
        btype: backend type 'Native' or 'TensorRT'
        fp16: only set 'True' for btype using 'TensorRT'
        device: infer gpu id
        batchsize: number of images
    Returns: detector
    """
    detector = module_handle.load_detect(enable_profiler=True,
                                         backend_type=btype,
                                         use_fp16=fp16,
                                         cache_engine=True,
                                         device_id=device,
                                         max_batch_size=batchsize)
    return detector

caps = cv2.VideoCapture(args.video_path)
fps = int(caps.get(cv2.CAP_PROP_FPS))
width = int(caps.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(caps.get(cv2.CAP_PROP_FRAME_HEIGHT))
all_frames = int(caps.get(cv2.CAP_PROP_FRAME_COUNT))

module_handle = pyalgorithm.ModuleHandle(args.tronmodel)
traffic_detector = detector_init(args.tronmodel,
                                     btype='Native',
                                     fp16=True)
mot_tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3)

target_buffer = []
imgs = os.listdir(args.img_path)
cnt = 0
for img_name in tqdm(np.sort(imgs)):
    cnt += 1
    if cnt % 2 != 1:
        continue
    # if cnt <1501 or cnt >  1560 :
    #     continue
    # if cnt <1719 or cnt >  1929 :
    #     continue

    img_path = os.path.join(args.img_path ,img_name)
    frame = cv2.imread(img_path)
    det_result = traffic_detector.run(data=[frame])
  
    target = np.array([[res.xmin, res.ymin, res.xmax, res.ymax, res.score] \
        for res in det_result[0] if res.xmax-res.xmin>=MIN_WIDTH and res.score >= 0.2858 and res.label==2]) # 0.2858
    
    # prinst("target:",target)
    result = mot_tracker.update(frame,cnt,target)
    # result = mot_tracker.update(target)

    cnt_object = 1
    for i, box in enumerate(target):
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(width-1, int(box[2]))
        y2 = min(height-1, int(box[3]))
        
        # object_frmae = frame[y1:y2, x1:x2,:] 
        # cv2.imwrite("/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/object_det/"+str(cnt).zfill(4)+"_"+str(cnt_object).zfill(3)+".jpg",object_frmae)
        # cnt_object = cnt_object + 1
        color = [0, 0, 0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color)
        cv2.putText(
            frame,
            #"det" +str(round(box[-1], 5)), (x1, y2),
            "det", (x1, y2),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    for j, box in enumerate(result):
        boxid = int(box[4])
        boxlabel = int(box[5])
        point_x = int((box[0] + box[2])/2)
        point_y = int(box[3])
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(width-1, int(box[2]))
        y2 = min(height-1, int(box[3]))
        
        
    
        color = colors[boxid % len(colors)]
        if box[-1] != 1: 
            # color = [0, 0, 0]
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color)
        cv2.putText(
            frame,
            str(boxid) + '_' + str(boxlabel), (x1, y2),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    
    cv2.putText(frame,
                str(cnt) + "/" + str(all_frames), (0, 20),
                cv2.FONT_HERSHEY_DUPLEX, 1, [0, 255, 0], 1)
    cv2.imwrite(args.frame_save_path+str(cnt)+".png",frame)


    

    
    



