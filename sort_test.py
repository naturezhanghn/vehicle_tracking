# encoding: utf-8
import cv2
# import libs.pymot as pymot
import libs.pyalgorithm as pyalgorithm
from tqdm import tqdm
import numpy as np
import os
import argparse
import pdb
from collections import OrderedDict
import json
from sort.sort2 import Sort
# from sort.sort_origin import Sort

parser = argparse.ArgumentParser()
parser.add_argument('--video_path',
                    # default='/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/UA-DETRAC-TRACKING-TEST-VIDEOS/MVI_40172.mp4',
                    # default = r'/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/遮挡/a1.mp4',
                    default='/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/parkcarerror.mp4',
                    # default='/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/遮挡/a2.mp4',
                    # default = '/workspace/mnt/storage/zhangziran/zhangzr4/data/test_video/cover/a4.mp4',
                    help='video path')
parser.add_argument(
    '--tronmodel',
    default=
    'models/PeleeNet_ATSSv2_4class_det_20210311_e29cd266_OPENCV_960cut8-v2.13.4.tronmodel',
    help='tronmodel path')
parser.add_argument('--save_path',
                    default='sort_results',
                    help='path to save results')
parser.add_argument('--save_txt',
                    default='--save_txt',
                    action='store_true',
                    help='flag for save txt results or not')

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
    module_handle = pyalgorithm.ModuleHandle(tronmodel)
    detector = module_handle.load_detect(enable_profiler=True,
                                         backend_type=btype,
                                         use_fp16=fp16,
                                         cache_engine=True,
                                         device_id=device,
                                         max_batch_size=batchsize)
    return detector


def attribute_init(tronmodel,
                   btype='Native',
                   fp16=False,
                   device=0,
                   batchsize=1):
    """
    attribute model init
    Args:
        tronmodel: tronmodel path
        btype: backend type 'Native' or 'TensorRT'
        fp16: only set 'True' for btype using 'TensorRT'
        device: infer gpu id
        batchsize: number of images
    Returns: extract
    """
    module_handle = pyalgorithm.ModuleHandle(tronmodel)
    extract = module_handle.load_extract(enable_profiler=True,
                                         backend_type=btype,
                                         use_fp16=fp16,
                                         cache_engine=True,
                                         device_id=device,
                                         max_batch_size=batchsize)
    return extract


def plate_init(tronmodel, btype='Native', fp16=False, device=0, batchsize=1):
    """
    plate recognition init
    Args:
        tronmodel: tronmodel path
        btype: backend type 'Native' or 'TensorRT'
        fp16: only set 'True' for btype using 'TensorRT'
        device: infer gpu id
        batchsize: number of images
    Returns: ocr
    """
    module_handle = pyalgorithm.ModuleHandle(tronmodel)
    ocr = module_handle.load_ocr(enable_profiler=True,
                                 backend_type=btype,
                                 use_fp16=fp16,
                                 cache_engine=True,
                                 device_id=device,
                                 max_batch_size=batchsize)
    return ocr


# def tracker_init(config):
#     """
#     tracker init
#     Args:
#         src_path: config file
#     Returns: tracker
#     """
#     tracker = pymot.Tracker(config)
#     return tracker


def main():
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    #init detector
    traffic_detector = detector_init(args.tronmodel,
                                     btype='Native',
                                     fp16=True)
    #you can modify this cfg
    # cfg = pymot.getConfig()

    # tracker = tracker_init(cfg)
    mot_tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3) 

    caps = cv2.VideoCapture(args.video_path)
    fps = int(caps.get(cv2.CAP_PROP_FPS))
    width = int(caps.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = (args.video_path.split('/')[-1]).split('.')[0]

    videoWriters = cv2.VideoWriter(
        os.path.join(args.save_path, 
        args.video_path.split('.')[0].split('/')[-1] + '_result.mp4'),
        cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, (width, height))
    index = 0
    key = ord('r')
    max_id = 0
    all_frames = int(caps.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    if args.save_txt:
        f = open(os.path.join(args.save_path, name + '.txt'), 'w')
    tmp_res = []
    for frame_num in tqdm(range(all_frames)):
        ret, frame = caps.read()
        if not ret:
            break

        if frame_num % 2 == 1:  #隔1帧做跟踪
            continue
        # if frame_num>200:
        #     continue

        one_frame = OrderedDict()
        one_frame["frame_id"] = frame_id
        one_frame["data"] = []

        det_result = traffic_detector.run(data=[frame])
        target = []
        for res in det_result[0]:
            if res.score < 0.2858:
                continue
            x = res.xmin / frame.shape[1]
            y = res.ymin / frame.shape[0]
            w = (res.xmax - res.xmin) / frame.shape[1]
            h = (res.ymax - res.ymin) / frame.shape[0]
            if (res.xmax - res.xmin) >= MIN_WIDTH:
                target.append([x, y, w, h, res.score, res.label])

        # result = pymot.tracker_update(tracker, target)
        target = np.array([[res.xmin, res.ymin, res.xmax, res.ymax, res.label] for res in det_result[0] if res.xmax-res.xmin>=MIN_WIDTH and res.score >= 0.2858 and res.label==2])

        # result = mot_tracker.update(target)
        result = mot_tracker.update(frame,frame_num,target)

        for j, box in enumerate(result):
            boxid = int(box[4])
            boxlabel = int(box[5])
            point_x = int((box[0] + box[2])/2)
            point_y = int(box[3])
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(width-1, int(box[2]))
            y2 = min(height-1, int(box[3]))
            if True:
                color = colors[boxid % len(colors)]
                if box[-1] != 1: 
                    # color = [0, 0, 0]
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), color)
                cv2.putText(
                    frame,
                    str(boxid) + '_' + str(boxlabel), (x1, y2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                if box[-1] != 1: continue
                target = OrderedDict()
                target["id"] = boxid
                target["label"] = boxlabel
                target["bbox"] = [
                    max(0,x1/width), max(0,y1/height), (x2-x1)/width, (y2-y1)/height
                ]
                if boxlabel == 2:
                    one_frame["data"].append(target)
                if boxid == 1:
                    tmp_res.append((frame_num, [x1, y1, x2, y2]))
        one_frame["width"] = frame.shape[1]
        one_frame["height"] = frame.shape[0]
        if args.save_txt:
            f.write(json.dumps(one_frame))
            f.write('\n')
        cv2.putText(frame,
                    str(frame_num) + "/" + str(all_frames), (0, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1, [0, 255, 0], 1)
        videoWriters.write(frame)
        print("------------")

        frame_id += 1
    if args.save_txt:
        f.close()
    caps.release()
    videoWriters.release()
    for res in tmp_res:
        print(res[0], res[1])


if __name__ == '__main__':
    main()
