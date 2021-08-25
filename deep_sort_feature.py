# encoding: utf-8
from deep_sort.deep_sort.detection import Detection
import cv2
import libs.pymot as pymot
import libs.pyalgorithm as pyalgorithm
from tqdm import tqdm
import numpy as np
import os
import argparse
import pdb
from collections import OrderedDict
import json
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from skimage.feature import hog


parser = argparse.ArgumentParser()
parser.add_argument('--video_path',
                    default='videos/example.mp4',
                    help='video path')
parser.add_argument(
    '--tronmodel',
    default=
    'models/PeleeNet_ATSSv2_4class_det_20210311_e29cd266_OPENCV_960cut8-v2.13.4.tronmodel',
    help='tronmodel path')
parser.add_argument('--save_path',
                    default='./sort_results',
                    help='path to save results')
parser.add_argument('--save_txt',
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


def tracker_init(config):
    """
    tracker init
    Args:
        src_path: config file
    Returns: tracker
    """
    tracker = pymot.Tracker(config)
    return tracker


def compute_hog(image, x, y, w, h):
    patch = image[int(y):int(y+h), int(x):int(x+w)]
    patch = cv2.resize(patch, (128, 128))
    # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    fd = hog(patch, orientations=9, pixels_per_cell=[32,32], cells_per_block=[2,2], visualize=False, transform_sqrt=True,block_norm='L2-Hys')
    # print(fd.shape)
    return fd


def concat_fix_height(frame, target_patch, patchs, scores):
    w = 256
    image = frame[int(target_patch[1]):int(target_patch[1]+target_patch[3]), int(target_patch[0]):int(target_patch[0]+target_patch[2])]
    new_w = int(target_patch[2] / target_patch[3] * w)
    image = cv2.resize(image, (new_w, w))
    for i, img in enumerate(patchs):
        new_w = int(img.shape[1] / img.shape[0] * w)
        img = cv2.resize(img, (new_w, w))
        cv2.putText(img,
                    str(scores[i])[:6], (0, w),
                    cv2.FONT_HERSHEY_DUPLEX, 1, [0, 255, 0], 2)
        image = np.hstack([image, img])
        
    return image


def concat_v(images):
    img = images[0]
    for i in range(1, len(images)):
        h1, w1, c1 = img.shape
        h2, w2, c2 = images[i].shape
        if w1 > w2:
            tmp = np.zeros([h2, w1-w2, c1])
            tmp_img = np.hstack([images[i], tmp])
            img = np.vstack([img, tmp_img])
        elif w2 > w1:
            tmp = np.zeros([h1, w2-w1, c2])
            tmp_img = np.hstack([img, tmp])
            img = np.vstack([tmp_img, images[i]])
        else:
            img = np.vstack([img, images[i]])
    return img


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
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.1, 50)
    mot_tracker = Tracker(metric, n_init=1)

    caps = cv2.VideoCapture(args.video_path)
    fps = int(caps.get(cv2.CAP_PROP_FPS))
    width = int(caps.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = (args.video_path.split('/')[-1]).split('.')[0]

    all_frames = int(caps.get(cv2.CAP_PROP_FRAME_COUNT))

    former_targets = []
    former_features = []
    former_bboxes = []
    frame_id = 0
    if args.save_txt:
        f = open(os.path.join(args.save_path, name + '.txt'), 'w')
    for frame_num in tqdm(range(all_frames)):
        ret, frame = caps.read()
        if not ret:
            break

        if frame_num % 2 == 1:  #隔1帧做跟踪
            continue

        one_frame = OrderedDict()
        one_frame["frame_id"] = frame_id
        one_frame["data"] = []

        det_result = traffic_detector.run(data=[frame])
        target = []
        feature = []
        bboxes = []
        print(len(det_result[0]))
        tmp_image = frame.copy()
        for res in det_result[0]:
            cv2.rectangle(tmp_image, (int(res.xmin), int(res.ymin)), (int(res.xmax), int(res.ymax)), [0, 0, 255])
            cv2.putText(tmp_image,
                    str(res.score)[:6], (int(res.xmin/2+res.xmax/2), int(res.ymax)),
                    cv2.FONT_HERSHEY_DUPLEX, 1, [0, 255, 0], 1)
            
            if res.score < 0.2858:
                continue
            if res.label != 2:
                continue
            x = res.xmin
            y = res.ymin
            w = (res.xmax - res.xmin)
            h = (res.ymax - res.ymin)
            if (res.xmax - res.xmin) >= MIN_WIDTH:
                fea = np.reshape(compute_hog(frame, x, y, w, h), (-1))
                # print(fea.shape)
                target.append(Detection([x, y, w, h], res.score, fea))
                feature.append(fea)
                bboxes.append(frame[int(y):int(y+h), int(x):int(x+w)])
        cv2.imwrite('./deep_sort_results/' + str(frame_num) + '_bbox.jpg', tmp_image)
        # if frame_num//2 in range(945, 952):
        #     print(target)

        # result = pymot.tracker_update(tracker, target)
        # target = np.array([[res.xmin, res.ymin, res.xmax, res.ymax, res.label] for res in det_result[0] if res.xmax-res.xmin>=MIN_WIDTH and res.score >= 0.2858 and res.label==2])
        # print(target)
        print(len(former_features), len(feature))
        if len(former_targets) > 0:
            res = nn_matching._cosine_distance(feature, former_features)
            tops = np.argsort(res, axis=1)
            # print(tops.shape)
            # print(res[:5])
            # print(tops.shape)
            concated = []
            for kk in range(len(target)):
                concat_image = concat_fix_height(frame, target[kk].tlwh, [former_bboxes[tops[kk][jj]] for jj in range(5)], [res[kk][tops[kk][jj]] for jj in range(5)])
                concated.append(concat_image)
            final_image = concat_v(concated)
            cv2.imwrite('./deep_sort_results/' + str(frame_num) + '.jpg', final_image)
        former_targets.extend(target)
        former_targets = former_targets[-500:]
        former_features.extend(feature)
        former_features = former_features[-500:]
        former_bboxes.extend(bboxes)
        former_bboxes = former_bboxes[-500:]
    caps.release()


if __name__ == '__main__':
    main()
