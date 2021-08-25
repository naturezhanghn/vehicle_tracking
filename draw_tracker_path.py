# encoding: utf-8
import argparse
import os
import cv2
import json
from collections import OrderedDict
from tqdm import tqdm
import scipy
import numpy as np
from scipy.signal import savgol_filter


parser = argparse.ArgumentParser()
parser.add_argument('--video_path',
                    default='videos/example.mp4',
                    help='video path')
parser.add_argument('--save_video_path',
                    default='./results/example_path.mp4',
                    help='path to save results')
parser.add_argument('--track_file',
                    default='./results/example.txt',
                    help='path to track path txt')

colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [0, 69, 255],
          [203, 192, 255], [255, 255, 255], [139, 139, 0], [30, 105, 210],
          [169, 169, 169], [0, 0, 139], [10, 215, 255], [0, 128, 128],
          [144, 238, 144], [230, 216, 173], [130, 0, 75], [128, 0, 128],
          [147, 20, 255], [238, 130, 238]]


def draw(args):
    caps = cv2.VideoCapture(args.video_path)
    fps = int(caps.get(cv2.CAP_PROP_FPS))
    width = int(caps.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_frames = int(caps.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(all_frames)
    name = (args.video_path.split('/')[-1]).split('.')[0]
    videoWriters = cv2.VideoWriter(
        args.save_video_path,
        cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, (width, height))
    centers = dict()
    frames = dict()
    all_boxes = dict()
    target_order = dict()
    target_cnt = 0
    with open(args.track_file) as tracklog:
        ls = tracklog.readlines()
        for li in ls:
            js = json.loads(li)
            fid = js['frame_id']
            frames[fid] = set()
            bboxes = js['data']
            for bbox in bboxes:
                nid = bbox['id']
                frames[fid].add(nid)
                if nid not in centers.keys():
                    centers[nid] = OrderedDict()
                    target_order[nid] = target_cnt
                    target_cnt += 1
                if nid not in all_boxes.keys():
                    all_boxes[nid] = OrderedDict()
                bbox_x = bbox['bbox'][0]
                bbox_y = bbox['bbox'][1]
                bbox_w = bbox['bbox'][2]
                bbox_h = bbox['bbox'][3]
                center_x = int(bbox_x * width + bbox_w * width / 2)
                center_y = int(bbox_y * height + bbox_h * height / 2)
                centers[nid][fid] = (center_x, center_y)
                all_boxes[nid][fid] = bbox['bbox']

        points = dict()
        for nid in list(centers.keys()):
            array_x = []
            array_y = []
            for k in centers[nid]:
                array_x.append(centers[nid][k][0])
                array_y.append(centers[nid][k][1])
            array_x = scipy.signal.savgol_filter(array_x, 21, 3, mode='nearest')
            array_y = scipy.signal.savgol_filter(array_y, 21, 3, mode='nearest')
            fids = list(centers[nid].keys())
            array_loc = []
            for k in range(len(fids)):
                array_loc.append((fids[k], (array_x[k].astype(np.int64), array_y[k].astype(np.int64))))
            points[nid] = array_loc

    print('process log ok')
    for frame_num in tqdm(range(all_frames)):
        ret, frame = caps.read()
        if not ret:
            break

        if frame_num % 2 == 1:  #隔1帧做跟踪
            continue

        for nid in frames[frame_num//2]:
            color = colors[nid % len(colors)]
            # print(nid, all_boxes[nid].keys())
            cv2.rectangle(frame, (int(all_boxes[nid][frame_num//2][0]*width), 
                                int(all_boxes[nid][frame_num//2][1]*height)), 
                                (int(all_boxes[nid][frame_num//2][0]*width + all_boxes[nid][frame_num//2][2]*width), 
                                int(all_boxes[nid][frame_num//2][1]*height + all_boxes[nid][frame_num//2][3]*height)), color)
            cv2.putText(
                    frame,
                    str(target_order[nid]) + '_' + str(nid), (int(all_boxes[nid][frame_num//2][0]*width + all_boxes[nid][frame_num//2][2]*width/2), 
                    int(all_boxes[nid][frame_num//2][1]*height + all_boxes[nid][frame_num//2][3]*height)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            ncenter = points[nid]
            for cid in range(len(ncenter) - 1):
                fid = ncenter[cid][0]
                if fid < frame_num//2 - 200:
                    continue
                elif fid > frame_num//2:
                    break
                else:
                    cv2.line(frame, ncenter[cid][1], ncenter[cid+1][1], color)
        videoWriters.write(frame)
    
    caps.release()
    videoWriters.release()


if __name__ == '__main__':
    args = parser.parse_args()
    draw(args)