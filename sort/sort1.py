from os import read, wait
from pickle import FRAME
import numpy as np
import matplotlib
matplotlib.use('Agg')
from filterpy.kalman import KalmanFilter
from skimage.feature import hog
import cv2

np.random.seed(0)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0])
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):

  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):

  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):

  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


def area_compare(bb_test, bb_gt):
  test_a = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + 1e-8
  gt_a = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) + 1e-8
  test_a = np.expand_dims(test_a, 1)
  gt_a = np.expand_dims(gt_a, 0)
  o = test_a/gt_a
  return (o)

# 对遮挡部分mask
def getmask(dets,frame):
  w_,h_, c_ = frame.shape
  mask_blank = np.ones((w_,h_,c_))
  for d in dets:
    x1 = max(0, int(d[0]))
    y1 = max(0, int(d[1]))
    x2 = min(w_-1, int(d[2]))
    y2 = min(h_-1, int(d[3]))
    mask_blank[y1:y2,x1:x2,:] = mask_blank[y1:y2,x1:x2,:] + 1
  return mask_blank<=2

# def get_variance(det_object):
#   len_no0 = np.sum( det_object[:,:,0] != 0) 
#   gray_average = np.sum(det_object[:,:,0]) / (len_no0 + 0.000001)
#   gray_variance = 0
#   gray_list = det_object[:,:,0].flatten()
#   for i in gray_list:
#     if i != 0 :
#       gray_variance += (i - gray_average)*(i - gray_average)/ (len_no0 + 0.000001)
#   return gray_variance

def color_moments(img_rgb):
  hsv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)
  h,s,v = cv2.split(hsv)
  color_feature = []
  h_mean = np.mean(h)
  s_mean = np.mean(s)
  #v_mean = np.mean(v)
  #color_feature.extend([h_mean,s_mean,v_mean])
  color_feature.extend([h_mean,s_mean])
  h_std = np.std(h)
  s_std = np.std(s)
  # v_std = np.std(v)
  # color_feature.extend([h_std,s_std,v_std])
  color_feature.extend([h_std,s_std])
  h_skewness = np.mean(abs(h-h.mean())**3)
  s_skewness = np.mean(abs(s-s.mean())**3)
  # v_skewness = np.mean(abs(v-v.mean())**3)
  h_thirdMoment = h_skewness**(1./3)
  s_thirdMoment = s_skewness**(1./3)
  # v_thirdMoment = v_skewness**(1./3)
  # color_feature.extend([h_thirdMoment,s_thirdMoment,v_thirdMoment])
  color_feature.extend([h_thirdMoment,s_thirdMoment])
  return color_feature

# 计算特征
def cal_feature(d,width,height,frame,cnt_a):
  x1 = max(0, int(d[0]))
  y1 = max(0, int(d[1]))
  x2 = min(width-1, int(d[2]))
  y2 = min(height-1, int(d[3]))
  det_object = frame[ np.int(y1+(y2-y1)/6) : np.int(y2-(y2-y1)/6) , np.int(x1+(x2-x1)/6) : np.int(x2-(x2-x1)/6)]
  # det_object = frame[y1:y2,x1:x2]
  
  '''
  # 直方图
  det_hist_b = np.squeeze(cv2.calcHist([det_object],[0],None,[51],[1,255]))
  det_hist_g = np.squeeze(cv2.calcHist([det_object],[1],None,[51],[1,255]))
  det_hist_r = np.squeeze(cv2.calcHist([det_object],[2],None,[51],[1,255]))
  det_hist = np.hstack((det_hist_b,det_hist_g,det_hist_r))
  det_hist_average = (det_hist / sum(det_hist))*100
  # det_hist =  np.squeeze(cv2.calcHist([det_object],[0],None,[5],[1,255]))

  # 颜色
  b = np.sum(det_object[:,:,0]) / (np.sum( det_object[:,:,0] != 0) + 0.000001)
  g = np.sum(det_object[:,:,1]) / (np.sum( det_object[:,:,1] != 0) + 0.000001)
  r = np.sum(det_object[:,:,2]) / (np.sum( det_object[:,:,2] != 0) + 0.000001)
  det_bgr = [b,g,r]
  det_bgr_average = (det_bgr / sum(det_bgr))*100

  # 方差
  det_object_flatten = det_object.flatten()
  det_no0 = np.delete(det_object_flatten, np.where(det_object_flatten == 0))
  det_varaince = np.var(det_no0 )
  '''
  # 位置
  location = [x1,x2 ,y1,y2]

  # 颜色矩
  det_color_moments = color_moments(det_object)

  det_feature = np.hstack((det_color_moments,location, cnt_a))
  return  det_feature 

# 保存检测目标
def save_object(d,width,height,frame,path):
  x1 = max(0, int(d[0]))
  y1 = max(0, int(d[1]))
  x2 = min(width-1, int(d[2]))
  y2 = min(height-1, int(d[3]))
  det_object = frame[y1:y2,x1:x2]
  cv2.imwrite(path,det_object )

#获得检测目标
def get_object(d,width,height,frame):
  x1 = max(0, int(d[0]))
  y1 = max(0, int(d[1]))
  x2 = min(width-1, int(d[2]))
  y2 = min(height-1, int(d[3]))
  det_object = frame[y1:y2,x1:x2]
  return det_object
 
class KalmanBoxTracker(object):
  
  count = 0
  def __init__(self,bbox):
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 1.  # KF.measurementNoiseCov
    self.kf.P[4:,4:] *= 1000. # KF.errorCovPost give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01  # KF.processNoiseCov
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    KalmanBoxTracker.count += 1
    self.id = KalmanBoxTracker.count
    
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.label = bbox[-1]
    
    self.feature = [ ]
    self.trajectory = []
  
  def update(self,bbox):
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))
    self.feature = self.feature[-10:]
    self.trajectory.append(bbox[:4])

  def predict(self):
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):

  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  if len(detections) > 0:
    iou_matrix = iou_batch(detections, trackers)
    # area_matrix = area_compare(detections, trackers)

    if min(iou_matrix.shape) > 0:
      a = (iou_matrix > iou_threshold).astype(np.int32)
      if a.sum(1).max() == 1 and a.sum(0).max() == 1:
          matched_indices = np.stack(np.where(a), axis=1)
      else:
        matched_indices = linear_assignment(-iou_matrix)
    else:
      matched_indices = np.empty(shape=(0,2))
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<max(0.1, iou_threshold * pow(0.8, trackers[m[1]][-1]))):
    # if iou_matrix[m[0], m[1]]<iou_threshold:
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)
  return matches, (unmatched_detections), (unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
 
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
    # 辅助计数器
    self.flag = 0
    self.flag2 = 0
    self.flag3 = 0

  def update(self,frame ,cnt_a, dets=np.empty((0, 5))):
    
    self.frame_count += 1
    h, w, c = frame.shape
    judge_thred = 20

    # 利用卡尔曼滤波器预测轨迹
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    img = np.zeros((200,200,3))
    cv2.putText(img ,
                str( cnt_a), (0, 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, [0, 255, 0], 1)
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].time_since_update]
      # try:
      #   trk_plot =  cv2.resize(get_object(pos ,w,h,frame) , (200,200))
      #   img = np.hstack(( trk_plot ,img ))
      # except:
      #     pass
      if np.any(np.isnan(pos)):
        to_del.append(t)
    # cv2.imwrite("/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/a2_all_trk_frame/"+str(cnt_a)+".png",img)   
    # trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    
    for t in reversed(to_del):
      self.trackers.pop(t)
    

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets ,trks , self.iou_threshold )
     
    # pathk = '/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/a6_macthed_feature/'###
    # flagk = 0  ###
    for m in matched:
      dets_object = get_object(dets[m[0],:],w,h,frame)
      # dets_object = cv2.resize(dets_object_,(200,200))
      numof0 = dets_object.flatten()
      prop_of_mask = np.sum(numof0==0)/len(numof0)
      # print("比例：",prop_of_unmask)
      if prop_of_mask > 0.5:
        unmatched_dets.append(m[0])
        unmatched_trks.append(m[1])
        continue  
        
      feature = cal_feature(dets[m[0],:],w,h,frame,cnt_a)
      if len(self.trackers[m[1]].feature)>1:
        flag_matched_judge = -1
        for old_feature in self.trackers[m[1]].feature:
          judge_rgb = np.sum(np.abs(feature[0:3] - old_feature[0:3]))
          if judge_rgb > judge_thred:
            judge_rgb = judge_thred + np.log(judge_rgb) - np.log(judge_thred)
          judge_hist = np.sum(np.abs(feature[3:156] - old_feature[3:156])) 
          if judge_hist > judge_thred:
            judge_hist =  judge_thred + np.log(judge_hist) - np.log(judge_thred)
          judge_variance = np.sqrt(np.abs(feature[156] - old_feature[156]))
          if judge_variance > judge_thred:
            judge_variance = judge_thred + np.log(judge_variance) - np.log(judge_thred)
          # judge_hog = np.sum(np.abs(feature[157:2461] - old_feature[157:2461]))
          # print(judge_hog)

          # print(judge_variance ,feature[156] ,old_feature[156] )
          # judge = (judge_rgb * judge_hist * judge_variance )/20 # 原来的代码
          # judge = 0.075 * judge_rgb * judge_hist    # 原来的代码
          judge = 0.075 * judge_rgb * judge_hist    # 原来的代码
          
        
          # judge = (judge_rgb * judge_hist)

          if judge < judge_thred :
            flag_matched_judge = 1
            break
        if flag_matched_judge == 1:
          self.trackers[m[1]].update(dets[m[0], :])
          self.trackers[m[1]].feature.append(feature)
        else:
          unmatched_dets.append(m[0])
          unmatched_trks.append(m[1])

          # object_matched_trk = cv2.resize(get_object(self.trackers[m[1]].get_state()[0],w,h,frame),(200,200))###
          # object_matched_det = cv2.resize(get_object(dets[m[0], :],w,h,frame) ,(200,200))###
          # object_matched = np.hstack((object_matched_trk,object_matched_det))###
          # flagk += 1 ###
          # cv2.imwrite(pathk+str(cnt_a)+"_"+str(flagk)+".jpg" ,object_matched)###
      else:
        self.trackers[m[1]].update(dets[m[0], :])
        self.trackers[m[1]].feature.append(feature)

    # img2 = np.zeros((200,200,3))
    # for i in unmatched_trks :
    #   try:
    #     trks_object_ = cv2.resize(get_object(self.trackers[i].get_state()[0],w,h,frame) ,(200,200))
    #     cv2.putText(trks_object_ ,
    #             str(cnt_a)+" num:"+str(i), (0, 10),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, [0, 255, 0], 1)
    #     img2 = np.hstack(( trks_object_ ,img2 ))
    #   except:
    #       pass
    # path3  = "/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/a3_unmatched_trks/"+ str( cnt_a)+".png"
    # cv2.imwrite(path3, img2 )

    # 重叠的区域用mask遮挡掉，防止误判
    mask =  getmask(dets,frame)
    # cv2.imwrite("mask/"+str(cnt_a)+".png",mask*255)
    frame = frame * mask

    for i in unmatched_dets:
      dets_object =  get_object(dets[i][0:4],w,h,frame)
      # dets_object =  cv2.resize(dets_object_,(200,200))
      numof0 = dets_object.flatten()
      prop_of_unmask = np.sum(numof0==0)/len(numof0)
      # print("比例：",prop_of_unmask)
      if prop_of_unmask > 0.85:
        continue
      w_,h_, c_ = dets_object.shape
      self.flag2 = self.flag2 + 1
      min_judge = judge_thred
      min_distance =  w_  * w_  /2

      # min_val = np.log(min_judge + 2.7) * min_distance 
      matched_id = -1
      
      # path  = "/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/a4_unmatched_dets/"+ str( cnt_a)+".png"
      # cv2.putText(dets_object ,
      #           str(cnt_a)+" num:"+str(i) , (0, 10),
      #           cv2.FONT_HERSHEY_DUPLEX, 0.5, [0, 255, 0], 1)
      # cv2.imwrite( path , dets_object )
      un_matched_feature = cal_feature(dets[i],w,h,frame,cnt_a)
      for j in unmatched_trks:  
        trk_features = self.trackers[j].feature
        if self.trackers[j].time_since_update < 10:
          loss_time = self.trackers[j].time_since_update
        else:
          loss_time = 10

        cnt11 = 0
        for trk_feature in trk_features:
          cnt11 = cnt11 + 1
          judge_bgr = np.sum(np.abs(un_matched_feature[0:3] - trk_feature[0:3]))
          judge_var = np.sum(np.abs(un_matched_feature[156] - trk_feature[156]))
          judge = judge_bgr*np.log(judge_var+1)/6
          # print(np.log(judge_var+1)/3 )
          judge_distance = np.sum(np.square(un_matched_feature[-5:-1] - trk_feature[-5:-1])) / 4 
          if  (judge < min_judge) and (judge_distance  <  min_distance * np.log(loss_time + 2.73))  :
          # ( np.log(judge + 2.7 ) * distance <  min_val  ) and
          # if ( (judge < min_judge ) and (distance <  min_distance ) ) or ( (judge < 0.5 * min_judge ) and (distance <  2 * min_distance ) )\
          #                                                             or ( (judge < 1.5 * min_judge ) and (distance <  0.5 * min_distance ) ):
                                                                      # or ( (judge < 0.2 * min_judge ) and (distance <  4 * min_distance ) )\
                                                                      # or ( (judge < 2 * min_judge ) and (distance <  0.2 * min_distance ) ):
            # trks_object_ = get_object(self.trackers[j].get_state()[0],w,h,frame)   
            # try:
            if (judge_distance < min_distance ):
              min_distance = judge_distance 
            if (judge < min_judge):
              min_judge = judge
            matched_id = j
            #   trks_object = cv2.resize(trks_object_,(200, 200))
            #   matched_frame_num = trk_feature[-1]
            #   cv2.putText(trks_object ,
            #       str(matched_frame_num)+"/"+str(min_judge) , (0, 10),
            #       cv2.FONT_HERSHEY_DUPLEX, 0.5, [0, 255, 0], 1) 
            #   dets_object = np.hstack((dets_object,trks_object)) 
            # except:
            #     pass
      if matched_id == -1:
        trk = KalmanBoxTracker(dets[i,:] )
        self.trackers.append(trk)
      else:
        self.trackers[matched_id].update(dets[i, :])
        # path1  = "/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/a5_feature_matched/" + str( self.flag) + "det_trks" + ".jpg"
        # self.flag = self.flag + 1
        # cv2.imwrite(path1,dets_object )
    
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          if len(trk.trajectory) > 0:
            d = trk.trajectory[-1]
          ret.append(np.concatenate((d,[trk.id], [trk.label], [1])).reshape(1,-1))

        else:
          ret.append(np.concatenate((d,[trk.id], [trk.label], [0])).reshape(1,-1))
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,7))
    
