#ifndef _MOT_CONFIG_
#define _MOT_CONFIG_
#include <vector>
#include <iostream>

namespace mot {
    struct MotConfig{
        //类别数量
        int numClass;
        //目标生成初始帧数
        int initFrame;
        //最大目标数量
        int maxTrackerNum;
        //目标丢弃的有效期
        int maxLife;
        //是否开启外部特征, 预留
        int useReid;
        //是否开启光流，预留
        int useFlow;
        //特征阈值，按类别
        std::vector<float> firstStageThres;
        //IOU阈值，按类别
        std::vector<float> secondStageThres;
        //IOU阈值，按类别
        std::vector<float> lastStageThres;
        //人与非机动车/三轮车合并的类别
        std::vector<std::vector<int>> mergeList;
        //人与非机动车/三轮车合并的IOU阈值
        std::vector<float> mergeIouThres;
        //配置的版本号
        long version;
    };

    enum TrackerType{
        Linear, 
        MOSSE, 
        fDSST,
        Kalman
        };
}
#endif


