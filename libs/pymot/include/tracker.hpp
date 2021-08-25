#ifndef _MOT_TRACKER_
#define _MOT_TRACKER_

#include "core.hpp"
#include "matrix.hpp"
#include "configure.hpp"

namespace mot {

    struct MotConfig{
        //类别数量
        int numClass;
        //目标生成初始帧数
        int initFrame;
        //最大目标数量
        int maxTracker;
        //目标丢弃的有效期
        int maxLife;
        //是盃开启reid，目前只对人有效
        int useReid;
        //是否开启光流，预留
        int useFlow;
        //人与非机动车/三轮车合并的IOU阈值
        float mergeIouThres;
        //类别的列表
        std::vector<int> classList;
        //局部匹配阈值，按类别
        std::vector<float> clsLocalThres;
        //全局匹配阈值，按类别
        std::vector<float> clsGlobalThres;
        //目标直接合并阈值
        std::vector<float> miniAssign;
        //人与非机动车/三轮车合并的类别
        std::vector<std::vector<int>> mergeList;
        //是否开启动态阈值，在城市场景建议开启，高速不开启
        bool dynaThres;
        //配置的版本号
        long version;
        int level;
    };

    class Tracker {
    public:
        Tracker(MotConfig &config);
        std::vector<TracketBox> update(std::vector<std::vector<DetBox>> &objects);
        //~Tracker();
        int reset();
        long getVersion()
        {
            return YEAR<<7+MONTH<<5+DAY;
        }
    private:

        std::vector<std::vector<Tracket>> trackets;
        int numClass;     //class num
        int frame_num;    //cur frame num
        int initFrame;    //tracket init frame size
        int maxTracker;   //max tracker size
        int keep_max_life;//tracker lost max frame
        int useReid;      //is using reid?
        int useFlow;      //
        bool dynaThres;   //is open dyna thres
        float mergeIouThres; //iou thres for merge objs
        float addThres;
        float removeThres;
        int level;
        float addValue;
        size_t idCount;   //id accumulate
//        int support_class;//
        std::vector<float> localThres;
        std::vector<float> globalThres;
        std::vector<float> miniAssign;



        std::vector<std::vector<float>> costs;
        std::vector<std::vector<int>> mergeList;

        float calCost(const DetBox &obj, const Tracket &tracket);

        void localTrack(std::vector<std::vector<DetBox>> &objects);

        void globalTrack(std::vector<std::vector<DetBox>> &objects);

        void cretaTracker(std::vector<std::vector<DetBox>> &objects);

        void predict();

        void cleanTracker();

        void setTracker();

        void find_tracket_match(Matrix<double> &matrix, std::vector<int> &matck_tracket, int row, int col);

        void mini_tracket_assign(std::vector<std::vector<DetBox>> &objects, int label);

        void updateVelocity(const DetBox &obj, Tracket &tracket);

        void normalizeReidFeat(std::vector<std::vector<DetBox>> &objects);

        void mergeTrackets();

    };


}
#endif
