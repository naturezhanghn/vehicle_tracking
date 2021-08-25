#ifndef _MOT_TRACKER_
#define _MOT_TRACKER_

#include "core.hpp"
#include "matrix.hpp"
#include "configure.hpp"
using namespace std;
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
        //预测框冲量
        float momentum;
    };

    class Tracker {
    public:
        Tracker(const MotConfig &config);
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
        int maxTrackerNum;   //max tracker size
        int keep_max_life;//tracker lost max frame
        int useReid;      //using reid
        int useFlow;      //using flow
        size_t idCount;   //id accumulate

        float addThres;
        float removeThres;
        float addValue;
        float momentum;
    
        vector<float> firstStageThres;
        vector<float> secondStageThres;
        vector<float> lastStageThres;
    
        vector<vector<float>> costs;
        vector<vector<int>> mergeList;
        vector<float> mergeIouThres;

        float feaCost(const DetBox &obj, const Tracket &tracket);
        float iouCost(const DetBox &obj, const Tracket &tracket);

        void firstTrack(vector<vector<DetBox>> &objects);

        void secondTrack(vector<vector<DetBox>> &objects);

        void lastTrack(vector<vector<DetBox>> &objects);

        void createTracker(std::vector<std::vector<DetBox>> &objects);

        void predict();

        void cleanTracker();

        void setTracker();

        void find_tracket_match(Matrix<double> &matrix, std::vector<int> &matck_tracket, int row, int col);

        void updateTrackingCue(const DetBox &obj, Tracket &tracket);

        void normalizeReidFeat(std::vector<std::vector<DetBox>> &objects);

        void mergeTrackets();

    };


}
#endif
