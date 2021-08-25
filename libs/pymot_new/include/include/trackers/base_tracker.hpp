#ifndef _MOT_TRACKER_
#define _MOT_TRACKER_

#include <iostream>
#include <string.h>
#include "configs.hpp"
#include <opencv2/opencv.hpp>
using namespace std;

namespace mot{
    class MotTracker
    {
    public:
        MotTracker(const MotConfig &config);
        virtual vector<TracketBox> update(vector<vector<DetBox>> &objects);

        void reset();

    protected:
        void InvconvertCoor();

        float feaCost(const DetBox &obj, const Tracket &tracket);

        float iouCost(const DetBox &obj, const Tracket &tracket);

        virtual void firstTrack(vector<vector<DetBox>> &objects);

        virtual void secondTrack(vector<vector<DetBox>> &objects);

        virtual void lastTrack(vector<vector<DetBox>> &objects);

        void cleanTracker();

        void setTracker();

        void find_tracket_match(Matrix<double> &matrix, vector<int> &matck_tracket, int row, int col);

        void normalizeReidFeat(vector<vector<DetBox>> &objects);

        void mergeTrackets();

        virtual bool convertCoor(vector<vector<DetBox>> &objects);

        virtual void updateTrackingCue(DetBox &obj, Tracket &tracket) = 0;
        virtual void createTracker(vector<vector<DetBox>> &objects) = 0;
        virtual void predict() = 0;

    public:

        vector<vector<Tracket>> trackets;

    protected:

        int numClass;     //class num
        int frame_num;    //cur frame num
        int initFrame;    //tracket init frame size
        int maxTrackerNum;   //max tracker size
        int keep_max_life;//tracker lost max frame
        int useReid;      //using reid
        int useFlow;      //using flow
        size_t idCount;   //id accumulate
        
        vector<float> firstStageThres;
        vector<float> secondStageThres;
        vector<float> lastStageThres;
        
        vector<vector<float>> costs;
        vector<vector<int>> mergeList;
        vector<float> mergeIouThres;
        cv::Mat image;
    };
}
#endif