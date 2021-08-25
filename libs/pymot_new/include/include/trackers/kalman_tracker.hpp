#ifndef _KALMAN_TRACKER_
#define _KALMAN_TRACKER_
#include "trackers/base_tracker.hpp"

namespace mot{
    class KalmanTracker : public MotTracker
    {
    public:
        KalmanTracker(const MotConfig &config) : MotTracker(config){};

    protected:
        virtual void predict() override;
        virtual void createTracker(vector<vector<DetBox>> &objects) override;
        virtual void updateTrackingCue(DetBox &obj, Tracket &tracket) override;
    };
}
#endif