#ifndef _KALMAN_TRACKER_
#define _KALMAN_TRACKER_
#include "trackers/base_tracker.hpp"

namespace mot{
    class KalmanTracker : public MotTracker
    {
    public:
        KalmanTracker(const MotConfig &config) : MotTracker(config){};
        virtual vector<TracketBox> update(vector<vector<DetBox>> &objects) override;
        void TrackMatching(vector<vector<DetBox>> &objects);

    protected:
        virtual void predict() override;
        virtual void createTracker(vector<vector<DetBox>> &objects) override;
        virtual void updateTrackingCue(DetBox &obj, Tracket &tracket) override;
        virtual void setTracker() override;
    };
}
#endif