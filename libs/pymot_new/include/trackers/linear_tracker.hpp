#ifndef _LINEAR_TRACKER_
#define _LINEAR_TRACKER_
#include "trackers/base_tracker.hpp"

namespace mot{
    class LinearTracker : public MotTracker
    {
    public:
        LinearTracker(const MotConfig &config) : MotTracker(config){};

    protected:
        virtual void predict() override;
        virtual void createTracker(vector<vector<DetBox>> &objects) override;
        virtual void updateTrackingCue(DetBox &obj, Tracket &tracket) override;
    };
}
#endif