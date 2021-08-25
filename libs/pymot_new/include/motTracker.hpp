#ifndef _MotTracker_
#define _MotTracker_

#include "configs.hpp"
namespace mot{
    class MOTTracker 
    {
    public:
        MOTTracker();
        ~MOTTracker();

        void initMotTracker(const MotConfig &config, const TrackerType type);
        vector<TracketBox> runMotTracker(vector<vector<DetBox>> &objects);
    };
}
#endif