#ifndef _TRACKER_FACTORY_
#define _TRACKER_FACTORY_

#include "trackers/linear_tracker.hpp"
#include "trackers/mosse_tracker.hpp"
#include "trackers/fdsst_tracker.hpp"
#include "configs.hpp"

namespace mot{
    class TrackerFactory
    {
    public:
        MotTracker* createMotTracker(const mot::MotConfig &config, const mot::TrackerType type=TrackerType::Linear)
        {
            switch(type)
            {
                case Linear:
                    return (new LinearTracker(config));
                    break;
                case MOSSE:
                    return (new MosseTracker(config));
                    break;
                case fDSST:
                    return (new FDSSTracker(config));
                    break;
                default:
                    return NULL;
                    break;
            }
        }
    };
}
#endif