#ifndef _FDSST_TRACKER_
#define _FDSST_TRACKER_

#include "trackers/base_tracker.hpp"

namespace mot{
    class FDSSTracker: public MotTracker
    {
    public:
        FDSSTracker(const MotConfig &config) : MotTracker(config) {}
        void init(DetBox &obj, Tracket &tracket);
        
        void Tracking(Tracket &tracket);

    private:
        // detect object in the current frame
        cv::Point2f detect(Tracket &tracket, cv::Mat x, float &peak_value);

        // train tracker with a single image
        void train(Tracket &tracket, cv::Mat x, float train_interp_factor);

        //evaluates a gaussion kernel with bandwidth sigma for all relative shifts between input images X and Y, which must both be
        //MxN. They must also be periodic(ie., pre-processed with a cosine window)
        cv::Mat gaussianCorrelation(Tracket &tracket, cv::Mat x1, cv::Mat x2);

        // obtainsub-window from image, with replication-padding and extract features
        cv::Mat getFeatures(Tracket &tracket, bool inithann, float scale_adjust = 1.0f);

        // initialize hanning window. function called only in the first frame
        void createHanningMats(Tracket &tracket);

        // create gaussion peak. function called only in the first frame
        cv::Mat createGaussionPeak(Tracket &tracket, int sizey, int sizex);

        // calculate sub-pixel peak for one dimension
        float subPixelPeak(float left, float center, float right);

        // compute the hanning window for scaling
        cv::Mat createHanningMatsForScale(Tracket &tracket);

        // initialization for scales
        void dsstInit(Tracket &tracket);

        // compute the F^l in the paper
        cv::Mat get_scale_sample(Tracket &tracket);

        // update the ROI size after training
        void update_roi(Tracket &tracket);

        // Train method for scaling
        void train_scale(Tracket &tracket, bool ini = false);

        cv::Mat resizeDFT(Tracket &tracket, const cv::Mat & A, int real_scales);

        // detect the new scaling rate
        cv::Point2i detect_scale(Tracket &tracket);

        cv::Mat features_projection(Tracket &tracket, const cv::Mat &FeaturesMap);

    protected:
        virtual void predict() override;

        virtual void createTracker(vector<vector<DetBox>> &objects) override;

        virtual bool convertCoor(vector<vector<DetBox>> &objects) override;

        virtual void updateTrackingCue(DetBox &obj, Tracket &tracket) override;
    };
}
#endif // FDSSTTRACKER