#ifndef _MOSSE_TRACKER_
#define _MOSSE_TRACKER_
#include "trackers/base_tracker.hpp"

namespace mot{
    class MosseTracker : public MotTracker
    {
    public:
        MosseTracker(const mot::MotConfig &config) : MotTracker(config) {}

    private:
        void train(const cv::Mat& image, Tracket &tracket);

        cv::Mat imcrop(cv::Rect roi, const cv::Mat& image);

        cv::Mat createGaussKernel(cv::Size sz, float sigma, cv::Point center);

        cv::Mat fft(cv::Mat image, bool backwards = false);

        cv::Mat conj(const cv::Mat& image);

        cv::Mat preprocess(const cv::Mat& image);

        cv::Mat createHanningMats(int rows, int cols);

        cv::Mat rand_warp(const cv::Mat& image);

        cv::Mat convert(const cv::Mat& src);

        cv::Mat real(cv::Mat image);

        cv::Mat imag(cv::Mat image);

        cv::Mat complexDivision(cv::Mat a, cv::Mat b);

        cv::Mat complexMultiplication(cv::Mat a, cv::Mat b);

        
    protected:
        virtual void predict() override;

        virtual void createTracker(vector<vector<DetBox>> &objects) override;

        virtual bool convertCoor(vector<vector<DetBox>> &objects) override;

        virtual void updateTrackingCue(DetBox &obj, Tracket &tracket) override;
    };
}
#endif