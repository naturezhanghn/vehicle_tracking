
#ifndef _MOT_CORE_
#define _MOT_CORE_
#include <iostream>
#include <cstddef>
#include <vector>
#include "common/log.hpp"
#include <opencv2/opencv.hpp>
namespace mot {

    const int SUCESS = 1;
    const int FAIL = 0;
    const int USED = 1;
    const int UN_USED = 0;

    template<typename T>
    struct Point2D {
        Point2D(T x, T y) : x(x), y(y) {};
        T x;
        T y;
    };

    template<typename T>
    struct Rect {
        T x;
        T y;
        T w;
        T h;
    };

    template<typename T>
    struct Feature {
        std::vector <T> feat;
        int dim[4];
    };
    
    enum Status {
        TRACKET_LOST = 0,
        TRACKET_INIT,
        TRACKET_KEEP,
        TRACKET_DEAD
    };

    enum PredictType {
        FLOW = 0,
        LINEAR,
        KALMAN
    };

    enum ClassType {
        PERSON = 0,
        NON_MOTOR,
        CAR,
        TRICYCLE,
        MOTOCYCLE
    };


    struct DetBox {
        //size_t id;
        Rect<float> rect;
        int confidence;
        int label;
        int status;  //用来标记是否已经关联上跟踪框
        Feature<float> feat;                       //reid特征
        Feature<float> flow;                       //光流特征
        //int det_corre_index;
    };

    struct TracketBox {
        size_t id;
        Rect<float> rect;
        int confidence;
        int label;
        int status;  //用来标记是否已经关联上跟踪框
        int det_corre_index;
        int valid;
        std::vector <Point2D<float> > trajectory;
    };


    struct Velocity {
        float x;
        float y;
    };

    struct Tracket {
        size_t id;
        int time_stamp;                 //时间戳，用来计算跟踪寿命
        int init_time;                  //跟踪最少帧数计数
        Rect<float> rect;                       //当前跟踪的检测框
        Feature<float> feat;                       //reid特征
        Feature<float> flow;                       //光流特征
        Rect<float> predict_rect;               //预测的跟踪框
        Status status;                     //当前tracker 状态
        int det_corre_index;            //当前tracket关联的检测框索引
        int confidence;
        int label;                      //目标类别
        int update;                     //用来标记当前帧是否已经更新
        int valid;                      //用于标记当前框是否有较
        std::vector <Point2D<float> > trajectory;
        int confirm;
        struct Linear_param
        {
            PredictType predict_type;               //预测的方法,目前只支持0
            Velocity velocity;                   //目标平均速度
        }linear_param;
        
        struct Mosse_param
        {
            float _sigma;
            float _eta;
            cv::Mat guassKernelMatrix;
            cv::Mat gauss_fft;
            cv::Mat Ai;
            cv::Mat Bi;
            cv::Mat Hi;
            cv::Mat fi;
            cv::Mat fi_fft;			
            cv::Size init_sz;
        }mosse_param;

        struct fDsst_param
        {
            bool hog = true;
            bool fixed_window = false;
            bool multiscale = true;
            float padding = 2.5; // extra area surrounding the target
            float output_sigma_factor = 0.125; // bandwidth of gaussion target
            float lambda = 0.0001; // regularization

            float interp_factor; // linear interpolation factor for adaptation
            float sigma; // guassion kernel bandwidth
            int cell_size; // HOG cell size
            int num_compressed_dim;

            int size_patch[3];
            cv::Mat hann;
            cv::Size _tmpl_sz;
            float _scale;
            int _gaussion_size;
            
            cv::Mat s_hann;
            cv::Mat ysf;
            
            int template_size; // template size

            int base_width; // initial ROI width
            int base_height; // initial ROI height
            int scale_max_area; // max ROI size before compressing
            float scale_padding; // extra area surrounding the target for scaling
            float scale_step; // scale step for multi-scale estimation
            float scale_sigma_factor; // bandwidth of gaussian target

            int n_scales; // scaling windows
            int n_interp_scales; //interpolation scales

            float scale_lr; //scale learning rate

            std::vector<float> scaleFactors; // all scale changing rate, from larger to smaller with 1 to be the middle
            std::vector<float> interp_scaleFactors;

            int scale_model_width; // the model width for scaling
            int scale_model_height; // the model height for scaling
            float currentScaleFactor; // scaling rate
            float min_scale_factor; // min scale rate
            float max_scale_factor; // max_scaling rate
            float scale_lambda; // regularization

            cv::Mat _labCentroids;
            cv::Mat _alphaf;
            cv::Mat _prob;
            cv::Mat _tmpl;

            cv::Mat _proj_tmpl;

            cv::Mat _num;
            cv::Mat _den;

            cv::Mat sf_den;
            cv::Mat sf_num;

            cv::Mat proj_matrix;
        }fdsst_param;
        
    };
}
#endif

