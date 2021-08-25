
#ifndef _MOT_CORE_
#define _MOT_CORE_
#include <iostream>
#include <cstddef>
#include <vector>
// #include "log.hpp"
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
        PredictType predict_type;               //预测的方法,目前只支持0
        Velocity velocity;                   //目标平均速度
        Status status;                     //当前tracker 状态
        int det_corre_index;            //当前tracket关联的检测框索引
        int confidence;
        int label;                      //目标类别
        int update;                     //用来标记当前帧是否已经更新
        int valid;                      //用于标记当前框是否有较
        std::vector <Point2D<float> > trajectory;
        int confirm;
    };
}
#endif

