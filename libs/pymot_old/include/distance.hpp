
#ifndef _MOT_DISTANCE_
#define _MOT_DISTANCE_
#include<vector>
#include <cmath>
#include "core.hpp"
#include <algorithm>

namespace mot {

    float l2_distance(std::vector<float> feat1, std::vector<float> feat2);
    /*******************************************************************
     * cal cose distance
     * @param feat1
     * @param feat2
     * @return
     */
    float cos_distance(std::vector<float> feat1, std::vector<float> feat2);

    /********************************************************************
     *
     * @tparam T
     * @param rect1
     * @param rect2
     * @return
     */
    template<typename T>
    T shape_distance(const Rect<T> &rect1, const Rect<T> &rect2) {
        return 1.0 - exp(-(std::abs(rect1.h - rect2.h) / (rect1.h + rect2.h) + std::abs(rect1.w - rect2.w) / (rect1.w + rect2.w)));
    }

    template<typename T>
    float appearance_distance(Rect<T> rect1, Rect<T> rect2);
    /********************************************************************
     * cal motion diatance
     * @tparam T
     * @param rect1
     * @param rect2
     * @return
     */
    template<typename T>
    T motion_distance(const Rect<T> &rect1, const Rect<T> &rect2) {
        float dis_x = ((rect1.x + rect1.w / 2) - (rect2.x + rect2.w / 2)) *
                      ((rect1.x + rect1.w / 2) - (rect2.x + rect2.w / 2));
        float dis_y = ((rect1.y + rect1.h / 2) - (rect2.y + rect2.h / 2)) *
                      ((rect1.y + rect1.h / 2) - (rect2.y + rect2.h / 2));


        return std::sqrt(dis_x + dis_y);
    }


    /******************************************************
     *box1 and box2 iou
     * @tparam T
     * @param box1
     * @param box2
     * @return  iou
     */
    template<typename T>
    T bbox_iou(const Rect<T> &box1,const Rect<T> &box2) {

        T b1_x1 = std::max(box1.x, 0.f);
        T b1_y1 = std::max(box1.y, 0.f);
        T b1_x2 = std::max(box1.x + box1.w,0.f);
        T b1_y2 = std::max(box1.y + box1.h,0.f);

        T b2_x1 = std::max(box2.x,0.f);
        T b2_x2 = std::max(box2.x + box2.w,0.f);
        T b2_y1 = std::max(box2.y,0.f);
        T b2_y2 = std::max(box2.y + box2.h,0.f);

        T inter_rect_x1 = std::max(b1_x1, b2_x1);
        T inter_rect_y1 = std::max(b1_y1, b2_y1);
        T inter_rect_x2 = std::min(b1_x2, b2_x2);
        T inter_rect_y2 = std::min(b1_y2, b2_y2);
        T inter_area;
        inter_area = std::max((inter_rect_x2 - inter_rect_x1),0.f) * std::max(inter_rect_y2 - inter_rect_y1,0.f);
        T b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);
        T b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1);

        T iou = inter_area / (b1_area + b2_area - inter_area + 1e-16);

        return iou;
    }

    /*********************************************************
     * l2 normalize the reid feature
     * @tparam T
     * @param vector
     * @return
     */
    template<typename T>
    std::vector<T> normalize(const std::vector<T> &vector) {
        double mod = 0.0;
        std::vector<T> output(vector.size(), 0);
        for (size_t i = 0; i < vector.size(); ++i) {
            mod += vector[i] * vector[i];
        }

        double mag = std::sqrt(mod);

        if (mag == 0) {
            throw std::logic_error("The input vector is a zero vector");
        }

        for (size_t i = 0; i < vector.size(); ++i) {
            output[i] = vector[i] / mag;
        }

        return output;
    }
    /************************************************************
     *
     * @tparam T
     * @param vector
     */
    template<typename T>
    void normalize(std::vector<T> &vector) {
        if (vector.size() == 0)
            return;
        double mod = 0.0;
        for (size_t i = 0; i < vector.size(); ++i) {
            mod += vector[i] * vector[i];
        }

        double mag = std::sqrt(mod);
        if (mag == 0) {
            throw std::logic_error("The input vector is a zero vector");
        }

        for (size_t i = 0; i < vector.size(); ++i) {
            vector[i] = vector[i] / mag;
        }

    }

    /**************************************************************
     * shrink box1 and box2
     * @tparam T
     * @param box1
     * @param box2
     * @return iou
     */
    template<typename T>
    T bbox_iou_shrink(const Rect<T> &box1,const Rect<T> &box2) {

        T b1_x1 = box1.x;
        T b1_x2 = box1.x + box1.w;
        T b1_y1 = box1.y;
        T b1_y2 = box1.y + box1.h;

        T b2_x1;
        T b2_x2;
        T b2_y1;
        T b2_y2;

        if(box2.h/box2.w >= 2)
        {
            b2_x1 = box2.x;
            b2_x2 = box2.x + box2.w;
            b2_y1 = box2.y;
            b2_y2 = box2.y + box2.h*3/4;
        } else{
            b2_x1 = box2.x + box2.w /4 ;
            b2_x2 = box2.x + box2.w * 3 / 4;
            b2_y1 = box2.y;
            b2_y2 = box2.y + box2.h;
        }

        T inter_rect_x1 = std::max(b1_x1, b2_x1);
        T inter_rect_y1 = std::max(b1_y1, b2_y1);
        T inter_rect_x2 = std::min(b1_x2, b2_x2);
        T inter_rect_y2 = std::min(b1_y2, b2_y2);
        T inter_area;
        inter_area =
                std::max((inter_rect_x2 - inter_rect_x1),0.f) * std::max(inter_rect_y2 - inter_rect_y1,0.f);
        T b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);
        T b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1);

        T iou = inter_area / (b1_area + b2_area - inter_area + 1e-16);

        return iou;
    }
}
#endif
