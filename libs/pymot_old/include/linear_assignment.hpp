//
// Created by xujianhua on 2020/5/9.
//

#ifndef MOT_DEMO_LINEAR_ASSIGNMENT_HPP
#define MOT_DEMO_LINEAR_ASSIGNMENT_HPP

#include "matrix.hpp"
#include "tracker.hpp"
#include "distance.hpp"


namespace mot {


    template<class T>
    class LinearAssignment {
    public:

        LinearAssignment(Matrix<T> &cost, const std::vector<Tracket> &trackets, const std::vector<DetBox> &objects, const std::vector<int> &tracket_reindex,
                const std::vector<int> &det_reindex, const double &ad_thres,const double &remove_thres,const double &localThres,const bool &dynathres);

        void find_tracket_matchs(Matrix<T> &matrix,std::vector<int> &matck_tracket,int row,int col){
            matck_tracket = std::vector<int>(row,-1);
            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < col; ++j) {
                    if (matrix(i,j) == 0)
                    {
                        matck_tracket[i] = j;
                        break;
                    }
                }
            }
        }

        void SetAmplificationValue(double addValue){this->ADDVALUE = addValue;}

        void MinCostMatching(Matrix <T> &cost);

        void MatchingCascade(int level);

        void ConstructMatrix();

        std::vector<int> GetResult(){return assign_results;}

        void Amplification();

        void RemoveDetByThres();

    private:
        double ad_thres,remove_thres,cls_local_thres;
        bool dynathres;
        std::vector<int> assign_results,assign_results_track;
        std::vector<int> tracket_reindex,det_reindex;
        std::vector<int> current_det_reindex,current_trakcet_reindex;//获取当前的检测框和跟踪框
        std::vector<int> hash_det_origin, hash_tracket_origin;//跟踪当前的有效框的原始映射
        std::vector<Tracket> trackets;
        std::vector<DetBox> objects;
        double ADDVALUE = 1e3;
        Matrix<T> matrix;//多级匈牙利匹配的临时矩阵
        Matrix<T> cost; //多级匈牙利匹配前的原始矩阵
    };
}
#endif //MOT_DEMO_LINEAR_ASSIGNMENT_HPP
