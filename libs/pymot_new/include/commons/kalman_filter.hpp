#include <iostream>
#include <opencv2/core.hpp>

// abstract class for Kalman filter
// implementation could be KF/EKF/UKF...
class KalmanTrack {
public:
    /**
     * user need to define H matrix & R matrix
     * @param num_states
     * @param num_obs
     */
    // constructor
    explicit KalmanTrack();

    // destructor
    virtual ~KalmanTrack() = default;

    /**
     * Coast state and state covariance using the process model
     * User can use this function without change the internal
     * tracking state x_
     */
    // virtual void Coast();

    /**
     * Predict without measurement update
     */
    void Predict();

    /**
     * This function maps the true state space into the observed space
     * using the observation model
     * User can implement their own method for more complicated models
     */
    virtual cv::Mat PredictionToObservation(const cv::Mat &state);

    /**
     * Updates the state by using Extended Kalman Filter equations
     * @param z The measurement at k+1
     */
    virtual void Update(const cv::Mat &z);

    cv::Rect2f GetStateAsBbox();

    cv::Rect2f ConvertStateToBbox(const cv::Mat &state) const;

    cv::Mat ConvertBboxToObservation(const cv::Rect2f& bbox) const;

    // State vector
    cv::Mat x_, x_predict_, x_post_;

    // Error covariance matrix
    cv::Mat P_, P_predict_, P_post_;

    // State transition matrix
    cv::Mat F_;

    // Covariance matrix of process noise
    cv::Mat Q_;

    // measurement matrix
    cv::Mat H_;

    // covariance matrix of observation noise
    cv::Mat R_;

    unsigned int num_states_, num_obs_;
};