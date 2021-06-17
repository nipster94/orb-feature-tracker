#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/features2d.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "feature_params.h"

class orb_extractor
{
public:
    orb_extractor() = delete;

    //! Constructor
     orb_extractor(const unsigned int max_num_keypts, const float scale_factor, const unsigned int num_levels,
               const unsigned int ini_fast_thr, const unsigned int min_fast_thr,
               const std::vector<std::vector<float>>& mask_rects = {});
    //! Destructor
    virtual ~ orb_extractor() = default;

    void extract_keypoints(cv::Mat prev_image_, cv::Mat current_image_,
                        std::vector<cv::KeyPoint> &keypoints_prev_, std::vector<cv::KeyPoint> &keypoints_current_,
                        cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                        std::vector<cv::DMatch> &matches);

    void extract_keypoints(cv::Mat current_image_, cv::Mat prev_image_);

    cv::Mat extract_keypoints(cv::Mat initial_image_);

    std::vector<std::vector<cv::KeyPoint>> returnAllKeyPoints();
    std::vector<cv::Mat> returnAllDescriptors();
    std::vector<cv::DMatch> returnMatches();
    std::vector<cv::KeyPoint> returnInitialKeyPoints();
    std::vector<cv::KeyPoint> returnCurrentKeyPoints();

    void compare_performance(cv::Mat current_image_, cv::Mat prev_image_,int val,
                            std::vector<cv::KeyPoint> &keypoints_prev_, 
                            std::vector<cv::KeyPoint> &keypoints_current_,
                            cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                            std::vector<cv::DMatch> &matches);

private:
    unsigned int max_num_keypts_ = 2000;
    float scale_factor_ = 1.2;
    unsigned int num_levels_ = 8;
    unsigned int ini_fast_thr_ = 20;
    unsigned int min_fast_thr = 7;

    //! A vector of keypoint area represents mask area
    //! Each areas are denoted as form of [x_min / cols, x_max / cols, y_min / rows, y_max / rows]
    std::vector<std::vector<float>> mask_rects_;

    //! A list of the scale factor of each pyramid layer
    std::vector<float> scale_factors_;
    std::vector<float> inv_scale_factors_;
    //! A list of the sigma of each pyramid layer
    std::vector<float> level_sigma_sq_;
    std::vector<float> inv_level_sigma_sq_;

    std::vector<cv::KeyPoint> keypoints_prev_;
    std::vector<cv::KeyPoint> keypoints_current_;
    std::vector<cv::KeyPoint> initial_keypoints_;
    cv::Mat descriptors_prev_;
    cv::Mat descriptors_current_;
    std::vector<cv::DMatch> matches_;


    //! rectangle mask has been already initialized or not
    bool mask_is_initialized_ = false;
    cv::Mat rect_mask_;

    //! Maximum number of keypoint of each level
    std::vector<unsigned int> num_keypts_per_level_;
    //! Index limitation that used for calculating of keypoint orientation
    std::vector<int> u_max_;

    void change_intensity(cv::Mat input_image, cv::Mat output_image);


};

#endif
