#ifndef FEATURE_TRACKER_H
#define FEATURE_TRACKER_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/features2d.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

class orb_tracker
{
public:
    orb_tracker();
    void track_features(cv::Mat current_, cv::Mat prev_,
                        cv::Point2f current_pt_ , cv::Point2f prev_pt_,
                        cv::Rect mask_, bool is_initial_);
    std::vector<int> return_motion_vector();
    std::vector<cv::Rect> return_search_windows();
private:
    int height_ = 50;
    int width_ = 50;
    int alpha_ = 10;
    bool init_ = true;
    bool error_ = false;
    int image_height_;
    int image_width_;

    std::vector<cv::KeyPoint> keypoints_prev_;
    std::vector<cv::KeyPoint> keypoints_current_;
    cv::Mat descriptors_prev_;
    cv::Mat descriptors_current_;
    std::vector<cv::DMatch> matches_;

    std::vector<int> motion_vector_;
    std::vector<cv::Rect> all_windows_;

    cv::Rect get_bounding_box(int x, int y);
    cv::Rect get_search_window(cv::Point2f current_,cv::Point2f prev_);
    std::vector<cv::Rect> extract_orb(cv::Mat current_, cv::Mat prev_,
                                      cv::Rect current_rect_, cv::Rect prev_rect_, cv::Rect mask_);
    // std::vector<int> calculate_geomatric_center(cv::Rect current_rect_, cv::Rect prev_rect_);
    std::vector<int> calculate_geomatric_center(cv::Rect current_rect_, cv::Rect prev_rect_,cv::Rect mask_);
    cv::Point2f get_real_coordinate(cv::Point2f relative_, cv::Point2f absolute_);
    void re_calculate_bb(cv::Rect &current_rect_, cv::Rect mask_);
    void modify_images(cv::Mat &current_, cv::Mat &prev_);
};

#endif
