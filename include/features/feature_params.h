#ifndef FEATURE_PARAMS_H
#define FEATURE_PARAMS_H

#include <opencv2/imgproc/imgproc.hpp>          //Converting ROS to OpenCV images
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/features2d.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

enum image_location{
    Center,
    North,East,South,West,
    NorthE,SouthE,SouthW,NorthW
};

struct mask_image{
//    mask_image() = default;

//    mask_image(image_location image_location_,
//               bool got_init_val_,
//               cv::KeyPoint intial_kp_, std::vector<cv::KeyPoint> keypoints_,
//               cv::Point2f start_pt_, int height_, int width_);

    image_location image_location_;
    bool got_init_val_;
    cv::KeyPoint current_center_kp_;
    cv::KeyPoint prev_center_kp_;
    double total_distance_;
    double average_distance_;
    std::vector<int> motion_vector_;
    std::vector<cv::KeyPoint> initial_keypoints_;
    std::vector<cv::KeyPoint> current_keypoints_;
    std::vector<cv::KeyPoint> prev_keypoints_;
    std::vector<cv::Point2f> all_tracked_centers_;
    cv::Rect bounding_box_;
//    cv::Rect current_search_win_;
//    cv::Rect prev_search_win_;
//    cv::Point2f start_pt_;
//    int height_;
//    int width_;

};

class feature_params
{
public:
    feature_params();
    cv::Mat first_image_;
    cv::Mat current_image_;
    cv::Mat prev_image_;
    std::vector<mask_image> masks_;

    void initialize(cv::Size image_size, bool is_9_masks);
    void update_all_masks(std::vector<std::vector<cv::KeyPoint>> key_points_,
                          std::vector<cv::DMatch> matches_);

    std::vector<std::vector<cv::KeyPoint>> get_all_keypoints();
    void calculate_center_point(std::vector<cv::KeyPoint> keypoints_, bool is_initial_ = true);

private:
    bool initial_run_;
    std::vector<mask_image> set_initial_9_mask_info(cv::Size image_size);
    std::vector<mask_image> set_initial_3_mask_info(cv::Size image_size);
    std::vector<int> get_colfactors(int cols, double percentage);
    void insert_keypoints_to_masks(std::vector<cv::KeyPoint> all_key_points);
    double calculate_distances(cv::Point2f current_, cv::Point2f prev_);

    void update_average_distance();
    void update_initial_keypoints();
    void clean_old_keypoints();
};

#endif
