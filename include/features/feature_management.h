#ifndef FEATURE_MANAGEMENT_H
#define FEATURE_MANAGEMENT_H

#include "ros/ros.h"
#include "ros/package.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <stereo_msgs/DisparityImage.h>
#include <sensor_msgs/image_encodings.h>        //Converting ROS to OpenCV images
#include <cv_bridge/cv_bridge.h>                //Converting ROS to OpenCV images
#include <image_transport/image_transport.h>    //Publishing and subscribing to images in ROS
#include <opencv2/imgproc/imgproc.hpp>          //Converting ROS to OpenCV images
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <chrono>
#include <limits>
#include <numeric>
#include <time.h>
#include <ctime>
#include <opencv2/core/mat.hpp>



#include "orb_extractor.h"
#include "feature_params.h"
#include "orb_tracker.h"


//struct image_details{
//    cv::Mat first_image_;
//    cv::Mat current_image_;
//    cv::Mat prev_image_;
//    std::vector<mask_image> masks_;
//};
using ns = std::chrono::nanoseconds;
using get_time = std::chrono::steady_clock ;

class feature_management
{
public:
    feature_management();
    void execute(void);
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    //Debug value to test single image pair
    bool only_image_comparision_;
    bool init_tracking_;
    bool is_9_mask_;
    bool first_pair_;
    std::string one_time_path_;
    std::string image_seq_path_;
    std::string current_;
    std::string prev_;
    std::string pattern_;
    std::vector<std::string> images_paths_;
    std::vector<int> motion_vector_;

    unsigned int max_num_keypts_ = 2000;
    float scale_factor_ = 1.2;
    unsigned int num_levels_ = 8;
    unsigned int ini_fast_thr_ = 20;
    unsigned int min_fast_thr = 7;

    cv::Mat initial_descriptor_;

    std::vector<cv::Rect> all_current_sw_;
    std::vector<cv::Rect> all_prev_sw_;

//    feature_params* image_feature_details_ = nullptr;
    feature_params image_feature_details_;
    orb_extractor* orb_extractor_ = nullptr;
    orb_tracker* orb_tracker_;

    void extract_onetime_features(std::string base_path_, std::string current_, std::string prev_);
    cv::Mat draw_grid(cv::Mat input_image_);
//    std::vector<mask_image> set_mask_info(cv::Mat image);
    std::vector<cv::Point2f> extrat_points(std::vector<cv::KeyPoint> keypoints_);
    bool sort_keypoints(const cv::Point2f &p1, const cv::Point2f &p2);
    void draw_all_points(cv::Mat input_image_, std::vector<cv::Point2f> points_);

    std::vector<std::string> get_image_paths(std::string directory_path_, std::string pattern_);
    void manage_features(int lower_bound, int upper_bound);
    cv::Mat draw_all_points();
    cv::Mat draw_all_points_center(cv::Mat image_);
    cv::Mat put_text_single(cv::Mat input_);
    std::vector<cv::Mat> put_text(cv::Mat input_);
    void check_feature_lost_percentage(cv::Mat input_);
    void track_features();

    void update_center_points();
    cv::Mat draw_tracked_points(cv::Mat image_);
    cv::Mat draw_search_windows(cv::Mat image_);

    void compare_performance(cv::Mat current_image_, cv::Mat prev_image_, int val,
                            std::vector<cv::KeyPoint> &keypoints_current_,
                            std::vector<cv::KeyPoint> &keypoints_prev_, 
                            cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                            std::vector<cv::DMatch> &matches);

    void change_intensity(cv::Mat input_image, cv::Mat output_image);

    cv::Mat test_point_tracker(cv::Mat input_image);

    void manage_lost_features(int lower_bound, int upper_bound);
 

    //no masks tracker
    std::vector<float> motion_vector_single_mask_;
    void manage_no_masks(int lower_bound, int upper_bound);
    cv::Point track_features_single_mask(cv::Mat current_image_, cv::Mat prev_image_, cv::Point prev_center_);
    void extract_single_mask(cv::Mat current_image_, cv::Mat prev_image_, 
                            std::vector<cv::KeyPoint> &keypoints_current_,
                            std::vector<cv::KeyPoint> &keypoints_prev_, 
                            cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                            std::vector<cv::DMatch> &matches);
    std::vector<float> calculate_geomatric_center(std::vector<cv::DMatch> matches,
                                                std::vector<cv::KeyPoint> keypoints_current_,
                                                std::vector<cv::KeyPoint> keypoints_prev_);
    
    cv::Mat draw_search_window(cv::Mat input, cv::Point current_, cv::Point prev_);


};

#endif
