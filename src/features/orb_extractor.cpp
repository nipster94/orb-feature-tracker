#include "orb_extractor.h"

orb_extractor::orb_extractor(const unsigned int max_num_keypts, const float scale_factor, const unsigned int num_levels,
                           const unsigned int ini_fast_thr, const unsigned int min_fast_thr,
                           const std::vector<std::vector<float>>& mask_rects)
        : max_num_keypts_(max_num_keypts), scale_factor_(scale_factor), num_levels_(num_levels),
          ini_fast_thr_(ini_fast_thr), min_fast_thr(min_fast_thr),
          mask_rects_(mask_rects)
{
//        keypoints_current_ = *new std::vector<cv::KeyPoint>;
//        keypoints_prev_ = *new std::vector<cv::KeyPoint>;
//        descriptors_current_ = *new cv::Mat;
//        descriptors_prev_ = *new cv::Mat;
//        matches_ = *new std::vector<cv::DMatch>;

        for (const auto& v : mask_rects_) {
            if (v.size() != 4) {
                throw std::runtime_error("mask rectangle must contain four parameters");
            }
            if (v.at(0) >= v.at(1)) {
                throw std::runtime_error("x_max must be greater than x_min");
            }
            if (v.at(2) >= v.at(3)) {
                throw std::runtime_error("y_max must be greater than x_min");
            }
        }
}

void orb_extractor::extract_keypoints(cv::Mat prev_image_, cv::Mat current_image_,
                                      std::vector<cv::KeyPoint> &keypoints_prev_, std::vector<cv::KeyPoint> &keypoints_current_,
                                      cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                                      std::vector<cv::DMatch> &matches){

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num_keypts_,scale_factor_,num_levels_);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
//    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect ( current_image_, keypoints_current_ );
    detector->detect ( prev_image_,keypoints_prev_ );

    descriptor->compute ( current_image_,keypoints_current_ , descriptors_1 );
    descriptor->compute ( prev_image_, keypoints_prev_ , descriptors_2 );

    matcher->match ( descriptors_1, descriptors_2, matches );
}

void orb_extractor::extract_keypoints(cv::Mat current_image_, cv::Mat prev_image_){

//    if(current_image_.channels() == 3){
//        cv::cvtColor(current_image_,current_image_,cv::COL)
//    }


    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num_keypts_,scale_factor_,num_levels_);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

//    std::cout << "in" << '\n';

//    detector->detect ( image_feature_details.current_image_, keypoints_current_ );
//    detector->detect ( image_feature_details.prev_image_, keypoints_prev_ );

//    descriptor->compute ( image_feature_details.current_image_ ,keypoints_current_ , descriptors_current_ );
//    descriptor->compute ( image_feature_details.prev_image_ , keypoints_prev_ , descriptors_prev_ );

    detector->detect ( current_image_, keypoints_current_ );
    detector->detect ( prev_image_, keypoints_prev_ );

    descriptor->compute ( current_image_ ,keypoints_current_ , descriptors_current_ );
    descriptor->compute ( prev_image_ , keypoints_prev_ , descriptors_prev_ );

    matcher->match ( descriptors_current_,descriptors_prev_, matches_ );
}

cv::Mat orb_extractor::extract_keypoints(cv::Mat initial_image_){
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num_keypts_,scale_factor_,num_levels_);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    cv::Mat initial_descriptor_;

    detector->detect ( initial_image_, initial_keypoints_ );
    descriptor->compute ( initial_image_ ,initial_keypoints_ , initial_descriptor_ );
//    std::cout << initial_descriptor_.rows << '\n';
//    std::cout << initial_descriptor_.cols << '\n';
//    cv::imshow("initial descriptor",initial_descriptor_);
//    cv::waitKey(0);

    return initial_descriptor_;
}

void orb_extractor::compare_performance(cv::Mat current_image_, cv::Mat prev_image_, int val,
                                        std::vector<cv::KeyPoint> &keypoints_prev_, 
                                        std::vector<cv::KeyPoint> &keypoints_current_,
                                        cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                                        std::vector<cv::DMatch> &matches){
                                            
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num_keypts_,scale_factor_,num_levels_);
    // cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    cv::Mat tmp_current_ = current_image_.clone();
    change_intensity(current_image_,tmp_current_);

    cv::imshow("current",current_image_);
    cv::imshow("tmpcurrent",tmp_current_);
    cv::waitKey(0);


    detector->detect ( current_image_, keypoints_current_ );
    detector->detect ( prev_image_,keypoints_prev_ );

    descriptor->compute ( current_image_,keypoints_current_ , descriptors_1 );
    descriptor->compute ( prev_image_, keypoints_prev_ , descriptors_2 );

    matcher->match ( descriptors_1, descriptors_2, matches );

}


void orb_extractor::change_intensity(cv::Mat input_image, cv::Mat output_image){
    cv::Scalar img_mean = cv::mean(input_image);
    double mean_ = img_mean[0];
    std::cout << "initial mean : " << mean_ << '\n';
    double minVal, maxVal;
    cv::minMaxLoc(input_image, &minVal, &maxVal);
    //take 10% of the mean and increase
    double adjusted_intensity_ = mean_*0.1;
    double intensity_factor_ = 255/(maxVal + adjusted_intensity_);
    for (int i = 0; i < output_image.rows; i++){
        for (int j = 0; j < output_image.cols; j++){
            output_image.at<uchar>(i,j) = (output_image.at<uchar>(i,j) + adjusted_intensity_)*intensity_factor_;
        }
    }
    
    cv::Scalar img_mean_output_ = cv::mean(output_image);
    double mean_after_ = img_mean_output_[0];
    std::cout << "mean after : " << mean_after_ << '\n';

}


std::vector<std::vector<cv::KeyPoint>> orb_extractor::returnAllKeyPoints(){
    std::vector<std::vector<cv::KeyPoint>> allKeyPoints = {keypoints_prev_,keypoints_current_};
    return allKeyPoints;
}

std::vector<cv::Mat> orb_extractor::returnAllDescriptors(){
    std::vector<cv::Mat> allDescriptors = {descriptors_prev_,descriptors_current_};
    return allDescriptors;
}

std::vector<cv::DMatch> orb_extractor::returnMatches(){
    return matches_;
}

std::vector<cv::KeyPoint> orb_extractor::returnInitialKeyPoints(){
    std::cout << initial_keypoints_.size() << '\n';
    return initial_keypoints_;
}

std::vector<cv::KeyPoint> orb_extractor::returnCurrentKeyPoints(){
    std::cout << "returning current kp " << keypoints_current_.size() << '\n';
    return keypoints_current_;
}







