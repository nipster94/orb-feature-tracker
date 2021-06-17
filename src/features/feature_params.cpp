#include "feature_params.h"

feature_params::feature_params()
{
    initial_run_ = true;
}

void feature_params::initialize(cv::Size image_size, bool is_9_masks){
    std::cout << "in" << '\n';
//    std::vector<mask_image> tmp_ = set_initial_mask_info(image_size);
    masks_ = is_9_masks ? set_initial_9_mask_info(image_size) : set_initial_3_mask_info(image_size);
    std::cout << "out" << '\n';
}

std::vector<mask_image> feature_params::set_initial_9_mask_info(cv::Size image_size){
    std::vector<mask_image> masks_;
    int colFact = image_size.width/3;
    int rowFact = image_size.height/3;
    int count_ = 1;

    std::cout << colFact << '\n';
    std::cout << rowFact << '\n';
    std::cout << image_size.width << '\n';
    std::cout << image_size.height << '\n';

    for (int x = 0; x < colFact*3; x += colFact)
    {
        for (int y = 0; y < image_size.height - 1; y += rowFact)
        {
            mask_image mask_;
            mask_.bounding_box_.x = x;
            mask_.bounding_box_.y = y;
            mask_.bounding_box_.width = colFact;
            mask_.bounding_box_.height = rowFact;
            mask_.got_init_val_ = false;
            mask_.total_distance_ = 0.0;

            switch (count_) {
                case 1 : {
                    mask_.image_location_ = image_location::NorthW;
                    break;
                }
                case 2 : {
                    mask_.image_location_ = image_location::West;
                    break;
                }
                case 3 : {
                    mask_.image_location_ = image_location::SouthW;
                    break;
                }
                case 4 : {
                    mask_.image_location_ = image_location::North;
                    break;
                }
                case 5 : {
                    mask_.image_location_ = image_location::Center;
                    break;
                }
                case 6 : {
                    mask_.image_location_ = image_location::South;
                    break;
                }
                case 7 : {
                    mask_.image_location_ = image_location::NorthE;
                    break;
                }
                case 8 : {
                    mask_.image_location_ = image_location::East;
                    break;
                }
                case 9 : {
                    mask_.image_location_ = image_location::SouthE;
                    break;
                }
//                default:{
//                    break;
//                }
            }

            mask_image tmp_ = mask_;

            std::cout << tmp_.bounding_box_.x << " " << tmp_.bounding_box_.y << " "
                      << tmp_.bounding_box_.width << " "
                      << tmp_.bounding_box_.height  << " "
                      << tmp_.image_location_ << '\n';

            masks_.push_back(mask_);
            count_ +=1;
        }

        std::cout << "Exit" << '\n';
    }

    return masks_;

}

std::vector<mask_image> feature_params::set_initial_3_mask_info(cv::Size image_size){
    std::vector<mask_image> masks_;
    std::vector<int> col_factors_ = get_colfactors(image_size.width,0.5);
    bool init_ = true;
    int count_ = 1;
    int current_col_fact_ = 0;
    for(unsigned long int i = 0; i < col_factors_.size(); ++i ){

        std::cout << "current col fact : " << current_col_fact_ << '\n';
        mask_image mask_;
        mask_.got_init_val_ = false;
        mask_.total_distance_ = 0.0;
        mask_.bounding_box_.width = col_factors_[i];
        mask_.bounding_box_.height = image_size.height;
        mask_.bounding_box_.y = 0;
        if(init_){
            std::cout << "in the init" << '\n';
            mask_.bounding_box_.x = 0;

            init_ = false;
        }
        else {
            current_col_fact_ +=  col_factors_[i-1];
            std::cout << "image width : " << image_size.width
                      << " col fact : " << col_factors_[i-1] << '\n';
            mask_.bounding_box_.x = current_col_fact_;
        }

        switch (count_) {
            case 1 : {
                mask_.image_location_ = image_location::West;
                break;
            }
            case 2 : {
                mask_.image_location_ = image_location::Center;
                break;
            }
            case 3 : {
                mask_.image_location_ = image_location::East;
                break;
            }
        }

        mask_image tmp_ = mask_;

        std::cout << tmp_.bounding_box_.x << " " << tmp_.bounding_box_.y << " "
                  << tmp_.bounding_box_.width << " "
                  << tmp_.bounding_box_.height  << " "
                  << tmp_.image_location_ << '\n';

        masks_.push_back(mask_);
        count_ +=1;

    }

    return masks_;
}

std::vector<int> feature_params::get_colfactors(int cols, double percentage){
    std::vector<int> col_factors_;
    int center_cols_ = std::floor(cols*percentage);
    int left_ = (cols - center_cols_)/2;
    int right_ = (cols - center_cols_)/2;

    col_factors_ = {left_,center_cols_,right_};

    return col_factors_;
}


void feature_params::update_all_masks(std::vector<std::vector<cv::KeyPoint> > key_points_,
                                      std::vector<cv::DMatch> matches_){

    if(!initial_run_) clean_old_keypoints();

    std::vector<cv::KeyPoint> keypoints_prev_ = key_points_[0];
    std::vector<cv::KeyPoint> keypoints_current_ = key_points_[1];

    int count_ = 0;
    std::vector<cv::KeyPoint> all_points_(matches_.size()*2,cv::KeyPoint());

    for(std::vector<cv::DMatch>::iterator it_ = matches_.begin(); it_ != matches_.end();++it_){
        int end_index_ = matches_.size() + count_;
        all_points_.at(count_) = keypoints_current_.at(it_->queryIdx);
        all_points_.at(end_index_) = keypoints_prev_.at(it_->trainIdx);

        count_ += 1;
    }

    insert_keypoints_to_masks(all_points_);
    update_average_distance();

    if(initial_run_){
        update_initial_keypoints();
        initial_run_ = false;
    }

}

std::vector<std::vector<cv::KeyPoint> > feature_params::get_all_keypoints(){
   std::vector<std::vector<cv::KeyPoint> > all_keypoints_;
   std::vector<cv::KeyPoint> initial_;
   std::vector<cv::KeyPoint> current_;
   bool init_ = true;
   for (unsigned long int j = 0; j < masks_.size();++j){
       if (init_){
          initial_ = masks_[j].initial_keypoints_;
          current_ = masks_[j].current_keypoints_;
          init_ = false;
       }
       initial_.insert(initial_.end(),masks_[j].initial_keypoints_.begin(),
                       masks_[j].initial_keypoints_.end());

       current_.insert(current_.begin(), masks_[j].current_keypoints_.begin(),
                       masks_[j].current_keypoints_.end());
   }

   all_keypoints_.push_back(initial_);
   all_keypoints_.push_back(current_);

   return all_keypoints_;
}

void feature_params::insert_keypoints_to_masks(std::vector<cv::KeyPoint> all_key_points){
    for(unsigned long int i = 0;i < all_key_points.size()/2; ++i){
        cv::KeyPoint current_ = all_key_points[i];
        cv::KeyPoint prev_ = all_key_points[all_key_points.size()/2 + i];

        for (unsigned long int j = 0; j < masks_.size();++j){
            if(masks_[j].bounding_box_.contains(current_.pt) &&
               masks_[j].bounding_box_.contains(prev_.pt)){
                masks_[j].current_keypoints_.push_back(current_);
                masks_[j].prev_keypoints_.push_back(prev_);
                masks_[j].total_distance_ += calculate_distances(current_.pt,prev_.pt);
                break;
            }
        }

    }
}

double feature_params::calculate_distances(cv::Point2f current_, cv::Point2f prev_){
    return std::sqrt((current_.x - prev_.x)*(current_.x - prev_.x) + (current_.y - prev_.y)*(current_.y - prev_.y));
}

void feature_params::calculate_center_point(std::vector<cv::KeyPoint> keypoints_, bool is_initial_){
    std::cout << "returning current kp in feature params " << keypoints_.size() << '\n';
    std::cout << "is initial " << is_initial_ << '\n';
    for (unsigned long int j = 0; j < masks_.size();++j){
        cv::Point2f actual_center_ = cv::Point2f(masks_[j].bounding_box_.x + masks_[j].bounding_box_.width/2,
                                                 masks_[j].bounding_box_.y + masks_[j].bounding_box_.height/2);
        double current_distance_ = 10000.0;
        std::cout << "actual center " << actual_center_ << '\n';
        for(std::vector<cv::KeyPoint>::iterator it_ = keypoints_.begin(); it_ != keypoints_.end(); ++it_){
            cv::KeyPoint tmp = *it_;
            cv::Point2f current_ = cv::Point2f(tmp.pt.x,
                                               tmp.pt.y);

            if(masks_[j].bounding_box_.contains(current_)){
                double dist_ = calculate_distances(actual_center_,current_);

                if(dist_ < current_distance_){
                    current_distance_ = dist_;
//                    std::cout << "current : " << current_distance_ << '\n';
                    masks_[j].current_center_kp_ = tmp;
                    if(is_initial_) masks_[j].prev_center_kp_ = tmp;
                }
//                keypoints_.erase(it_);
            }
        }
    }
}

void feature_params::update_average_distance(){
    for (unsigned long int j = 0; j < masks_.size();++j){
        unsigned long int keypoints_size_ = masks_[j].current_keypoints_.size();
        masks_[j].average_distance_ = masks_[j].total_distance_/keypoints_size_;
    }
}

void feature_params::update_initial_keypoints(){
    for (unsigned long int j = 0; j < masks_.size();++j){
        masks_[j].initial_keypoints_ = masks_[j].prev_keypoints_;
        masks_[j].got_init_val_ = true;
    }
}

void feature_params::clean_old_keypoints(){
    for (unsigned long int j = 0; j < masks_.size();++j){
        masks_[j].current_keypoints_.clear();
        masks_[j].prev_keypoints_.clear();
        masks_[j].total_distance_ = 0.0;
        masks_[j].average_distance_ = 0.0;
    }
}






