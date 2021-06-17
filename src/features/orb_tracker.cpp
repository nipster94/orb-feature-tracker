#include "orb_tracker.h"

orb_tracker::orb_tracker(){
    std::cout << "init class orb tracking" << '\n';
    height_ = 50;
    width_ = 50;
    alpha_ = 10;
    init_ = true;
    std::cout << "no error" << '\n';
}

void orb_tracker::track_features(cv::Mat current_, cv::Mat prev_,
                                 cv::Point2f current_pt_, cv::Point2f prev_pt_,
                                 cv::Rect mask_, bool is_initial_){
    std::cout << "in" << '\n';

    image_height_ = current_.rows;
    image_width_ = current_.cols;
    height_ = 50;
    width_ = 50;
    init_ = is_initial_;

//    if(!init_){

//    }

    matches_.clear();
    keypoints_prev_.clear();
    keypoints_current_.clear();
    descriptors_current_ = cv::Mat();
    descriptors_prev_ = cv::Mat();
    motion_vector_ = {0,0};

    std::cout << "after" << '\n';
    cv::Rect search_window_ = get_search_window(current_pt_,prev_pt_);
    std::cout << "got search window" << '\n';
    cv::Rect rec_window_ = get_bounding_box(prev_pt_.x,prev_pt_.y);
    std::cout << "got rec window" << '\n';

//    cv::Mat croppedC = current_(search_window_);
//    cv::Mat croppedR = prev_(rec_window_);
//    cv::Mat croppedMask = current_(mask_);
//    cv::Mat croppedPMask = prev_(mask_);

//    cv::cvtColor(croppedC,croppedC,cv::COLOR_GRAY2RGB);
//    cv::cvtColor(croppedR,croppedR,cv::COLOR_GRAY2RGB);
//    cv::cvtColor(croppedMask,croppedMask,cv::COLOR_GRAY2RGB);
//    cv::cvtColor(croppedPMask,croppedPMask,cv::COLOR_GRAY2RGB);

//    cv::Point2f c_mo_pt_ = cv::Point2f(current_pt_.x - search_window_.x,
//                                       current_pt_.y - search_window_.y);

//    cv::Point2f p_mo_pt_ = cv::Point2f(prev_pt_.x - rec_window_.x,
//                                       prev_pt_.y - rec_window_.y);

//    cv::Point2f c_m_pt_ = cv::Point2f(current_pt_.x + mask_.x,
//                                      current_pt_.y - mask_.y);

//    cv::Point2f p_m_pt_ = cv::Point2f(prev_pt_.x - mask_.x,
//                                      prev_pt_.y - mask_.y);

//    cv::circle(croppedC,c_mo_pt_ ,5,cv::Scalar(0,0,255),cv::FILLED,cv::LINE_8);
//    cv::circle(croppedR,p_mo_pt_,5,cv::Scalar(0,0,255),cv::FILLED,cv::LINE_8);
//    cv::circle(croppedMask,c_m_pt_,5,cv::Scalar(0,0,255),cv::FILLED,cv::LINE_8);
//    cv::circle(croppedPMask,p_m_pt_,5,cv::Scalar(0,0,255),cv::FILLED,cv::LINE_8);

//    cv::imshow("init current", croppedC);
//    cv::imshow("init prev", croppedR);
//    cv::imshow("inti current mask", croppedMask);
//    cv::imshow("init prev mask", croppedPMask);
//    cv::waitKey(0);

    all_windows_ = extract_orb(current_,prev_,search_window_,rec_window_,mask_);
    // motion_vector_ = calculate_geomatric_center(search_window_,rec_window_);
    motion_vector_ = calculate_geomatric_center(search_window_,rec_window_,mask_);
}

cv::Rect orb_tracker::get_bounding_box(int x, int y){
    int x_ = x - width_/2;
    x_ = x_ < 0 ? 0 : x_;
    x_ = x_ > image_width_ ? image_width_ : x_;
    int y_ = y - height_/2;
    y_ = y_ < 0 ? 0 : y_;
    y_ = y_ > image_height_ ? image_height_ : y_;
    return cv::Rect(x_,y_,width_,height_);
}

cv::Rect orb_tracker::get_search_window(cv::Point2f current_, cv::Point2f prev_){
    std::cout << "current : " << current_ << '\n';
    std::cout << "prev : " << prev_ << '\n';
//    std::cout << init_ << '\n';
    cv::Point2f tmp_ = current_ - prev_;
//    std::cout << "diff : " << tmp_ << '\n';


    if(init_){
//        init_ = false;
        std::cout << "in init tracking " << '\n';
    //    int tmp_x_ = (int)(current_.x - prev_.x);
    //    int tmp_y_ = (int)(current_.y - prev_.y);
    //    std::vector<int> motion_vec_ = {tmp_x_, tmp_y_};
    //    std::cout << motion_vec_[0] << '\n';
    //    std::cout << motion_vec_[1] << '\n';

    //    motion_vector_ = {tmp_x_, tmp_y_};


    //    motion_vector_ = motion_vec_;

        // For the initial case we need to expand the search window
        // since we dont know whih direction its gonna move
        height_ += 2*alpha_;
        width_ += 2*alpha_;
    }



//    int dx_ = (int)(current_.x - prev_.x);
//    int dy_ = (int)(current_.y - prev_.y);

//    motion_vector_.push_back(tmp_.x);
//    motion_vector_.push_back(tmp_.y);
    motion_vector_ = {static_cast<int>(tmp_.x),static_cast<int>(tmp_.y)};

    std::cout << motion_vector_[0] << '\n';
    std::cout << motion_vector_[1] << '\n';

    height_ += alpha_;
    width_ += alpha_;

    int x_ = current_.x + motion_vector_[0] - width_/2;
    x_ = x_ < 0 ? 0 : x_;
    x_ = x_ > image_width_ ? image_width_ : x_;
    int y_ = current_.y + motion_vector_[1] - height_/2;
    y_ = y_ < 0 ? 0 : y_;
    y_ = y_ > image_height_ ? image_height_ : y_;

    std::cout << "x : " << x_ << " y : " << y_
              << " end x : " << x_ + width_ << " end y : " << y_ + width_
              <<" width: " << width_ << " height : " << height_  << '\n';
    std::cout <<"image width: " << image_width_ << " height : " << image_height_ << '\n';

    return cv::Rect(x_, y_,width_,height_);
}

std::vector<cv::Rect> orb_tracker::extract_orb(cv::Mat current_, cv::Mat prev_,
                              cv::Rect current_rect_, cv::Rect prev_rect_,
                              cv::Rect mask_){
    std::cout << "in extract orb" << '\n';
    error_ = false;
    if((current_rect_.x + current_rect_.width) > image_width_) current_rect_.width = image_width_ -  current_rect_.x;
    if((current_rect_.y + current_rect_.height) > image_height_) current_rect_.height = image_height_ - current_rect_.y;
    if((prev_rect_.x + prev_rect_.width) > image_width_) prev_rect_.width = image_width_ -  prev_rect_.x;
    if((prev_rect_.y + prev_rect_.height) > image_height_) prev_rect_.height = image_height_ - prev_rect_.y;

    std::cout << current_rect_ << '\n';
    std::cout << prev_rect_ << '\n';

    // std::cout << "Before" << '\n';
    // std::cout << keypoints_current_.size() << '\n';
    // std::cout << descriptors_current_.size << '\n';
    // std::cout << keypoints_prev_.size() << '\n';
    // std::cout << descriptors_prev_.size << '\n';

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(2000,1.4,8);
    // cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect( current_(current_rect_), keypoints_current_ );
    detector->detect( prev_(prev_rect_),keypoints_prev_ );

    descriptor->compute( current_(current_rect_),keypoints_current_ , descriptors_current_);
    descriptor->compute( prev_(prev_rect_), keypoints_prev_ , descriptors_prev_ );

    std::cout << "After" << '\n';
    std::cout << keypoints_current_.size() << '\n';
    std::cout << descriptors_current_.size << '\n';
    std::cout << keypoints_prev_.size() << '\n';
    std::cout << descriptors_prev_.size << '\n';

   if(descriptors_prev_.size == 0 || descriptors_current_.size == 0 ||
           keypoints_prev_.size() == 0 || keypoints_current_.size() == 0){

        std::cout << "ERROR" << '\n';
        error_ = true;
        re_calculate_bb(current_rect_,mask_);
        re_calculate_bb(prev_rect_,mask_);
        std::cout << current_rect_ << '\n';
        std::cout << prev_rect_ << '\n';

        detector->detect( current_(mask_), keypoints_current_ );
        detector->detect( prev_(mask_),keypoints_prev_ );

        descriptor->compute( current_(mask_),keypoints_current_ , descriptors_current_);
        descriptor->compute( prev_(mask_), keypoints_prev_ , descriptors_prev_ );

        std::cout << "After ERROR RECT" << '\n';
        std::cout << keypoints_current_.size() << '\n';
        std::cout << descriptors_current_.size << '\n';
        std::cout << keypoints_prev_.size() << '\n';
        std::cout << descriptors_prev_.size << '\n';

        matcher->match ( descriptors_current_,descriptors_prev_, matches_ );
        std::cout << matches_.size() << '\n';
        // std::vector<cv::KeyPoint> updated_kp_current_, updated_kp_prev_;
        // std::vector<cv::DMatch> updated_matches_;
        // if(matches_.size() != 0){
        //     for(std::vector<cv::DMatch>::iterator it_ = matches_.begin(); it_ != matches_.end();++it_){
        //         // cv::Point2f current_ = get_real_coordinate(keypoints_current_[it_->queryIdx].pt,
        //         //                                             cv::Point(current_rect_.x,current_rect_.y));
        //         // cv::Point2f prev_ = get_real_coordinate(keypoints_prev_[it_->trainIdx].pt,
        //         //                                         cv::Point2f(prev_rect_.x,prev_rect_.y));
        //         cv::Point2f current_ = get_real_coordinate(keypoints_current_[it_->queryIdx].pt,
        //                                                 cv::Point(mask_.x,mask_.y));
        //         cv::Point2f prev_ = get_real_coordinate(keypoints_prev_[it_->trainIdx].pt,
        //                                                 cv::Point2f(mask_.x,mask_.y));

        //         if(current_rect_.contains(current_) && prev_rect_.contains(prev_)){
        //             updated_kp_current_.push_back(keypoints_current_[it_->queryIdx]);
        //             updated_kp_prev_.push_back(keypoints_prev_[it_->trainIdx]);
        //             updated_matches_.push_back(*it_);
        //         }
        //     }
        // }
        
        // if(updated_kp_current_.size() != 0 && updated_kp_prev_.size() != 0){
        //     keypoints_current_ = updated_kp_current_;
        //     keypoints_prev_ = updated_kp_prev_;
        //     std::cout << "After re-match whit full mask" << '\n';
        //     std::cout << keypoints_current_.size() << '\n';
        //     std::cout << descriptors_current_.size << '\n';
        //     std::cout << keypoints_prev_.size() << '\n';
        //     std::cout << descriptors_prev_.size << '\n';
        //     std::cout << "updated matches : " << updated_matches_.size() << '\n';
        //     matches_ = updated_matches_;
        // }
        // else
        // {
        //     std::cout << "ERROR AGAIN" << '\n';
        // }
    }
    else
    {
        std::cout << "NO ERROR" << '\n';
        matcher->match ( descriptors_current_,descriptors_prev_, matches_ );
        std::cout << matches_.size() << '\n';
    }

    //Matching point pair screening
    double min_dist=10000, max_dist=0;

    //Find the minimum and maximum distances between all matches, that is,
    //the distance between the most similar and least similar two sets of points
    for ( int i = 0; i < descriptors_current_.rows; i++ )
    {
        double dist = matches_[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //When the distance between the descriptors is greater than twice the minimum distance,
    //the match is considered wrong. But sometimes the minimum distance will be very small,
    //so set an empirical value of 30 as the lower limit.
    std::vector< cv::DMatch > good_matches;
    for ( int i = 0; i < descriptors_current_.rows; i++ )
    {
        if ( matches_[i].distance <= cv::max ( 2*min_dist, 40.0 ) )
        {
            good_matches.push_back ( matches_[i] );
        }
    }

    std::cout << "good matches size " << good_matches.size() << '\n';
    matches_ = good_matches;

    std::vector<cv::Rect> all_wins_ = {current_rect_,prev_rect_};

    return all_wins_;

//    std::cout << "values defined" << '\n';


//    if(descriptors_prev_.size == 0 || descriptors_current_.size == 0 ||
//            keypoints_prev_.size() == 0 || keypoints_current_.size() == 0){
//        std::cout << "ERROR" << '\n';

//        cv::imshow("current", current_(current_rect_));
//        cv::imshow("prev", prev_(prev_rect_));
//        cv::imshow("new current", current_(mask_));
//        cv::imshow("new prev", prev_(mask_));
//        cv::waitKey(0);
//        std::cout << "check with the full image" << '\n';
//        detector->detectAndCompute(current_,current_(mask_),keypoints_current_,descriptors_current_);
//        std::cout << "detect current" << '\n';
//        detector->detectAndCompute(prev_,prev_(mask_),keypoints_prev_,descriptors_prev_);
////        detector->detectAndCompute(current_,current_,keypoints_current_,descriptors_current_);
////        std::cout << "detect current" << '\n';
////        detector->detectAndCompute(prev_,prev_,keypoints_prev_,descriptors_prev_);
//        std::cout << "detect prev" << '\n';
//    }

//    if(descriptors_prev_.size == 0 || descriptors_current_.size == 0 ||
//            keypoints_prev_.size() == 0 || keypoints_current_.size() == 0){
//        std::cout << "ERROR AGAIN" << '\n';
//        detector->detectAndCompute(current_,current_,keypoints_current_,descriptors_current_);
//        std::cout << "detect current" << '\n';
//        detector->detectAndCompute(prev_,prev_,keypoints_prev_,descriptors_prev_);

//    }

//    detector->detectAndCompute(current_,current_(current_rect_),keypoints_current_,descriptors_current_);
//    std::cout << "detect current" << '\n';
//    detector->detectAndCompute(prev_,prev_(prev_rect_),keypoints_prev_,descriptors_prev_);
//    std::cout << "detect prev" << '\n';

//    while (descriptors_prev_.size == 0 || descriptors_current_.size == 0 ||
//           keypoints_prev_.size() == 0 || keypoints_current_.size() == 0) {
//        std::cout << "ERROR RE DETECT" << '\n';

//        re_calculate_bb(current_rect_,mask_);
//        re_calculate_bb(prev_rect_,mask_);
//        std::cout << current_rect_ << '\n';
//        std::cout << prev_rect_ << '\n';

//        int c_size_ = current_rect_.width*current_rect_.height;
//        int mask_size_ = mask_.width*mask_.height;

//        if(c_size_ >= mask_size_*0.5){
//            modify_images(current_,prev_);
//        }


        // detector->detectAndCompute(current_,current_(mask_),keypoints_current_,descriptors_current_);
        // detector->detectAndCompute(prev_,prev_(mask_),keypoints_prev_,descriptors_prev_);

//        std::cout << keypoints_current_.size() << '\n';
//        std::cout << descriptors_current_.size << '\n';
//        std::cout << keypoints_prev_.size() << '\n';
//        std::cout << descriptors_prev_.size << '\n';

//        cv::imshow("current", current_(current_rect_));
//        cv::imshow("prev", prev_(prev_rect_));
////        cv::imshow("new current", current_(mask_));
////        cv::imshow("new prev", prev_(mask_));
//        cv::waitKey(0);




//    }

//    if(descriptors_prev_.size == 0 || descriptors_current_.size == 0 ||
//        keypoints_prev_.size() == 0 || keypoints_current_.size() == 0){

//    }


//    std::cout << "ERROR RE DETECT" << '\n';

//    //Detect in the whole mask
//    detector->detect ( current_(mask_),keypoints_current_ );
//    detector->detect ( prev_(mask_),keypoints_prev_ );

//    //If still no keypoints, the ignore this motion vector
//    if(keypoints_prev_.size() == 0 || keypoints_current_.size() == 0){
//         std::cout << "SILL NO KEYPOINTS" << '\n';
//    }
//    else {
//        int min_allowed_c_kp_ = (int)keypoints_current_.size()*0.10;
//        int min_allowed_p_kp_ = (int)keypoints_prev_.size()*0.10;

//        std::vector<cv::KeyPoint> updated_kp_1_, updated_kp_2_;

//        for(unsigned long int i = 0; i < keypoints_current_.size(); ++i){
//            cv::Point2f point_ = cv::Point2f(keypoints_current_[i].pt.x + mask_.x,
//                                             keypoints_current_[i].pt.y + mask_.y);
//            if(current_rect_.contains(point_)){
//              updated_kp_1_.push_back(keypoints_current_[i]);
//            }
//        }

//        for(unsigned long int i = 0; i < keypoints_prev_.size(); ++i){
//            cv::Point2f point_ = cv::Point2f(keypoints_prev_[i].pt.x + mask_.x,
//                                             keypoints_prev_[i].pt.y + mask_.y);
//            if(prev_rect_.contains(point_)){
//              updated_kp_2_.push_back(keypoints_prev_[i]);
//            }
//        }

//        while ((int)updated_kp_1_.size() < min_allowed_c_kp_ || (int)updated_kp_2_.size() < min_allowed_p_kp_) {
//            std::cout << "NO KEY POINTS IN THE SEARCH WINDOW" << '\n';
//            updated_kp_1_.clear();
//            updated_kp_2_.clear();
//            re_calculate_bb(current_rect_,mask_);
//            re_calculate_bb(prev_rect_,mask_);

//            for(unsigned long int i = 0; i < keypoints_current_.size(); ++i){
//                cv::Point2f point_ = cv::Point2f(keypoints_current_[i].pt.x + mask_.x,
//                                                 keypoints_current_[i].pt.y + mask_.y);
//                if(current_rect_.contains(point_)){
//                  updated_kp_1_.push_back(keypoints_current_[i]);
//                }
////                  else {
////                     std::cout << keypoints_current_[i].pt << '\n';
////                  }
//            }

//            for(unsigned long int i = 0; i < keypoints_prev_.size(); ++i){
//                cv::Point2f point_ = cv::Point2f(keypoints_prev_[i].pt.x + mask_.x,
//                                                 keypoints_prev_[i].pt.y + mask_.y);
//                if(prev_rect_.contains(point_)){
//                  updated_kp_2_.push_back(keypoints_prev_[i]);
//                }
//            }

//            std::cout << "---------------------------------" <<  '\n';
//            std::cout << current_rect_<< '\n';
//            std::cout << prev_rect_ << '\n';
//            std::cout << mask_ << '\n';
//            std::cout << keypoints_current_.size()<< '\n';
//            std::cout << keypoints_prev_.size()<< '\n';
//            std::cout << updated_kp_1_.size()<< '\n';
//            std::cout << updated_kp_2_.size()<< '\n';
//            std::cout << "---------------------------------" <<  '\n';
////                cv::imshow("current", current_(current_rect_));
////                cv::imshow("prev", prev_(prev_rect_));

//            cv::Mat outimg1;
//            cv::Mat outimg2;
//            cv::Mat outimg3;
//            cv::Mat outimg4;

//            cv::drawKeypoints( current_(mask_), keypoints_current_, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//            cv::drawKeypoints( prev_(mask_), keypoints_prev_, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//            cv::drawKeypoints( current_(mask_),  updated_kp_1_ , outimg3, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//            cv::drawKeypoints( prev_(mask_),  updated_kp_2_ , outimg4, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

//            cv::Point2f strat_ = cv::Point2f(current_rect_.x, current_rect_.y);
//            cv::Point2f end_ = cv::Point2f(current_rect_.x + current_rect_.width,
//                                           current_rect_.y + current_rect_.height);
//            cv::rectangle(outimg1,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);
//            cv::rectangle(outimg3,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);

//            strat_ = cv::Point2f(prev_rect_.x, prev_rect_.y);
//            end_ = cv::Point2f(prev_rect_.x + prev_rect_.width,
//                               prev_rect_.y + prev_rect_.height);
//            cv::rectangle(outimg2,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);
//            cv::rectangle(outimg4,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);

//            cv::imshow("original c",outimg1);
//            cv::imshow("original p",outimg2);
//            cv::imshow("updated c",outimg3);
//            cv::imshow("updated p",outimg4);

//    //        cv::imshow("new current", current_(mask_));
//    //        cv::imshow("new prev", prev_(mask_));
//            cv::waitKey(0);

//        }

//        descriptor->compute ( current_(mask_), updated_kp_1_, descriptors_current_ );
//        descriptor->compute ( prev_(mask_),  updated_kp_2_, descriptors_prev_ );

//        keypoints_current_.clear();
//        keypoints_prev_.clear();

//        keypoints_current_ = updated_kp_1_;
//        keypoints_prev_ = updated_kp_2_;
//    }






//    cv::Mat outimg1;
//    cv::Mat outimg2;
//    cv::Mat outimg3;
//    cv::Mat outimg4;

//    cv::drawKeypoints( current_(mask_), keypoints_current_, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//    cv::drawKeypoints( prev_(mask_), keypoints_prev_, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//    cv::drawKeypoints( current_(mask_),  keypoints_current_ , outimg3, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//    cv::drawKeypoints( prev_(mask_),  keypoints_prev_ , outimg4, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

//    cv::Point2f strat_ = cv::Point2f(current_rect_.x, current_rect_.y);
//    cv::Point2f end_ = cv::Point2f(current_rect_.x + current_rect_.width,
//                                   current_rect_.y + current_rect_.height);
//    cv::rectangle(outimg1,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);
//    cv::rectangle(outimg3,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);

//    strat_ = cv::Point2f(prev_rect_.x, prev_rect_.y);
//    end_ = cv::Point2f(prev_rect_.x + prev_rect_.width,
//                       prev_rect_.y + prev_rect_.height);
//    cv::rectangle(outimg2,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);
//    cv::rectangle(outimg4,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);

//    cv::imshow("original c",outimg1);
//    cv::imshow("original p",outimg2);
//    cv::imshow("updated c",outimg3);
//    cv::imshow("updated p",outimg4);

//    cv::imshow("current", current_(current_rect_));
//    cv::imshow("prev", prev_(prev_rect_));
//    cv::imshow("new current", current_(mask_));
//    cv::imshow("new prev", prev_(mask_));
//    cv::waitKey(0);
}

void orb_tracker::re_calculate_bb(cv::Rect &current_rect_, cv::Rect mask_){
    current_rect_.x -= 10;
    current_rect_.y -= 10;
    current_rect_.width += 20;
    current_rect_.height += 20;

    current_rect_.x = current_rect_.x < 0 ? 0 : current_rect_.x;
    current_rect_.y = current_rect_.y < 0 ? 0 : current_rect_.y;
    if((current_rect_.x + current_rect_.width) > image_width_) current_rect_.width = image_width_ -  current_rect_.x;
    if((current_rect_.y + current_rect_.height) > image_height_) current_rect_.height = image_height_ - current_rect_.y;

    cv::Point2f start_ = cv::Point2f(current_rect_.x,current_rect_.y);
    cv::Point2f end_ = cv::Point2f(current_rect_.x + current_rect_.width,
                                   current_rect_.y + current_rect_.height);

    if(!mask_.contains(start_))
    {
        std::cout << "start point out of the mask" << '\n';
        if (start_.x < mask_.x){
            start_.x  = mask_.x;
        }
        else if (start_.x > mask_.x + mask_.width) {
            start_.x = mask_.x + mask_.width;
        }

        if(start_.y < mask_.y){
            start_.y = mask_.y;
        }
        else if (start_.y > mask_.y + mask_.height) {
            start_.y = mask_.y + mask_.height;
        }
    }
    if(!mask_.contains(end_))
    {
        std::cout << "end point out of the mask" << '\n';
        if (end_.x < mask_.x){
            end_.x  = mask_.x;
        }
        else if (end_.x > mask_.x + mask_.width) {
            end_.x = mask_.x + mask_.width;
        }

        if(end_.y < end_.y){
            end_.y = end_.y;
        }
        else if (end_.y > mask_.y + mask_.height) {
            end_.y = mask_.y + mask_.height;
        }
    }

    current_rect_.x = start_.x;
    current_rect_.y = start_.y;
    current_rect_.width = end_.x - start_.x;
    current_rect_.height = end_.y - start_.y;
}

void orb_tracker::modify_images(cv::Mat &current_, cv::Mat &prev_){
    for(int i = 0; i < current_.rows; ++i){
        for(int j = 0; j < current_.cols; ++j){
            current_.at<uchar>(i,j) = current_.at<uchar>(i,j) < 30 ? 150 : current_.at<uchar>(i,j);
            current_.at<uchar>(i,j) = current_.at<uchar>(i,j) > 230 ? 150 : current_.at<uchar>(i,j);
        }
    }

    for(int i = 0; i < prev_.rows; ++i){
        for(int j = 0; j < prev_.cols; ++j){
            prev_.at<uchar>(i,j) = prev_.at<uchar>(i,j) < 30 ? 150 : prev_.at<uchar>(i,j);
            prev_.at<uchar>(i,j) = prev_.at<uchar>(i,j) > 230 ? 150 : prev_.at<uchar>(i,j);
        }
    }
}


std::vector<int> orb_tracker::calculate_geomatric_center(cv::Rect current_rect_, 
                                                         cv::Rect prev_rect_,
                                                         cv::Rect mask_){
    // error_ = true;                                                         
    int C_cx_ = 0, C_cy_ = 0, C_px_ = 0, C_py_ = 0;
    std::vector<int> motion_vec_ = {0,0};
    cv::Point absolute_ = error_ ? cv::Point(mask_.x,mask_.y) 
                            : cv::Point(current_rect_.x,current_rect_.y);

    std::cout << "matches size " << matches_.size() << '\n';
    std::cout << mask_ << '\n';
    if(matches_.size() != 0){
        for(std::vector<cv::DMatch>::iterator it_ = matches_.begin(); it_ != matches_.end();++it_){
        //    cv::Point2f current_ = get_real_coordinate(keypoints_current_[it_->queryIdx].pt,
        //                                               cv::Point(current_rect_.x,current_rect_.y));
        //    cv::Point2f prev_ = get_real_coordinate(keypoints_prev_[it_->trainIdx].pt,
        //                                            cv::Point2f(prev_rect_.x,prev_rect_.y));
            // cv::Point2f current_ = get_real_coordinate(keypoints_current_[it_->queryIdx].pt,
            //                                           absolute_);
            // cv::Point2f prev_ = get_real_coordinate(keypoints_prev_[it_->trainIdx].pt,
            //                                        absolute_);
            // cv::Point2f current_ = keypoints_current_[it_->queryIdx].pt;
            // cv::Point2f prev_ = keypoints_prev_[it_->trainIdx].pt;
            cv::Point2f current_ = keypoints_current_.at(it_->queryIdx).pt;
            cv::Point2f prev_ = keypoints_prev_.at(it_->trainIdx).pt;

            C_cx_ += current_.x;
            C_cy_ += current_.y;
            C_px_ += prev_.x;
            C_py_ += prev_.y;

            // if(current_rect_.contains(current_) && prev_rect_.contains(prev_)){
            //     C_cx_ += current_.x;
            //     C_cy_ += current_.y;
            //     C_px_ += prev_.x;
            //     C_py_ += prev_.y;
            // }
            // else
            // {
            //     std::cout  << "current : " << current_ << " prev : " << prev_ << '\n';
            // }
            
        }

        C_cx_ /= matches_.size();
        C_cy_ /= matches_.size();
        C_px_ /= matches_.size();
        C_py_ /= matches_.size();

        motion_vec_ = {C_cx_ - C_px_, C_cy_- C_py_};


        if(motion_vec_[0] > 100 || motion_vec_[1] > 100){

            std::cout << keypoints_current_.size()<< '\n';
            std::cout << keypoints_prev_.size()<< '\n';
            std::cout << matches_.size()<< '\n';


            C_cx_ = 0, C_cy_ = 0, C_px_ = 0, C_py_ = 0;
            for(std::vector<cv::DMatch>::iterator it_ = matches_.begin(); it_ != matches_.end();++it_){
                cv::Point2f current_ = get_real_coordinate(keypoints_current_[it_->queryIdx].pt,
                                                           cv::Point(current_rect_.x,current_rect_.y));
                cv::Point2f prev_ = get_real_coordinate(keypoints_current_[it_->trainIdx].pt,
                                                        cv::Point2f(prev_rect_.x,prev_rect_.y));

                cv::Point2f current_new_ = get_real_coordinate(keypoints_current_[it_->trainIdx].pt,
                                                           cv::Point(current_rect_.x,current_rect_.y));
                cv::Point2f prev_new_ = get_real_coordinate(keypoints_current_[it_->queryIdx].pt,
                                                        cv::Point2f(prev_rect_.x,prev_rect_.y));

                C_cx_ += current_.x;
                C_cy_ += current_.y;
                C_px_ += prev_.x;
                C_py_ += prev_.y;

                std::cout << "current : " << current_ << " prev : " << prev_ << '\n';
                std::cout << "current : " << current_new_ << " prev : " << prev_new_ << '\n';
                std::cout << " C_cx_ : " << C_cx_ << " C_cy_ : " << C_cy_ << " C_px_ : "
                          << C_px_ << " C_py_ : " << C_py_ << '\n';

                std::cout << "--------------------------------------------------------------------" << '\n';


            }

            C_cx_ /= matches_.size();
            C_cy_ /= matches_.size();
            C_px_ /= matches_.size();
            C_py_ /= matches_.size();

            std::cout  << "C_cx_ : " <<C_cx_ << " C_cy_ : " << C_cy_
                       << " C_px_ : "  << C_px_ << " C_py_ : " << C_py_ << '\n';

            std::cout << matches_.size() << '\n';

            motion_vec_ = {0, 0};
        }


    }
    else {
        std::cout << "no matches found" << '\n';
    }

    return motion_vec_;
}

cv::Point2f orb_tracker::get_real_coordinate(cv::Point2f relative_, cv::Point2f absolute_){
    return cv::Point2f(relative_.x + absolute_.x,
                       relative_.y + absolute_.y);
}

std::vector<int> orb_tracker::return_motion_vector(){
    return motion_vector_;
}

std::vector<cv::Rect> orb_tracker::return_search_windows(){
    std::cout << "all window" << '\n';
    std::cout << all_windows_[0] << '\n';
    std::cout << all_windows_[1] << '\n';
    return all_windows_;
}

