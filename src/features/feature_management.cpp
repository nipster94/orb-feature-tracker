#include "feature_management.h"

feature_management::feature_management() :
    it_(nh_)
{
    only_image_comparision_ = false;
    init_tracking_ = true;
    is_9_mask_ = false;

    if(only_image_comparision_){
//        one_time_path_ = "/home/nipun/MPSYS/Thesis/Semcon/git_ws/src/lundb_ws/src/sensor_packs/"
//                         "object_detection/resource/17_17_29_new_image_set/image_0/";
        one_time_path_ = "/home/nipun/lund_ws/src/lundb_ws/src/sensor_packs/object_detection/resource/17_17_29_new_image_set/image_0/";
        current_ = "001040.png";
        prev_ = "001040.png";
        // orb_extractor_ = new orb_extractor(max_num_keypts_,scale_factor_,num_levels_,ini_fast_thr_,min_fast_thr);
    }
    else {
        orb_extractor_ = new orb_extractor(max_num_keypts_,scale_factor_,num_levels_,ini_fast_thr_,min_fast_thr);
        orb_tracker_ = new orb_tracker();
        image_seq_path_ = "/home/nipun/dataset/05/image_0/";
//        image_seq_path_ = "/home/nipun/MPSYS/Thesis/Semcon/git_ws/src/lundb_ws/src/sensor_packs/"
//                          "object_detection/resource/17_17_29_new_image_set/image_0/";
//        current_ = "000408.png";
//        prev_ = "000412.png";
        pattern_ = "*.png";

    }
}

void feature_management::execute(){
    std::cout << "in execute" << '\n';
    bool debug = false;
    if (only_image_comparision_){
        extract_onetime_features(one_time_path_,current_,prev_);
    }
    else if (debug) {
        images_paths_ = get_image_paths(image_seq_path_,pattern_);
        std::cout << "done reading" << '\n';
        cv::Mat current_image_ = cv::imread(images_paths_[950],CV_LOAD_IMAGE_GRAYSCALE);
        image_feature_details_.initialize(current_image_.size(),is_9_mask_);
    }
    else {
        images_paths_ = get_image_paths(image_seq_path_,pattern_);
        std::cout << "done reading" << '\n';
//        manage_features(950,960);
        // manage_features(407,415);
        // manage_lost_features(405,415);
        manage_no_masks(784,792);
    }
}

void feature_management::extract_onetime_features(std::string base_path_,
                                                  std::string current_, std::string prev_){

    bool is_debug_rect_ = false;

    cv::Mat current_image_ = cv::imread(base_path_ + current_,CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat prev_image_ = cv::imread(base_path_ + prev_,CV_LOAD_IMAGE_GRAYSCALE);
    std::cout << "images are ready" << '\n';
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches;

    compare_performance(current_image_,prev_image_,1,
    keypoints_1,keypoints_2,descriptors_1,descriptors_2,matches);

    // orb_extractor->compare_performance(current_image_,prev_image_,1,
    // keypoints_1,keypoints_2,descriptors_1,descriptors_2,matches);

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

    // detector->detect ( current_image_,keypoints_1 );
    // detector->detect ( prev_image_,keypoints_2 );

    // descriptor->compute ( current_image_, keypoints_1, descriptors_1 );
    // descriptor->compute ( prev_image_, keypoints_2, descriptors_2 );

    if(keypoints_1.size() == 0 || keypoints_2.size() == 0){
        std::cout << "NO KEYPOINTS" << '\n';
    }
    else
    {
        std::cout << keypoints_1.size() << '\n';
        std::cout << keypoints_2.size() << '\n';
    }  

    std::vector<cv::KeyPoint> updated_kp_1_, updated_kp_2_;
    
    if(is_debug_rect_){
        cv::Point2f cc_ = cv::Point2f(current_image_.cols/2, current_image_.rows/2);
        cv::Point2f pc_ = cv::Point2f(prev_image_.cols/2, prev_image_.rows/2);
        cv::Rect current_rec_  = cv::Rect(cc_.x - 50,cc_.y - 50, 100,100);
        cv::Rect prev_rec_ = cv::Rect(pc_.x - 50, pc_.y - 50, 100,100);

        

        for(unsigned long int i = 0; i < keypoints_1.size(); ++i){
          if(current_rec_.contains(keypoints_1[i].pt)){
              updated_kp_1_.push_back(keypoints_1[i]);
          }
        }

        for(unsigned long int i = 0; i < keypoints_2.size(); ++i){
          if(prev_rec_.contains(keypoints_2[i].pt)){
              updated_kp_2_.push_back(keypoints_2[i]);
          }
        }

        //create new descriptors
        cv::Mat updated_descriptors_1, updated_descriptors_2;

        if(updated_kp_1_.size() == 0 || updated_kp_2_.size() == 0){
            std::cout << "NO KEYPOINTS" << '\n';
        }

        descriptor->compute ( current_image_, updated_kp_1_, updated_descriptors_1 );
        descriptor->compute ( prev_image_, updated_kp_2_, updated_descriptors_2 );


        std::cout << current_rec_<< '\n';
        std::cout << prev_rec_ << '\n';
        std::cout << keypoints_1.size()<< '\n';
        std::cout << keypoints_2.size()<< '\n';
        std::cout << updated_kp_1_.size()<< '\n';
        std::cout << updated_kp_2_.size()<< '\n';
        std::cout << descriptors_1.size << '\n';
        std::cout << descriptors_2.size << '\n';
        std::cout << updated_descriptors_1.size << '\n';
        std::cout << updated_descriptors_2.size << '\n';


        //Match the BRIEF descriptors in the two images, using Hamming distance
        std::vector<cv::DMatch> updated_matches;
        //BFMatcher matcher ( NORM_HAMMING );
        matcher->match ( updated_descriptors_1, updated_descriptors_2, updated_matches);

        cv::Mat outimg1;
        cv::Mat outimg2;
        cv::Mat outimg3;
        cv::Mat outimg4;

        cv::drawKeypoints( current_image_, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
        cv::drawKeypoints( prev_image_, keypoints_2, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
        cv::drawKeypoints( current_image_,  updated_kp_1_ , outimg3, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
        cv::drawKeypoints( prev_image_,  updated_kp_2_ , outimg4, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

        cv::Point2f strat_ = cv::Point2f(current_rec_.x, current_rec_.y);
        cv::Point2f end_ = cv::Point2f(current_rec_.x + current_rec_.width,
                                       current_rec_.y + current_rec_.height);
        cv::rectangle(outimg1,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);
        cv::rectangle(outimg3,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);

        strat_ = cv::Point2f(prev_rec_.x, prev_rec_.y);
        end_ = cv::Point2f(prev_rec_.x + prev_rec_.width,
                           prev_rec_.y + prev_rec_.height);
        cv::rectangle(outimg2,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);
        cv::rectangle(outimg4,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);

        cv::imshow("original c",outimg1);
        cv::imshow("original p",outimg2);
        cv::imshow("updated c",outimg3);
        cv::imshow("updated p",outimg4);

        cv::Mat updated_image_;
        cv::drawMatches (current_image_ , updated_kp_1_, prev_image_ , updated_kp_2_ , updated_matches, updated_image_);
        cv::imshow("UPDATED",updated_image_);
        cv::waitKey(0);
    }
    else
    {
        cv::Mat outimg1;
        cv::drawKeypoints( current_image_, keypoints_1, outimg1, 
                            cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
        cv::imshow("ORB",outimg1);
        // cv::waitKey(0);
        //Matching point pair screening
        double min_dist=10000, max_dist=0;

        //Find the minimum and maximum distances between all matches, that is,
        //the distance between the most similar and least similar two sets of points
        for ( int i = 0; i < descriptors_1.rows; i++ )
        {
            double dist = matches[i].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }

        printf ( "-- Max dist : %f \n", max_dist );
        printf ( "-- Min dist : %f \n", min_dist );

        //When the distance between the descriptors is greater than twice the minimum distance,
        //the match is considered wrong. But sometimes the minimum distance will be very small,
        //so set an empirical value of 30 as the lower limit.
        std::vector< cv::DMatch > good_matches,bad_matches;
        for ( int i = 0; i < descriptors_1.rows; i++ )
        {
            if ( matches[i].distance <= cv::max ( 2*min_dist, 20.0 ) )
            {
                good_matches.push_back ( matches[i] );
            }
            else
            {
                bad_matches.push_back(matches[i]);
            }
            
        }

        std::vector<cv::Point2f> all_points_(good_matches.size()*2,cv::Point2f(0,0));
        //    all_points_.reserve(matches.size()*2);
        // std::cout << all_points_.size() << '\n';
        int count_ = 0;
        for(std::vector<cv::DMatch>::iterator it_ = good_matches.begin(); it_ != good_matches.end();++it_){
            // std::cout<< it_->imgIdx << " "
            //             << it_->distance << " "
            //             << it_->queryIdx << " "
            //             << it_->trainIdx << " "
            //             << keypoints_1.at(it_->queryIdx).pt << " "  //current
            //             << keypoints_2.at(it_->trainIdx).pt << '\n';  //prev
                        // << keypoints_2.at(it_->queryIdx).pt << " "  //current
                        // << keypoints_1.at(it_->trainIdx).pt << '\n';  //prev

            int end_index_ = good_matches.size() + count_;
            // std::cout << "index :" << count_  << " end index : " << end_index_ << '\n';

            all_points_.at(count_) = keypoints_1.at(it_->queryIdx).pt;
            all_points_.at(end_index_) = keypoints_2.at(it_->trainIdx).pt;
            // all_points_.at(count_) = keypoints_1.at(it_->trainIdx).pt;
            // all_points_.at(end_index_) = keypoints_2.at(it_->queryIdx).pt;
            
            // updated_kp_1_.push_back(keypoints_1.at(it_->trainIdx));
            // updated_kp_2_.push_back(keypoints_2.at(it_->queryIdx));

            updated_kp_1_.push_back(keypoints_1.at(it_->queryIdx));
            updated_kp_2_.push_back(keypoints_2.at(it_->trainIdx));

            count_ += 1;

        }

        float percentage = (float)good_matches.size()/matches.size();

        std::cout << "all : " << matches.size() << '\n';
        std::cout << "good : " << good_matches.size()<< '\n';
        std::cout << "bad : " << bad_matches.size()<< '\n';
        std::cout << updated_kp_1_.size() << '\n';
        std::cout << updated_kp_2_.size()<< '\n';
        std::cout << percentage << '\n';

        //draw matching results
        cv::Mat img_match;
        cv::Mat img_goodmatch;
        cv::Mat img_badmatch;
        cv::drawMatches (current_image_ , keypoints_1,
                         prev_image_ , keypoints_2, matches, img_match,
                         cv::Scalar::all(-1), cv::Scalar::all(-1),
                         std::vector<char>(),cv::DrawMatchesFlags::DEFAULT );
        cv::drawMatches (current_image_, keypoints_1, 
                        prev_image_, keypoints_2, good_matches, img_goodmatch,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::DEFAULT );
        cv::drawMatches (current_image_, keypoints_1, 
                        prev_image_, keypoints_2, bad_matches, img_badmatch,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::DEFAULT );

        draw_all_points(prev_image_,all_points_);
        // good_matches.clear();
        // cv::imshow ( "All matching points", img_match );
        // cv::imshow ( "Match point pairs after optimization", img_goodmatch );
        // cv::imshow ( "Bad matches", img_badmatch );
        // cv::imshow("grid",outut_grid_);
        cv::imshow("all points", prev_image_);
        cv::imshow("current",current_image_);
        cv::waitKey(0);
    }
    



//    cv::Mat outimg1;
//    cv::drawKeypoints( current_image_, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//    cv::imshow("ORB",outimg1);

////    std::vector<mask_image> tmp_masks_ = set_mask_info(current_image_);

//    //Match the BRIEF descriptors in the two images, using Hamming distance
//    std::vector<cv::DMatch> matches;
//    //BFMatcher matcher ( NORM_HAMMING );
//    matcher->match ( descriptors_1, descriptors_2, matches );

//    //Matching point pair screening
//    double min_dist=10000, max_dist=0;

//    //Find the minimum and maximum distances between all matches, that is,
//    //the distance between the most similar and least similar two sets of points
//    for ( int i = 0; i < descriptors_1.rows; i++ )
//    {
//        double dist = matches[i].distance;
//        if ( dist < min_dist ) min_dist = dist;
//        if ( dist > max_dist ) max_dist = dist;
//    }

//    printf ( "-- Max dist : %f \n", max_dist );
//    printf ( "-- Min dist : %f \n", min_dist );

//    //When the distance between the descriptors is greater than twice the minimum distance,
//    //the match is considered wrong. But sometimes the minimum distance will be very small,
//    //so set an empirical value of 30 as the lower limit.
//    std::vector< cv::DMatch > good_matches;
//    for ( int i = 0; i < descriptors_1.rows; i++ )
//    {
//        if ( matches[i].distance <= cv::max ( 2*min_dist, 30.0 ) )
//        {
//            good_matches.push_back ( matches[i] );
//        }
//    }

//    std::vector<cv::Point2f> points_ = extrat_points(keypoints_1);
//    std::sort(points_.begin(),points_.end(),
//              [](const cv::Point2f &a, const cv::Point2f &b)
//               {
//                   return a.x < b.x;
//               });

////    std::sort(points_.begin(),points_.end(),
////              [](const cv::Point2f &a, const cv::Point2f &b)
////               {
////                   return a.y < b.y;
////               });

//    for(std::vector<cv::Point2f>::iterator it_ = points_.begin();it_ != points_.end();++it_){
//        std::cout << *it_ << '\n';
//    }

//    std::vector<cv::Point2f> all_points_(good_matches.size()*2,cv::Point2f(0,0));
////    all_points_.reserve(matches.size()*2);
//    std::cout << all_points_.size() << '\n';
//    int count_ = 0;
//    for(std::vector<cv::DMatch>::iterator it_ = good_matches.begin(); it_ != good_matches.end();++it_){
//        std::cout<< it_->imgIdx << " "
//                 << it_->distance << " "
//                 << it_->queryIdx << " "
//                 << it_->trainIdx << " "
//                 << keypoints_1.at(it_->queryIdx).pt << " "  //current
//                 << keypoints_2.at(it_->trainIdx).pt << '\n';  //prev

//        int end_index_ = good_matches.size() + count_;
//        std::cout << "index :" << count_  << " end index : " << end_index_ << '\n';
//        all_points_.at(count_) = keypoints_1.at(it_->queryIdx).pt;
//        all_points_.at(end_index_) = keypoints_2.at(it_->trainIdx).pt;

//        count_ += 1;

//    }

//    draw_all_points(prev_image_,all_points_);

////    for(std::vector<mask_image>::iterator it_ = tmp_masks_.begin(); it_ != tmp_masks_.end(); it_++){
////        mask_image tmp_ = *it_;
////        std::cout << tmp_.image_location_  << " "
////                  << tmp_.start_pt_ << " "
////                  << tmp_.height_ << " "
////                  << tmp_.width_ << '\n';
////    }


//    cv::Mat outut_grid_ = draw_grid(current_image_);
//    std::cout << matches.size() << '\n';
//    //    std::cout << tmp_masks_.size() << '\n';
//        std::cout << good_matches.size()<< '\n';

//    //draw matching results
//    cv::Mat img_match;
//    cv::Mat img_goodmatch;
//    cv::drawMatches (current_image_ , keypoints_1,prev_image_ , keypoints_2, matches, img_match );
//    cv::drawMatches (current_image_, keypoints_1, prev_image_, keypoints_2, good_matches, img_goodmatch );
////    cv::imshow ( "All matching points", img_match );
////    cv::imshow ( "Match point pairs after optimization", img_goodmatch );
//    cv::imshow("grid",outut_grid_);
//    cv::imshow("all points", prev_image_);
//    cv::imshow("current",current_image_);
//    cv::waitKey(0);
}

void feature_management::compare_performance(cv::Mat current_image_, cv::Mat prev_image_, int val,
                                            std::vector<cv::KeyPoint> &keypoints_current_,
                                            std::vector<cv::KeyPoint> &keypoints_prev_, 
                                            cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                                            std::vector<cv::DMatch> &matches){
    auto start = get_time::now();                                        
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num_keypts_,scale_factor_,num_levels_);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    // cv::Ptr<cv::DescriptorExtractor> descriptor = cv::xfeatures2d::SURF::create();
    // cv::Ptr<cv::DescriptorExtractor> descriptor = cv::xfeatures2d::SIFT::create();
    // cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
     cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create("BruteForce");

    cv::Mat tmp_current_ = current_image_.clone();
    change_intensity(current_image_,tmp_current_);

    cv::imshow("current",current_image_);
    cv::imshow("tmpcurrent",tmp_current_);
    // cv::imwrite("/home/nipun/fov_intensity_current.png",current_image_);
    cv::imwrite("/home/nipun/fov_intensity_n30.png",tmp_current_);

    cv::waitKey(0);

    // detector->detect ( current_image_, keypoints_current_ );
    detector->detect ( tmp_current_, keypoints_current_ );
    detector->detect ( prev_image_,keypoints_prev_ );
    auto detect = get_time::now();
    // descriptor->compute ( current_image_,keypoints_current_ , descriptors_1 );
    descriptor->compute ( tmp_current_,keypoints_current_ , descriptors_1 );
    descriptor->compute ( prev_image_, keypoints_prev_ , descriptors_2 );
    auto descript = get_time::now();
    matcher->match ( descriptors_1, descriptors_2, matches );

    auto end = get_time::now();
    auto diff = end - start;
    auto detect_time = detect - start;
    auto discript_time = descript - detect;
    auto match_time = end - descript;
    std::cout <<" Elapsed time is :  "<< std::chrono::duration_cast<ns>(diff).count()/1000000<<" ms "<< std::endl;
    std::cout <<" detector time is :  "<< std::chrono::duration_cast<ns>(detect_time).count()/1000000<<" ms "<< std::endl;
    std::cout <<" descriptor time is :  "<< std::chrono::duration_cast<ns>(discript_time).count()/1000000<<" ms "<< std::endl;
    std::cout <<" matcher time is :  "<< std::chrono::duration_cast<ns>(match_time).count()/1000000<<" ms "<< std::endl;

}

void feature_management::change_intensity(cv::Mat input_image, cv::Mat output_image){
    cv::Scalar img_mean = cv::mean(input_image);
    double mean_ = img_mean[0];
    std::cout << "initial mean : " << mean_ << '\n';
    double minVal, maxVal;
    cv::minMaxLoc(input_image, &minVal, &maxVal);
    //take 10% of the mean and increase
    double adjusted_intensity_ = mean_*0.3;
    double intensity_factor_ = 255/(maxVal - adjusted_intensity_);
    // double intensity_factor_ = 255/(maxVal + adjusted_intensity_);
    for (int i = 0; i < output_image.rows; i++){
        for (int j = 0; j < output_image.cols; j++){
            double new_intensity_val_ = (output_image.at<uchar>(i,j) - adjusted_intensity_) > 0 ? (output_image.at<uchar>(i,j) - adjusted_intensity_) 
                                        : -(output_image.at<uchar>(i,j) - adjusted_intensity_);   

            // output_image.at<uchar>(i,j) = (output_image.at<uchar>(i,j) + adjusted_intensity_)*intensity_factor_;
            output_image.at<uchar>(i,j) = new_intensity_val_*intensity_factor_;
        }
    }
    
    cv::Scalar img_mean_output_ = cv::mean(output_image);
    double mean_after_ = img_mean_output_[0];
    std::cout << "mean after : " << mean_after_ << '\n';

}


cv::Mat feature_management::draw_grid(cv::Mat input_image_){
    std::vector<cv::Mat> blocks;
    cv::Mat maskImg = input_image_.clone();
    if(is_9_mask_){
        int colFact = input_image_.cols/3;
        int rowFact = input_image_.rows/3;
        for (int y = 0; y < input_image_.cols; y += colFact)
        {
            for (int x = 0; x < input_image_.rows; x += rowFact)
            {
                cv::rectangle(maskImg, cv::Point(y, x),
                              cv::Point(y + colFact - 1, x + rowFact - 1),
                              CV_RGB(255, 0, 0), 1);
            }
        }
    }
    else{
        for(unsigned long int i = 0 ; i < image_feature_details_.masks_.size(); ++i){
            cv::Point2f start_ = cv::Point2f(image_feature_details_.masks_[i].bounding_box_.x,
                                             image_feature_details_.masks_[i].bounding_box_.y);
            cv::Point2f end_ = cv::Point2f(start_.x + image_feature_details_.masks_[i].bounding_box_.width,
                                           start_.y + image_feature_details_.masks_[i].bounding_box_.height);
            cv::rectangle(maskImg,
                          start_,
                          end_,
                          CV_RGB(255, 0, 0), 3);
        }

    }

    return maskImg;

}

//std::vector<mask_image> feature_management::set_mask_info(cv::Mat input_image_){
//    std::vector<mask_image> masks_;
//    int colFact = input_image_.cols/3;
//    int rowFact = input_image_.rows/3;
//    int count_ = 1;
//    for (int y = 0; y < input_image_.cols; y += colFact)
//    {
//        for (int x = 0; x < input_image_.rows; x += rowFact)
//        {
//            mask_image mask_;
//            mask_.start_pt_ = cv::Point2f(x,y);
//            mask_.height_ = rowFact;
//            mask_.width_ = colFact;

//            switch (count_) {
//                case 1 : {
//                    mask_.image_location_ = image_location::NorthW;
//                    break;
//                }
//                case 2 : {
//                    mask_.image_location_ = image_location::West;
//                    break;
//                }
//                case 3 : {
//                    mask_.image_location_ = image_location::SouthW;
//                    break;
//                }
//                case 4 : {
//                    mask_.image_location_ = image_location::North;
//                    break;
//                }
//                case 5 : {
//                    mask_.image_location_ = image_location::Center;
//                    break;
//                }
//                case 6 : {
//                    mask_.image_location_ = image_location::South;
//                    break;
//                }
//                case 7 : {
//                    mask_.image_location_ = image_location::NorthE;
//                    break;
//                }
//                case 8 : {
//                    mask_.image_location_ = image_location::East;
//                    break;
//                }
//                case 9 : {
//                    mask_.image_location_ = image_location::SouthE;
//                    break;
//                }
////                default:{
////                    break;
////                }
//            }

////            mask_image tmp_ = mask_;

////            std::cout << tmp_.start_pt_ << " "
////                      << tmp_.height_ << " "
////                      << tmp_.width_ << '\n';

//            masks_.push_back(mask_);
//            count_ +=1;
//        }

////        std::cout << "Exit" << '\n';
//    }

//    return masks_;

//}




std::vector<cv::Point2f> feature_management::extrat_points(std::vector<cv::KeyPoint> keypoints_){
   std::vector<cv::Point2f> points_;
   std::vector<cv::KeyPoint>::iterator iter_;

   for(iter_ = keypoints_.begin(); iter_ != keypoints_.end(); ++iter_){
       cv::KeyPoint tmp_ = *iter_;
       std::cout << tmp_.pt  << " "
                 << tmp_.size << " "
                 << tmp_.angle << " "
                 << tmp_.class_id << '\n';

       points_.push_back(tmp_.pt);
   }

   return points_;
}

bool feature_management::sort_keypoints(const cv::Point2f &p1, const cv::Point2f &p2){
    return ((p1.x + p1.y) < (p2.x + p2.y));
}

void feature_management::draw_all_points(cv::Mat input_image_, std::vector<cv::Point2f> points_){
    for(unsigned int i = 0; i < points_.size(); ++i){
        if(i < points_.size()/2){
            cv::circle(input_image_, points_[i], 1, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0);
        }
        else {
            cv::circle(input_image_, points_[i], 1, cv::Scalar(0, 255, 0), 1, cv::LINE_8, 0);
        }
    }
}

std::vector<std::string> feature_management::get_image_paths(std::__cxx11::string directory_path_, std::__cxx11::string pattern_){
    std::vector<std::string> image_paths_;
    std::vector<cv::String> fn;
    pattern_ = directory_path_ + "/"+ pattern_;
    glob(pattern_, fn, false);
    size_t count = fn.size();
    for(size_t i=0; i<count; i++){
        image_paths_.push_back(fn[i]);
    }

    return image_paths_;
}


void feature_management::manage_features(int lower_bound, int upper_bound){
    bool initial_image_ = true;
    std::cout << "NEW CASE" << '\n';    
    std::cout << images_paths_.size() << '\n';
    std::cout << lower_bound << '\n';
    std::cout << upper_bound << '\n';
    int img_count_ = 0;
    cv::Mat test_image_;
    for(int i = lower_bound; i < upper_bound;++i){
        cv::Mat current_image_ = cv::imread(images_paths_[i],CV_LOAD_IMAGE_GRAYSCALE);
//        cv::imshow("current image", current_image_);
//        cv::waitKey(0);
        std::cout << initial_image_ << '\n';
        if(initial_image_){
            std::cout << "initial case" << '\n';
            image_feature_details_.initialize(current_image_.size(),is_9_mask_);
            initial_descriptor_ = orb_extractor_->extract_keypoints(current_image_);
            std::cout << "key points extracted" << '\n';
            image_feature_details_.calculate_center_point(orb_extractor_->returnInitialKeyPoints());

//            image_feature_details_->first_image_ = current_image_;
//            image_feature_details_->current_image_ = current_image_;
//            image_feature_details_->prev_image_ = current_image_;
            image_feature_details_.first_image_ = current_image_;
            image_feature_details_.current_image_ = current_image_;
            image_feature_details_.prev_image_ = current_image_;
            initial_image_ = false;
            test_image_ = image_feature_details_.first_image_;
            std::cout << "end initial case" << '\n';
            // continue;
        }
        else
        {
            // std::cout << initial_image_ << '\n';
            image_feature_details_.current_image_ = current_image_;

            if(!image_feature_details_.current_image_.empty() &&
                    !image_feature_details_.prev_image_.empty()){
                std::cout << "images are not empty" << '\n';
            }
            else
            {
            std::cout << "images are empty" << '\n';
            }

            orb_extractor_->extract_keypoints(image_feature_details_.current_image_,
                                            image_feature_details_.prev_image_);
    //        orb_extractor_->extract_keypoints(tmp_);
            image_feature_details_.update_all_masks(orb_extractor_->returnAllKeyPoints(),
                                                    orb_extractor_->returnMatches());

            cv::Mat modified_image_ = draw_all_points();
            cv::Mat grid_image_ = draw_grid(modified_image_);


            cv::Mat current_rgb_ = image_feature_details_.current_image_;
            cv::cvtColor(current_rgb_,current_rgb_,cv::COLOR_GRAY2RGB);
            cv::Mat tmp_grid_image_ = draw_grid(current_rgb_);
            cv::Mat tmp_initial_rgb_ = image_feature_details_.first_image_;
            cv::cvtColor(tmp_initial_rgb_,tmp_initial_rgb_,cv::COLOR_GRAY2RGB);
            // check_feature_lost_percentage(draw_grid(tmp_initial_rgb_));
            check_feature_lost_percentage(draw_grid(current_rgb_));

            // grid_image_ = put_text(grid_image_);

            std::vector<cv::Mat> all_images_ = put_text(modified_image_);
            std::string path_grid_ = "/home/nipun/dataset/test_vision_pose/grid_" + std::to_string(img_count_) + ".png";
            std::string path_all_ = "/home/nipun/dataset/test_vision_pose/grid_all_" + std::to_string(img_count_) + ".png";
            std::string path_grid_vcat_ = "/home/nipun/dataset/test_vision_pose/grid_vcat_" + std::to_string(img_count_) + ".png";
            std::string path_grid_tmpgrid_ = "/home/nipun/dataset/test_vision_pose/tmpgrid_" + std::to_string(img_count_) + ".png";
            std::string path_motion_vec_ = "/home/nipun/dataset/test_vision_pose/motion_vec_" + std::to_string(img_count_) + ".png";
            img_count_ += 1;
            // cv::imwrite(path_grid_,all_images_[0]);
            // cv::imwrite(path_all_,all_images_[1]);

            if(first_pair_){
                std::cout << "in the first pair" << '\n';
                image_feature_details_.calculate_center_point(orb_extractor_->returnCurrentKeyPoints(),
                                                            false);        
                first_pair_ = false;
            }
            std::cout << "masks size " << image_feature_details_.masks_.size() <<  '\n';
            std::cout << "prev mask left " << image_feature_details_.masks_[0].prev_center_kp_.pt << '\n';
            std::cout << "current mask left " << image_feature_details_.masks_[0].current_center_kp_.pt << '\n';
            std::cout << "prev mask center " << image_feature_details_.masks_[1].prev_center_kp_.pt << '\n';
            std::cout << "current mask center " << image_feature_details_.masks_[1].current_center_kp_.pt << '\n';
            std::cout << "prev mask right " << image_feature_details_.masks_[2].prev_center_kp_.pt << '\n';
            std::cout << "current mask right " << image_feature_details_.masks_[2].current_center_kp_.pt << '\n';

            // cv::imwrite(path_grid_tmpgrid_,tmp_grid_image_);

            track_features();
            cv::Mat tracked_image_ = draw_tracked_points(current_rgb_);
            // cv::Mat search_windows_image_ = draw_search_windows(current_rgb_);
            cv::Mat search_windows_image_ = draw_tracked_points(tmp_initial_rgb_);
            search_windows_image_ = draw_search_windows(tmp_initial_rgb_);
            search_windows_image_ = draw_grid(search_windows_image_);
            // test_image_ = draw_search_windows(test_image_);
            test_image_ = test_point_tracker(test_image_);

            cv::Mat vcat_;
            cv::vconcat(grid_image_,current_rgb_,vcat_);
            // cv::imwrite(path_grid_vcat_, vcat_);
            // cv::imshow("grid image", vcat_);
            // cv::imshow("tracked image", tracked_image_);
            cv::imshow("search windows", search_windows_image_);
            cv::imwrite(path_motion_vec_, search_windows_image_);
            // cv::imshow("test window", test_image_);
            if ((cv::waitKey() & 255) == 27) {
                image_feature_details_.prev_image_ = current_image_;
                update_center_points();
                break;
            }

            // break;

        }
        
 
    }

}

void feature_management::manage_lost_features(int lower_bound, int upper_bound){
    bool initial_image_ = true;
    bool initial_pair_ = true;
    std::cout << "NEW CASE" << '\n';    
    std::cout << images_paths_.size() << '\n';
    std::cout << lower_bound << '\n';
    std::cout << upper_bound << '\n';
    int img_count_ = 0;
    cv::Mat test_image_;
    cv::Rect bb_left_, bb_right_; 
    int inital_gm_;

    for(int i = lower_bound; i < upper_bound;++i){
        img_count_ += 1;
        cv::Mat current_image_ = cv::imread(images_paths_[i],CV_LOAD_IMAGE_GRAYSCALE);
        if(initial_image_){
            std::cout << "initial case" << '\n';
            image_feature_details_.initialize(current_image_.size(),is_9_mask_);
            bb_left_ = image_feature_details_.masks_[0].bounding_box_;
            bb_right_ = image_feature_details_.masks_[2].bounding_box_;          
            cv::Mat croppedImageLeft = current_image_(bb_left_);
            cv::Mat croppedImageRight = current_image_(bb_right_);

            croppedImageLeft.setTo(cv::Scalar(0));
            croppedImageRight.setTo(cv::Scalar(0));

            image_feature_details_.first_image_ = current_image_;
            initial_image_ = false;
        }
        else
        {
            //detect features frame initial frame and frame i
            //matched feature
            //if the matched feature in frame i is in center mask
            //the draw both train and query point 

            if(!image_feature_details_.first_image_.empty() &&
                    !current_image_.empty()){
                std::cout << "images are not empty" << '\n';
            }
            else
            {
                std::cout << "images are empty" << '\n';
            }

            std::vector<cv::KeyPoint> keypoints_current_, keypoints_prev_;
            cv::Mat descriptors_1, descriptors_2;
            std::vector<cv::DMatch> matches;

            cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num_keypts_,scale_factor_,num_levels_);
            cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
            cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

            detector->detect ( current_image_, keypoints_current_ );
            detector->detect ( image_feature_details_.first_image_,keypoints_prev_ );
            descriptor->compute ( current_image_,keypoints_current_ , descriptors_1 );
            descriptor->compute ( image_feature_details_.first_image_, keypoints_prev_ , descriptors_2 );
            matcher->match ( descriptors_1, descriptors_2, matches );

            cv::Mat current_rgb_ = current_image_;
            cv::Mat croppedLeftCRGB = current_rgb_(bb_left_);
            cv::Mat croppedRightCRGB = current_rgb_(bb_right_);
            croppedLeftCRGB.setTo(cv::Scalar(0));
            croppedRightCRGB.setTo(cv::Scalar(0));
            cv::cvtColor(current_rgb_,current_rgb_,cv::COLOR_GRAY2RGB);

            std::cout << "matches size " << matches.size() << '\n';

            //Matching point pair screening
            double min_dist=10000, max_dist=0;

            //Find the minimum and maximum distances between all matches, that is,
            //the distance between the most similar and least similar two sets of points
            for ( int i = 0; i < descriptors_1.rows; i++ )
            {
                double dist = matches[i].distance;
                if ( dist < min_dist ) min_dist = dist;
                if ( dist > max_dist ) max_dist = dist;
            }

            printf ( "-- Max dist : %f \n", max_dist );
            printf ( "-- Min dist : %f \n", min_dist );

            //When the distance between the descriptors is greater than twice the minimum distance,
            //the match is considered wrong. But sometimes the minimum distance will be very small,
            //so set an empirical value of 30 as the lower limit.
            std::vector< cv::DMatch > good_matches;
            for ( int i = 0; i < descriptors_1.rows; i++ )
            {
                if ( matches[i].distance <= cv::max ( 2*min_dist, 40.0 ) )
                {
                    good_matches.push_back ( matches[i] );
                }
            }

            std::cout << "good matches size " << good_matches.size() << '\n';

            if(initial_pair_){
                initial_pair_ = false;
                inital_gm_ = good_matches.size();
            }
            else
            {
                double val_ = 100 - (static_cast<double>(good_matches.size())/static_cast<double>(inital_gm_))*100;
                std::cout << "percentage " << val_ << '\n';

                std::stringstream stringStremPre, stringStremFN;
                stringStremPre << std::setprecision(2) << std::fixed << val_;
                stringStremFN << std::setprecision(2) << std::fixed << img_count_;

                cv::putText(current_rgb_,
                            "Frame number : " + stringStremFN.str(),
                            cv::Point(image_feature_details_.masks_[0].bounding_box_.x + 20,
                                      image_feature_details_.masks_[0].bounding_box_.y + 20), // Coordinates
                            cv::FONT_HERSHEY_PLAIN, // Font
                            1, // Scale. 2.0 = 2x bigger
                            cv::Scalar(0,215,255),
                            2); // Line Thickness (Optional)

                cv::putText(current_rgb_,
                            "Lost KP : " + stringStremPre.str() + "%",
                            cv::Point(image_feature_details_.masks_[0].bounding_box_.x + 20,
                                      image_feature_details_.masks_[0].bounding_box_.y + 40), // Coordinates
                            cv::FONT_HERSHEY_PLAIN, // Font
                            1, // Scale. 2.0 = 2x bigger
                            cv::Scalar(0,215,255),
                            2); // Line Thickness (Optional)
                

            }
            

            cv::Rect center_mask_ = image_feature_details_.masks_[1].bounding_box_;
            cv::Rect left_mask_ = image_feature_details_.masks_[2].bounding_box_;
            left_mask_.width = left_mask_.width - 200;
            for(std::vector<cv::DMatch>::iterator it_ = good_matches.begin(); it_ != good_matches.end();++it_){
                cv::Point2f current_ = keypoints_current_[it_->queryIdx].pt;
                cv::Point2f prev_ = keypoints_prev_[it_->trainIdx].pt;

                cv::circle(current_rgb_, prev_,
                            1, cv::Scalar(255,0,0), 1, cv::LINE_8, 0);

                if(center_mask_.contains(current_)){
                    cv::circle(current_rgb_, current_,
                            1, cv::Scalar(0,255,0), 1, cv::LINE_8, 0);
                }
                else if (left_mask_.contains(current_))
                {
                    cv::circle(current_rgb_, current_,
                            1, cv::Scalar(0,0,255), 1, cv::LINE_8, 0);
                }
 
            }
            current_rgb_ = draw_grid(current_rgb_);
            cv::imshow("current", current_rgb_);
            std::string path_lost_kp_ = "/home/nipun/dataset/test_vision_pose/lostKP_" + std::to_string(img_count_) + ".png";
            cv::imwrite(path_lost_kp_,current_rgb_);
            // cv::imshow("initial", image_feature_details_.first_image_);
            cv::waitKey(0);
            // break;

        }
        

    }
}

void feature_management::check_feature_lost_percentage(cv::Mat input_){
    cv::Rect bb_left_ = image_feature_details_.masks_[0].bounding_box_;
    cv::Rect bb_right_ = image_feature_details_.masks_[2].bounding_box_;
    // cv::Mat croppedImageLeft = input_(bb_left_);
    // cv::Mat croppedImageRight = input_(bb_right_);

    // croppedImageLeft.setTo(cv::Scalar(0,0,0));
    // croppedImageRight.setTo(cv::Scalar(0,0,0));

    cv::Mat test_ = input_;
    // test_.copyTo(croppedImageLeft);
    // test_.copyTo(croppedImageRight);

    test_ = draw_all_points_center(test_);

    cv::imshow("croped",test_);
    cv::waitKey(0);

}

void feature_management::track_features(){
    std::cout << "in tracking" << '\n';
    for(unsigned long int i = 0; i< image_feature_details_.masks_.size(); ++i){
        std::cout << "=============== NEW MASK ===============" << '\n';
        cv::Rect bb_ = image_feature_details_.masks_[i].bounding_box_;
        cv::Mat croppedImageCurrent = image_feature_details_.current_image_(bb_);
        cv::Mat croppedImagePrev = image_feature_details_.prev_image_(bb_);
        cv::Point2f current_pt_, prev_pt_;

//        cv::imshow("current", croppedImageCurrent);
//        cv::imshow("prev", croppedImagePrev);
//        if ((cv::waitKey() & 255) == 27) {
//            break;
//        }

        if(init_tracking_){
            std::cout << "in init tracking -- FM" << '\n';
            current_pt_ = cv::Point2f(image_feature_details_.masks_[i].current_center_kp_.pt.x,
                                      image_feature_details_.masks_[i].current_center_kp_.pt.y);
            image_feature_details_.masks_[i].all_tracked_centers_.
                   push_back(image_feature_details_.masks_[i].prev_center_kp_.pt);
        }
        else {
            // This is the prev motion vector
            motion_vector_ = image_feature_details_.masks_[i].motion_vector_;
            std::cout << "prev motion vector" << '\n';
            std::cout << motion_vector_[0] << '\n';
            std::cout << motion_vector_[1] << '\n';
            //We know the motion vector has not changed since the last update.
            //So we use the motion vector to estimate the current, using the prev center
            current_pt_ = cv::Point2f(image_feature_details_.masks_[i].prev_center_kp_.pt.x + motion_vector_[0],
                                      image_feature_details_.masks_[i].prev_center_kp_.pt.y + motion_vector_[1]);
        }

        prev_pt_ = image_feature_details_.masks_[i].prev_center_kp_.pt;
        std::cout << "current : " << current_pt_ << '\n';
        std::cout << "prev : " << prev_pt_ << '\n';

        orb_tracker_->track_features(image_feature_details_.current_image_,
                                     image_feature_details_.prev_image_,
                                     current_pt_,prev_pt_,bb_,init_tracking_);

        motion_vector_ = orb_tracker_->return_motion_vector();
        std::vector<cv::Rect> all_windows_ = orb_tracker_->return_search_windows();
        std::cout << motion_vector_[0] << '\n';
        std::cout << motion_vector_[1] << '\n';
        //update the current center
        image_feature_details_.masks_[i].current_center_kp_.pt.x =
                image_feature_details_.masks_[i].prev_center_kp_.pt.x + motion_vector_[0];
        image_feature_details_.masks_[i].current_center_kp_.pt.y =
                image_feature_details_.masks_[i].prev_center_kp_.pt.y + motion_vector_[1];

        image_feature_details_.masks_[i].all_tracked_centers_.
                push_back(image_feature_details_.masks_[i].current_center_kp_.pt);

        //update the motion vector of the mask
        image_feature_details_.masks_[i].motion_vector_ = motion_vector_;


//        image_feature_details_.masks_[i].current_search_win_ = all_windows_[0];
//        image_feature_details_.masks_[i].prev_search_win_ = all_windows_[1];

        all_current_sw_.push_back(all_windows_[0]);
        all_prev_sw_.push_back(all_windows_[1]);
    }
    init_tracking_ = false;
    std::cout << "=============== DONE ALL MASKS ===============" << '\n';
}

void feature_management::manage_no_masks(int lower_bound, int upper_bound){
    std::cout << "NEW CASE" << '\n';    
    std::cout << images_paths_.size() << '\n';
    std::cout << lower_bound << '\n';
    std::cout << upper_bound << '\n';
    
    bool initial_image_ = true;
    int img_count_ = 0;
    cv::Mat test_image_;
    cv::Point current_center_, prev_center_;
    cv::Rect image_rec_;
    cv::Mat prev_image_;
    cv::Mat initial_frame_;


    for(int i = lower_bound; i < upper_bound;++i){
        cv::Mat current_image_ = cv::imread(images_paths_[i],CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat current_rgb_ = current_image_;
        cv::cvtColor(current_rgb_,current_rgb_,cv::COLOR_GRAY2RGB);

        if (initial_image_){
            image_rec_.x = 0;
            image_rec_.y = 0;
            image_rec_.height = current_image_.rows;
            image_rec_.width = current_image_.cols;

            current_center_.x = current_image_.cols/2;
            current_center_.y = current_image_.rows/2;
            prev_center_.x = current_image_.cols/2;
            prev_center_.y = current_image_.rows/2;

            prev_image_ = current_image_;
            initial_frame_ = current_image_;
            cv::cvtColor(initial_frame_,initial_frame_,cv::COLOR_GRAY2RGB);

            initial_image_ = false;
            img_count_ += 1;
        }
        else
        {
            std::cout << "frame " << img_count_ << '\n';
            if(!prev_image_.empty() && !current_image_.empty()){
                std::cout << "images are not empty" << '\n';
            }
            else
            {
                std::cout << "images are empty" << '\n';
            }

            std::cout << "prev : " << prev_center_ << '\n';
            current_center_ = track_features_single_mask(current_image_,prev_image_,prev_center_);
            std::cout << "current : " << current_center_ << '\n';
            std::cout << "-----------------------" << '\n';
            cv::line(current_rgb_,prev_center_,current_center_,cv::Scalar(0,0,255),2,cv::LINE_8);
            cv::line(initial_frame_,prev_center_,current_center_,cv::Scalar(0,0,255),2,cv::LINE_8);

            initial_frame_ = draw_search_window(initial_frame_,current_center_,prev_center_);

            cv::imshow("current", current_rgb_);
            cv::imshow("initial", initial_frame_);

            std::string path_no_mask_ = "/home/nipun/dataset/test_vision_pose/no_mask_" + std::to_string(img_count_) + ".png";
            cv::imwrite(path_no_mask_,initial_frame_);

            cv::waitKey(0);
            img_count_ += 1;
            prev_center_ = current_center_;
            prev_image_ = current_image_;

            // break;
        }
        
        

    }
}

cv::Point feature_management::track_features_single_mask(cv::Mat current_image_, cv::Mat prev_image_,
                                                    cv::Point prev_center_){
    cv::Point updated_center_;
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches;  

    extract_single_mask(current_image_,prev_image_,
    keypoints_1,keypoints_2,descriptors_1,descriptors_2,matches); 

    //calculate motion vector
    motion_vector_single_mask_ = calculate_geomatric_center(matches,keypoints_1,keypoints_2);
    std::cout << motion_vector_single_mask_[0] << " " << motion_vector_single_mask_[1] << '\n';
    //update the current center
    updated_center_.x = prev_center_.x + motion_vector_single_mask_[0];
    updated_center_.y = prev_center_.y + motion_vector_single_mask_[1];

    return updated_center_;

}

void feature_management::extract_single_mask(cv::Mat current_image_, cv::Mat prev_image_,
                            std::vector<cv::KeyPoint> &keypoints_current_,
                            std::vector<cv::KeyPoint> &keypoints_prev_, 
                            cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                            std::vector<cv::DMatch> &matches){

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(max_num_keypts_,scale_factor_,num_levels_);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect ( current_image_, keypoints_current_ );
    detector->detect ( prev_image_,keypoints_prev_ );
    descriptor->compute ( current_image_,keypoints_current_ , descriptors_1 );
    descriptor->compute ( prev_image_, keypoints_prev_ , descriptors_2 );
    matcher->match ( descriptors_1, descriptors_2, matches );

    std::cout << "matches size " << matches.size() << '\n';

    //Matching point pair screening
    double min_dist=10000, max_dist=0;

    //Find the minimum and maximum distances between all matches, that is,
    //the distance between the most similar and least similar two sets of points
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //When the distance between the descriptors is greater than twice the minimum distance,
    //the match is considered wrong. But sometimes the minimum distance will be very small,
    //so set an empirical value of 30 as the lower limit.
    std::vector< cv::DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= cv::max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    std::cout << "good matches size " << good_matches.size() << '\n';

    matches.clear();    
    matches = good_matches;
    // std::vector<cv::KeyPoint> updated_kp_1_,updated_kp_2_;
    // for(std::vector<cv::DMatch>::iterator it_ = good_matches.begin(); it_ != good_matches.end();++it_)
    // {
    //     // std::cout<< it_->imgIdx << " "
    //     //         << it_->distance << " "
    //     //         << it_->queryIdx << " "
    //     //         << it_->trainIdx << " "
    //     //         << keypoints_current_.at(it_->queryIdx).pt << " "  //current
    //     //         << keypoints_prev_.at(it_->trainIdx).pt << '\n';  //prev
    //     updated_kp_1_.push_back(keypoints_current_.at(it_->queryIdx));
    //     updated_kp_2_.push_back(keypoints_prev_.at(it_->trainIdx));
    // }

    // keypoints_current_.clear();
    // keypoints_prev_.clear();

    // keypoints_current_ = updated_kp_1_;
    // keypoints_prev_ = updated_kp_2_;

}

std::vector<float> feature_management::calculate_geomatric_center(std::vector<cv::DMatch> matches_,
                            std::vector<cv::KeyPoint> keypoints_current_,
                            std::vector<cv::KeyPoint> keypoints_prev_){
    
    float C_cx_ = 0, C_cy_ = 0, C_px_ = 0, C_py_ = 0;
    std::vector<float> motion_vec_ = {0,0};
    std::cout << "matches size " << matches_.size() << '\n';
    std::cout << "current kp size " << keypoints_current_.size() << '\n';
    std::cout << "prev kp size " << keypoints_prev_.size() << '\n';

    if(matches_.size() != 0){
        for(std::vector<cv::DMatch>::iterator it_ = matches_.begin(); it_ != matches_.end();++it_){
            // std::cout<< it_->imgIdx << " "
            //         << it_->distance << " "
            //         << it_->queryIdx << " "
            //         << it_->trainIdx << " "
            //         << keypoints_current_.at(it_->queryIdx).pt << " "  //current
            //         << keypoints_prev_.at(it_->trainIdx).pt << '\n';  //prev

            cv::Point2f current_ = keypoints_current_.at(it_->queryIdx).pt;
            cv::Point2f prev_ = keypoints_prev_.at(it_->trainIdx).pt;

            // std::cout << "prev : " << prev_ << '\n';
            // std::cout << "current : " << current_ << '\n';
            // std::cout << "-----------------------" << '\n';

            C_cx_ += current_.x;
            C_cy_ += current_.y;
            C_px_ += prev_.x;
            C_py_ += prev_.y;
        }

        C_cx_ /= matches_.size();
        C_cy_ /= matches_.size();
        C_px_ /= matches_.size();
        C_py_ /= matches_.size();

        motion_vec_ = {C_cx_ - C_px_, C_cy_- C_py_};
    }

    return motion_vec_;

}

cv::Mat feature_management::draw_search_window(cv::Mat input, cv::Point current_, cv::Point prev_){
    int alpha_ = 30;
    int image_height_ = input.rows;
    int image_width_ = input.cols;
    int height_ = 2*alpha_;
    int width_ = 2*alpha_;

    std::cout << "current : " << current_ << '\n';
    std::cout << "prev : " << prev_ << '\n';
    cv::Point2f tmp_ = current_ - prev_;

    int x_ = current_.x + static_cast<int>(tmp_.x) - width_/2;
    x_ = x_ < 0 ? 0 : x_;
    x_ = x_ > image_width_ ? image_width_ : x_;
    int y_ = current_.y + static_cast<int>(tmp_.y) - height_/2;
    y_ = y_ < 0 ? 0 : y_;
    y_ = y_ > image_height_ ? image_height_ : y_;

    std::cout << "x : " << x_ << " y : " << y_
              << " end x : " << x_ + width_ << " end y : " << y_ + width_
              <<" width: " << width_ << " height : " << height_  << '\n';
    std::cout <<"image width: " << image_width_ << " height : " << image_height_ << '\n';

    cv::Point2f strat_ = cv::Point2f(x_, y_);
    cv::Point2f end_ = cv::Point2f(x_ + width_, y_ + width_);
    cv::rectangle(input,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);

    return input;

}

void feature_management::update_center_points(){
    for(unsigned int i = 0; i < image_feature_details_.masks_.size(); ++i){
        image_feature_details_.masks_[i].prev_center_kp_ =  image_feature_details_.masks_[i].current_center_kp_;
    }
}

cv::Mat feature_management::draw_all_points(){
    cv::Mat modified_image_ = image_feature_details_.first_image_.clone();
    cv::cvtColor(modified_image_,modified_image_,cv::COLOR_GRAY2RGB);
    for(unsigned int i = 0; i < image_feature_details_.masks_.size(); ++i){
        mask_image current_mask_ = image_feature_details_.masks_[i];
        for(unsigned long int j = 0; j < current_mask_.current_keypoints_.size(); ++j){            
            // if(current_mask_.current_keypoints_.size() == current_mask_.initial_keypoints_.size()){
            //     cv::circle(modified_image_, current_mask_.initial_keypoints_[j].pt,
            //                1, cv::Scalar(255,0,0), 1, cv::LINE_8, 0);
            // }
            cv::circle(modified_image_, current_mask_.initial_keypoints_[j].pt,
                           1, cv::Scalar(255,0,0), 1, cv::LINE_8, 0);
            cv::circle(modified_image_, current_mask_.prev_keypoints_[j].pt,
                       1, cv::Scalar(0,255,0), 1, cv::LINE_8, 0);
            cv::circle(modified_image_, current_mask_.current_keypoints_[j].pt,
                       1, cv::Scalar(0,0,255), 1, cv::LINE_8, 0);
        }
    }

    return modified_image_;
}

cv::Mat feature_management::draw_all_points_center(cv::Mat image_){
    cv::Mat modified_image_ = image_.clone();
    mask_image current_mask_ = image_feature_details_.masks_[1];
    for(unsigned long int j = 0; j < current_mask_.current_keypoints_.size(); ++j){            
        // if(current_mask_.current_keypoints_.size() == current_mask_.initial_keypoints_.size()){
        //     cv::circle(modified_image_, current_mask_.initial_keypoints_[j].pt,
        //                1, cv::Scalar(255,0,0), 1, cv::LINE_8, 0);
        // }
        cv::circle(modified_image_, current_mask_.initial_keypoints_[j].pt,
                        1, cv::Scalar(255,0,0), 1, cv::LINE_8, 0);
        cv::circle(modified_image_, current_mask_.prev_keypoints_[j].pt,
                    1, cv::Scalar(0,255,0), 1, cv::LINE_8, 0);
        cv::circle(modified_image_, current_mask_.current_keypoints_[j].pt,
                    1, cv::Scalar(0,0,255), 1, cv::LINE_8, 0);
    }

    std::cout << "initial " << current_mask_.initial_keypoints_.size() <<  '\n';
    std::cout << "prev " << current_mask_.prev_keypoints_.size() << '\n';
    std::cout << "current " << current_mask_.current_keypoints_.size() << '\n';
    return modified_image_;
}

cv::Mat feature_management::put_text_single(cv::Mat input_){

    for(unsigned long int i = 0; i < image_feature_details_.masks_.size(); ++i){
        std::stringstream stringStremNumKP, stringStremTotal, stringStremAve;
        stringStremNumKP << std::setprecision(2) << std::fixed << image_feature_details_.masks_[i].current_keypoints_.size();
        stringStremTotal << std::setprecision(2) << std::fixed << image_feature_details_.masks_[i].total_distance_;
        stringStremAve << std::setprecision(2) << std::fixed << image_feature_details_.masks_[i].average_distance_;

        cv::putText(input_,
                    "KP : " + stringStremNumKP.str(),
                    cv::Point(image_feature_details_.masks_[i].bounding_box_.x,
                              image_feature_details_.masks_[i].bounding_box_.y + 20), // Coordinates
                    cv::FONT_HERSHEY_PLAIN, // Font
                    1, // Scale. 2.0 = 2x bigger
                    cv::Scalar(0,215,255),
                    2); // Line Thickness (Optional)

        cv::putText(input_,
                    "TD : " +stringStremTotal.str(),
                    cv::Point(image_feature_details_.masks_[i].bounding_box_.x,
                              image_feature_details_.masks_[i].bounding_box_.y + 40), // Coordinates
                    cv::FONT_HERSHEY_PLAIN, // Font
                    1, // Scale. 2.0 = 2x bigger
                    cv::Scalar(0,215,255),
                    2); // Line Thickness (Optional)

        cv::putText(input_,
                    "AD : " +stringStremAve.str(),
                    cv::Point(image_feature_details_.masks_[i].bounding_box_.x,
                              image_feature_details_.masks_[i].bounding_box_.y + 60), // Coordinates
                    cv::FONT_HERSHEY_PLAIN, // Font
                    1, // Scale. 2.0 = 2x bigger
                    cv::Scalar(0,215,255),
                    2); // Line Thickness (Optional)
    }

    // int tmp = 9;
    // std::string test_ = "thisis a test" + std::to_string(tmp);

    return input_;

//    cv::rectangle(maskImg, cv::Point(y, x),
//                  cv::Point(y + colFact - 1, x + rowFact - 1),
//                  CV_RGB(255, 0, 0), 1);
}

std::vector<cv::Mat> feature_management::put_text(cv::Mat input_){
    cv::Mat grid_image_ = draw_grid(input_);
    
    int current_num_kp_ = 0;
    double total = 0.0;
    double average = 0.0;

    for(unsigned long int i = 0; i < image_feature_details_.masks_.size(); ++i){
        std::stringstream stringStremNumKP, stringStremTotal, stringStremAve;
        stringStremNumKP << std::setprecision(2) << std::fixed << image_feature_details_.masks_[i].current_keypoints_.size();
        stringStremTotal << std::setprecision(2) << std::fixed << image_feature_details_.masks_[i].total_distance_;
        stringStremAve << std::setprecision(2) << std::fixed << image_feature_details_.masks_[i].average_distance_;
        
        current_num_kp_ += image_feature_details_.masks_[i].current_keypoints_.size();
        total += image_feature_details_.masks_[i].total_distance_;
        average += image_feature_details_.masks_[i].average_distance_;
        
        cv::putText(grid_image_,
                    "KP : " + stringStremNumKP.str(),
                    cv::Point(image_feature_details_.masks_[i].bounding_box_.x,
                              image_feature_details_.masks_[i].bounding_box_.y + 20), // Coordinates
                    cv::FONT_HERSHEY_PLAIN, // Font
                    1, // Scale. 2.0 = 2x bigger
                    cv::Scalar(0,215,255),
                    2); // Line Thickness (Optional)

        cv::putText(grid_image_,
                    "TD : " + stringStremTotal.str(),
                    cv::Point(image_feature_details_.masks_[i].bounding_box_.x,
                              image_feature_details_.masks_[i].bounding_box_.y + 40), // Coordinates
                    cv::FONT_HERSHEY_PLAIN, // Font
                    1, // Scale. 2.0 = 2x bigger
                    cv::Scalar(0,215,255),
                    2); // Line Thickness (Optional)

        cv::putText(grid_image_,
                    "AD : " + stringStremAve.str(),
                    cv::Point(image_feature_details_.masks_[i].bounding_box_.x,
                              image_feature_details_.masks_[i].bounding_box_.y + 60), // Coordinates
                    cv::FONT_HERSHEY_PLAIN, // Font
                    1, // Scale. 2.0 = 2x bigger
                    cv::Scalar(0,215,255),
                    2); // Line Thickness (Optional)
    }

    cv::Mat all_grid_ = input_.clone();
    std::stringstream stringStremTotalNumKP, stringStremTotalImage, stringStremAveImage;
    stringStremTotalNumKP << std::setprecision(2) << std::fixed << current_num_kp_;
    stringStremTotalImage << std::setprecision(2) << std::fixed << total;
    stringStremAveImage << std::setprecision(2) << std::fixed << average;

    cv::putText(all_grid_,
                "KP : " + stringStremTotalNumKP.str(),
                cv::Point(image_feature_details_.masks_[0].bounding_box_.x,
                            image_feature_details_.masks_[0].bounding_box_.y + 20), // Coordinates
                cv::FONT_HERSHEY_PLAIN, // Font
                1, // Scale. 2.0 = 2x bigger
                cv::Scalar(0,215,255),
                2); // Line Thickness (Optional)

    cv::putText(all_grid_,
                "TD : " + stringStremTotalImage.str(),
                cv::Point(image_feature_details_.masks_[0].bounding_box_.x,
                            image_feature_details_.masks_[0].bounding_box_.y + 40), // Coordinates
                cv::FONT_HERSHEY_PLAIN, // Font
                1, // Scale. 2.0 = 2x bigger
                cv::Scalar(0,215,255),
                2); // Line Thickness (Optional)

    cv::putText(all_grid_,
                "AD : " + stringStremAveImage.str(),
                cv::Point(image_feature_details_.masks_[0].bounding_box_.x,
                            image_feature_details_.masks_[0].bounding_box_.y + 60), // Coordinates
                cv::FONT_HERSHEY_PLAIN, // Font
                1, // Scale. 2.0 = 2x bigger
                cv::Scalar(0,215,255),
                2); // Line Thickness (Optional)
    
    std::vector<cv::Mat> image_vector_;
    image_vector_.push_back(grid_image_);
    image_vector_.push_back(all_grid_);

    return image_vector_;

}


cv::Mat feature_management::draw_tracked_points(cv::Mat image_){
    for(unsigned long int i = 0; i < image_feature_details_.masks_.size(); ++i){
        for(unsigned long int j = 1; j < image_feature_details_.masks_[i].all_tracked_centers_.size(); ++j){
            cv::Point2f current_ = image_feature_details_.masks_[i].all_tracked_centers_[j];
            cv::Point2f prev_ = image_feature_details_.masks_[i].all_tracked_centers_[j-1];
            cv::line(image_,prev_,current_,cv::Scalar(0,0,255),2,cv::LINE_8);
        }
    }

    return image_;
}

cv::Mat feature_management::draw_search_windows(cv::Mat image_){
    for(unsigned long int i = 0; i < all_current_sw_.size(); ++i){
        cv::Point2f strat_ = cv::Point2f(all_current_sw_[i].x,all_current_sw_[i].y);
        cv::Point2f end_ = cv::Point2f(all_current_sw_[i].x + all_current_sw_[i].width,
                                       all_current_sw_[i].y + all_current_sw_[i].height);
        cv::rectangle(image_,strat_,end_,cv::Scalar(0,150,255),2,cv::LINE_8);
    }

    return image_;

}

cv::Mat feature_management::test_point_tracker(cv::Mat input_image){
    cv::Mat output_ = image_feature_details_.first_image_;
    for(unsigned long int i = 0; i < image_feature_details_.masks_.size(); ++i){
        for(unsigned long int j = 1; j < image_feature_details_.masks_[i].all_tracked_centers_.size(); ++j){
            cv::Point2f current_ = image_feature_details_.masks_[i].all_tracked_centers_[j];
            cv::Point2f prev_ = image_feature_details_.masks_[i].all_tracked_centers_[j-1];
            cv::line(input_image,prev_,current_,cv::Scalar(0,0,255),2,cv::LINE_8);
        }
    }

    return input_image;

}






