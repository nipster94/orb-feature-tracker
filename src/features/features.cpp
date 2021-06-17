#include "orb_features.h"

Featuers::Featuers(){

}

void Featuers::execute(){

    std::string current_path_ = "/home/nipun/MPSYS/Thesis/Semcon/git_ws/src/lundb_ws/src/sensor_packs/"
                              "object_detection/resource/17_17_29_new_image_set/image_0/000001.png";

    std::string prev_path_ = "/home/nipun/MPSYS/Thesis/Semcon/git_ws/src/lundb_ws/src/sensor_packs/"
                              "object_detection/resource/17_17_29_new_image_set/image_0/000000.png";

    cv::Mat current_image_ = cv::imread(current_path_,CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat prev_image_ = cv::imread(prev_path_,CV_LOAD_IMAGE_UNCHANGED);


    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect ( current_image_,keypoints_1 );
    detector->detect ( prev_image_,keypoints_2 );

    descriptor->compute ( current_image_, keypoints_1, descriptors_1 );
    descriptor->compute ( prev_image_, keypoints_2, descriptors_2 );

    cv::Mat outimg1;
    cv::drawKeypoints( current_image_, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    cv::imshow("ORB",outimg1);
//    cv::waitKey(0);

    //Match the BRIEF descriptors in the two images, using Hamming distance
    std::vector<cv::DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );

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

    //draw matching results
    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches (current_image_ , keypoints_1,prev_image_ , keypoints_2, matches, img_match );
    cv::drawMatches (current_image_, keypoints_1, prev_image_, keypoints_2, good_matches, img_goodmatch );
    imshow ( "All matching points", img_match );
    imshow ( "Match point pairs after optimization", img_goodmatch );
    cv::waitKey(0);

}
