#ifndef FEATURES_H
#define FEATURES_H
//#ifndef OBJECT_DETECTION_H
//#define OBJECT_DETECTION_H

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

#include <opencv2/features2d.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

class Featuers
{
public:
    Featuers();
    void execute(void);

};


#endif
