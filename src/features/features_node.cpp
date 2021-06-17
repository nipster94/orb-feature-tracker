#include "feature_management.h"

int main(int argc, char** argv)
{
    ROS_INFO("FEATURES");
    ros::init(argc, argv, "features");
    feature_management feature_manager;
    feature_manager.execute();
    return 0;
}
