cmake_minimum_required(VERSION 3.5.1)
project(vision_pose_estimation)

set(CMAKE_CXX_STANDARD 14)
set(GCC_COVERAGE_COMPILE_FLAGS " -Ofast")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")

#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else()
#    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()

set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
			/usr/local/opencv-3.4.7/share/OpenCV
			/usr/share/OpenCV)

find_package(OpenCV 3.4.7 QUIET)
#find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  dynamic_reconfigure
  message_generation
)

generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES orb_extractor orb_tracker
  CATKIN_DEPENDS cv_bridge image_transport message_generation roscpp sensor_msgs std_msgs geometry_msgs message_runtime
)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include/features
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(feature_params
   src/features/feature_params.cpp
)

add_library(orb_extractor
   src/features/orb_extractor.cpp
)

add_library(orb_tracker
   src/features/orb_tracker.cpp
)

target_link_libraries(orb_extractor ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(orb_extractor ${catkin_EXPORTED_TARGETS})

target_link_libraries(orb_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(orb_tracker ${catkin_EXPORTED_TARGETS})

target_link_libraries(feature_params ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(feature_params ${catkin_EXPORTED_TARGETS})

#add_executable(features src/features/features.cpp src/features/features_node.cpp)
#target_link_libraries(features ${catkin_LIBRARIES} ${OpenCV_LIBS})
#add_dependencies(features vision_pose_estimation_generate_messages_cpp ${catkin_EXPORTED_TARGETS})

add_executable(feature_management src/features/features_node.cpp src/features/feature_management.cpp)
target_link_libraries(feature_management
                      orb_extractor orb_tracker feature_params
                      ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(feature_management vision_pose_estimation_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
