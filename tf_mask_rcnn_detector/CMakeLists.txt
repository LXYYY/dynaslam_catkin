cmake_minimum_required(VERSION 3.0.2)
project(tf_mask_rcnn_detector)

find_package(catkin_simple 0.1.0 REQUIRED)
catkin_simple()

add_definitions(-std=c++11)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

cs_add_library(${PROJECT_NAME} 
  src/tf_mask_rcnn_detector.cpp
)

target_link_libraries(${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_tests
   test/test_mask_detector.cpp)

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})


cs_install()
cs_export()
