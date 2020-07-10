#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "dynaslam/MaskNet.h"
#include "dynaslam/Parameters.h"
class MaskRCNNDetectorTest : public ::testing::Test
{
protected:
  std::string imageFilePath;
  cv::Mat image;
  DynaSLAM::SegParameters segParams;
  DynaSLAM::SegmentDynObject *maskNet;
  void SetUp() override
  {
    imageFilePath = "./src/dynaslam_catkin/dynaslam/test/kitti_test.png";
    image = cv::imread(imageFilePath, CV_LOAD_IMAGE_UNCHANGED);
    DynaSLAM::SegParameters segParams;
    segParams.mask_rcnn_model_pb_path = "/home/lxy/Workspace/mrslam/mask_rcnn.pb";
    maskNet = new DynaSLAM::SegmentDynObject(segParams);
  }

  void TearDown() override
  {
  }
};

TEST_F(MaskRCNNDetectorTest, test_mask_detector_working)
{
  cv::Mat tImage;
  ASSERT_NO_THROW(maskNet->GetSegmentation(tImage));
}