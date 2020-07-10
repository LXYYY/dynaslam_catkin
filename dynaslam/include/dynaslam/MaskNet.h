/**
 * This file is part of DynaSLAM.
 * Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/bertabescos/DynaSLAM>.
 *
 */

#ifndef __MASKNET_H
#define __MASKNET_H

//#include <object.h>

#include <memory>

namespace cv
{
class Mat;
} /* namespace cv */

#ifndef NULL
#define NULL ((void *)0)
#endif

#include <assert.h>
#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <tf_mask_rcnn_detector/tf_mask_rcnn_detector.hpp>
#include <boost/thread.hpp>
#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "dynaslam/Conversion.h"
#include "dynaslam/Parameters.h"

namespace DynaSLAM
{
class SegmentDynObject
{
public:
  explicit SegmentDynObject(const SegParameters &segParams) : mParams(segParams)
  {
    const std::string mask_rcnn_model_path = mParams.mask_rcnn_model_pb_path;
    LOG(INFO) << "Loading MaskRCNN model in " + mask_rcnn_model_path;
    tf_mask_rcnn_detector::MaskRCNNParameters maskParameters;
    mpMaskDetector.reset(new tf_mask_rcnn_detector::TensorFlowMaskRCNNDetector(maskParameters));
    mpMaskDetector->LoadModel(mask_rcnn_model_path);
  }

  ~SegmentDynObject()
  {
  }

  cv::Mat GetSegmentation(cv::Mat &image, std::string dir, std::string name);

private:
  std::shared_ptr<tf_mask_rcnn_detector::TensorFlowMaskRCNNDetector> mpMaskDetector;

  const SegParameters mParams;

  const std::string kInputTensorName = "input_image";
  const std::vector<std::string> kOutputTensorNames = { "detections", "mrcnn_class", "mrcnn_bbox", "mrcnn_mask",
                                                        "rpn_rois",   "rpn_class",   "rpn_bbox" };
};
}  // namespace DynaSLAM

#endif
