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

namespace cv {
class Mat;
} /* namespace cv */

#ifndef NULL
#define NULL   ((void *) 0)
#endif

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread.hpp>
#include "include/Conversion.h"
#include <opencv2/imgproc.hpp>
#include "include/Parameters.h"
#include <tf_graph_executor/tf_graph_executor.hpp>
#include <glog/logging.h>

namespace DynaSLAM {

class SegmentDynObject {
 public:
  explicit SegmentDynObject(const SegParameters &segParams)
      :
      mParams(segParams) {
    const std::string mask_rcnn_model_path = mParams.mask_rcnn_model_pb_path;
    const std::string mask_rcnn_graph_def_path=mask_rcnn_model_path+"/graph_def_for_reference.pb.ascii";
    const std::string mask_rcnn_weights_path=mask_rcnn_model_path+"/mask_rcnn.pb";
    LOG(INFO) << "Loading MaskRCNN model in " + mask_rcnn_model_path;
    mGraphExecutor.reset(
        new tf_graph_executor::TensorflowGraphExecutor(mask_rcnn_weights_path));
  }

  ~SegmentDynObject() {
  }

  cv::Mat GetSegmentation(cv::Mat &image, std::string dir, std::string name);
 private:
  std::shared_ptr<tf_graph_executor::TensorflowGraphExecutor> mGraphExecutor;

  SegParameters mParams;

  const std::string kInputTensorName="input_image";
  const std::vector<std::string> kOutputTensorNames={
      "output_detections",
      "output_mrcnn_class",
      "output_mrcnn_bbox",
      "output_mrcnn_mask",
      "output_rpn_rois",
      "output_rpn_class",
      "output_rpn_bbox"
  };
};
}

#endif
