/**
 * This file is part of DynaSLAM.
 *
 * Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/bertabescos/DynaSLAM>.
 *
 */

#include "MaskNet.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tf_graph_executor/tf_graph_executor.hpp>

namespace DynaSLAM
{

cv::Mat SegmentDynObject::GetSegmentation(cv::Mat &image, std::string dir, std::string name){
  tensorflow::Tensor inputImageTensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1,image.rows, image.cols, image.channels()}));
  uint8_t *tTensorData=inputImageTensor.flat<tensorflow::uint8>().data();
  std::vector<tensorflow::Tensor> outputMaskRCNNTensors;

  std::vector<std::pair<std::string, tensorflow::Tensor> > feedDict;
  feedDict.push_back(std::pair<std::string, tensorflow::Tensor>(kInputTensorName,inputImageTensor));

  mGraphExecutor->executeGraph(feedDict, kOutputTensorNames, outputMaskRCNNTensors);
  int i=0;
}
}

