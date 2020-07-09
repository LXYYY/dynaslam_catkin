/**
 * This file is part of DynaSLAM.
 *
 * Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/bertabescos/DynaSLAM>.
 *
 */

#include <dirent.h>
#include <errno.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "MaskNet.h"

namespace DynaSLAM
{
cv::Mat SegmentDynObject::GetSegmentation(cv::Mat &image, std::string dir, std::string name)
{

}
}  // namespace DynaSLAM
