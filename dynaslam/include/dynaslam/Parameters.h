/*
 * SegParamters.h
 *
 *  Created on: Jul 8, 2020
 *      Author: lxy
 */

#ifndef INCLUDE_PARAMTERS_H_
#define INCLUDE_PARAMTERS_H_

#include <string>

namespace DynaSLAM {

struct SegParameters
{
  std::string mask_rcnn_model_pb_path = "MUST_BE_SET";
};
}

#endif /* INCLUDE_PARAMTERS_H_ */
