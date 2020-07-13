#ifndef TF_MASK_RCNN_DETECTOR_TF_MASK_RCNN_DETECTOR_HPP
#define TF_MASK_RCNN_DETECTOR_TF_MASK_RCNN_DETECTOR_HPP

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include "data_format.hpp"
namespace tf_mask_rcnn_detector
{
class MaskRCNNParameters
{
public:
  int inputTnsW;      //输入tensor的宽
  int inputTnsH;      //输入tensor的高度
  int inputTnsC = 3;  //输入tensor的通道 input tensor channels,same as image channels
  // int inputImgW = 512;  //输入图像的宽度 image width
  // int inputImgH = 512;  //输入图像的高度 image height
  // int inputImgC = 3;    //输入图像的通道 image channels
  std::vector<int> inputImageShape = { 1241, 376, 3 };
  std::vector<int> resizedImageShape;
  int imageMinDim = 800;
  int imageMaxDim = 1024;
  bool imagePadding = true;
  int numInputImage = 1;
  int numClasses = 1 + 80;
  int tfMaskRCNNImageMetaDataLength;  //=12+num_classes;//这里的19是12+7(7是有七类)  //图像的meta数据的长度,一般
  int imagePerGPU = 1;
  int GPUCount = 1;
  int batchSize = 1;
  std::vector<int> rpnAnchorScales = { 32, 64, 128, 256, 512 };  // rpn阶段生成的anchor的尺度
  std::vector<float> rpnAnchorRatios = { 0.5, 1, 2 };            // rpn阶段生成的anchor的缩放因子
  std::vector<float> backboneStrides = {
    4, 8, 16, 32, 64
  };  //用于计算输入图像经过backbone的每一个阶段(可能是pooling或者conv等down
      // sample操作导致feature map缩小后的尺寸,这一部分没细看)后feature图的长宽
  std::vector<std::vector<int>> backboneShapes;
  int rpnAnchorStride = 1;                          // rpn阶段生成的anchor之间的间隔
  float meanPixel[3] = { 123.7f, 116.8f, 103.9f };  //图像三个通道对应的要减去的均值
  //下面是记录数组对应的个数,貌似可用sizeof来计算.....算了不想每次都计算,总觉得以后会用到...
  int numRpnAnchorScales = 5;
  int numBackboneStrides = 5;
  int numRpnAnchorRatios = 3;

  std::vector<std::string> inputTensorNames = { "input_image_1", "input_image_meta_1",
                                                "input_anchors_1" };  //网络输入的tensor的名字,对应于模型
  std::vector<std::string> outputTensorNames = { "output_detections", "output_mrcnn_class", "output_mrcnn_bbox",
                                                 "output_mrcnn_mask", "output_rois",        " output_rpn_class",
                                                 "output_rpn_bbox" };  //网络输出的tensor的名字,对应于模型
  std::vector<std::string> classNames = {
    "BG",           "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",
    "train",        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter",
    "bench",        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",
    "elephant",     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",
    "tie",          "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",
    "cup",          "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",
    "sandwich",     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",
    "cake",         "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",
    "tv",           "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",
    "oven",         "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",
    "scissors",     "teddy bear",     "hair drier", "toothbrush"
  };
  MaskRCNNParameters()
  {
    batchSize = imagePerGPU * GPUCount;
    resizedImageShape = { imageMaxDim, imageMaxDim, 3 };
    computeBackboneShapes();
  }

private:
  void computeBackboneShapes()
  {
    for (float &backboneStride : backboneStrides)
      backboneShapes.push_back({ static_cast<int>(ceil(resizedImageShape[0] / backboneStride)),
                                 static_cast<int>(ceil(resizedImageShape[1] / backboneStride)), 3 });
  }
};
class TensorFlowMaskRCNNDetector
{
public:
  explicit TensorFlowMaskRCNNDetector(const MaskRCNNParameters &parameters);

  ~TensorFlowMaskRCNNDetector();

  void LoadModel(std::string modelPath);

private:
  // void InitConfig(int input_w, int input_h);

  bool mbSession_open = false;
  std::vector<tensorflow::Tensor> mvtOutputs;
  std::vector<ImageDetectInfo> mvOutputsInfo;
  tensorflow::Session *mpSession;
  tensorflow::Session *mpSessionGPUConfig = nullptr;
  tensorflow::GraphDef mGraphDef;

private:
  void BuildTensors();
  void ComposeImageMeta();
  void GetAnchors();
  void GeneratePyramidAnchors();

  // float mImageMeta[93] = {};  // dim num from dynaslam
  int mBackboneShape[5][2];  // for backbone_shape
  int mAnchorCache[2] = {};  //用于缓存 for cache ;
  tensorflow::Tensor inputTensor;
  tensorflow::Tensor mtInputMetaDataTensor;
  tensorflow::Tensor mtInputAnchorsTensor;
  Eigen::MatrixXf mFinalBox;
  Eigen::MatrixXf mFinalBoxNorm;
  Eigen::MatrixXf mFinalboxMat;

  // TODO should be const
  MaskRCNNParameters mParameters;

public:
  cv::Mat DetectImages(std::vector<cv::Mat> &images)
  {
    assert(images.size() && (images.at(0).channels() == 1 || images.at(0).channels() == 3));
    if (images.at(0).channels() == 1)
      for (cv::Mat &image : images)
        cv::cvtColor(image, image, CV_GRAY2BGR);

    MoldInputImages(images);
    CvMatsToTensor(images, &inputTensor);

    std::vector<tensorflow::Tensor> outputTensors;
    BatchExecuteModel(inputTensor, &outputTensors);
    std::vector<ImageDetectInfo> outputVec;
    UnmoldDetections(outputTensors, outputVec);

    return cv::Mat();
  }

private:
  void MoldInputImages(std::vector<cv::Mat> &inputImages)
  {
    for (cv::Mat &image : inputImages)
    {
      std::vector<int> window;
      ResizeImage(image, window);
      cv::subtract(image, cv::Scalar(mParameters.meanPixel[0], mParameters.meanPixel[1], mParameters.meanPixel[2]),
                   image);
    }
  }

  void ResizeImage(cv::Mat &image, std::vector<int> &window)
  {
    int minDim = mParameters.imageMinDim;
    int maxDim = mParameters.imageMaxDim;
    bool padding = mParameters.imagePadding;
    int w = image.cols;
    int h = image.rows;
    window = { 0, 0, h, w };

    float scale = 1.0;
    if (minDim > 0)
      scale = std::max(1.f, static_cast<float>(minDim) / std::min(h, w));
    if (maxDim > 0)
    {
      float imageMax = std::max(h, w);
      if (round(imageMax * scale) > maxDim)
        scale = static_cast<float>(maxDim) / imageMax;
    }

    if (scale != 1)
    {
      cv::resize(image, image, cv::Size(), scale, scale);
    }
    if (padding)
    {
      w = image.cols;
      h = image.rows;
      int topPad = std::floor((maxDim - h) / 2);
      int bottomPad = maxDim - h - topPad;
      int leftPad = std::floor((maxDim - w) / 2);
      int rightPad = maxDim - w - leftPad;
      cv::copyMakeBorder(image, image, topPad, bottomPad, leftPad, rightPad, cv::BORDER_CONSTANT, cv::Scalar(0));
      window = { topPad, leftPad, h + topPad, w + leftPad };
    }
  }

  void CvMatsToTensor(const std::vector<cv::Mat> &inputImages, tensorflow::Tensor *outputTensor)
  {
    // TODO need to minus mean?
    // TODO what is this 4?
    // TODO channels?
    auto tensorData = outputTensor->tensor<float, 4>();
    for (size_t b = 0; b < inputImages.size(); b++)  //遍历图像张数
      for (int r = 0; r < tensorData.dimension(1); r++)
        for (int c = 0; c < tensorData.dimension(2); c++)  //遍历列数
          for (int chn = 0; chn < tensorData.dimension(3); chn++)
            tensorData(b, r, c, chn) = inputImages[b].at<cv::Vec3b>(r, c)[chn];
  }

  void BatchExecuteModel(tensorflow::Tensor &inputTensor, std::vector<tensorflow::Tensor> *outputTensors)
  {
    tensorflow::Status status_run = mpSession->Run({ { mParameters.inputTensorNames[0], inputTensor },
                                                     { mParameters.inputTensorNames[1], mtInputMetaDataTensor },
                                                     { mParameters.inputTensorNames[2], mtInputAnchorsTensor } },
                                                   { mParameters.inputTensorNames[0], mParameters.outputTensorNames[1],
                                                     mParameters.outputTensorNames[2], mParameters.outputTensorNames[3],
                                                     mParameters.inputTensorNames[4], mParameters.outputTensorNames[5],
                                                     mParameters.outputTensorNames[6] },
                                                   {}, { outputTensors });

    if (!status_run.ok())
      throw std::runtime_error("Error: Run failed!\n status: " + status_run.ToString() + "\n");
  }

  void UnmoldDetections(std::vector<tensorflow::Tensor> &inputTensors, std::vector<ImageDetectInfo> &outputVec)
  {
    tensorflow::Tensor &detections_tensor = inputTensors[0];
    auto boxes_tensor = detections_tensor.tensor<float, 3>();

    for (int imgNum = 0; imgNum < boxes_tensor.dimension(0); imgNum++)
    {
      std::vector<Eigen::RowVectorXf> noZeroRow;
      for (int boxNum = 0; boxNum < boxes_tensor.dimension(1); boxNum++)
      {
        if (boxes_tensor(imgNum, boxNum, 4) > 0)
        {
          Eigen::RowVectorXf eachrow(boxes_tensor.dimension(2));
          eachrow << boxes_tensor(imgNum, boxNum, 0), boxes_tensor(imgNum, boxNum, 1), boxes_tensor(imgNum, boxNum, 2),
              boxes_tensor(imgNum, boxNum, 3), boxes_tensor(imgNum, boxNum, 4), boxes_tensor(imgNum, boxNum, 5);
          noZeroRow.push_back(eachrow);
        }
      }

      Eigen::MatrixXf noZeroMat(noZeroRow.size(), 6);
      for (int r = 0; r < noZeroRow.size(); r++)
      {
        noZeroMat.row(r) = noZeroRow[r];
      }
      Eigen::MatrixXf boxMat(noZeroMat.rows(), 4);
      Eigen::MatrixXf classSoresMat(noZeroMat.rows(), 2);
      boxMat.block(0, 0, boxMat.rows(), 4) = noZeroMat.block(0, 0, noZeroMat.rows(), 4);
      classSoresMat.block(0, 0, classSoresMat.rows(), 2) = noZeroMat.block(0, 4, classSoresMat.rows(), 2);

      // get the window in image meta
      auto metaTensor = mtInputMetaDataTensor.tensor<float, 2>();
      Eigen::MatrixXf windowMat(1, 4);
      Eigen::MatrixXf scale_rMat(1, 4);
      windowMat << metaTensor(0, 7), metaTensor(0, 8), metaTensor(0, 7), metaTensor(0, 8);
      scale_rMat << metaTensor(0, 9) - metaTensor(0, 7), metaTensor(0, 10) - metaTensor(0, 8),
          metaTensor(0, 9) - metaTensor(0, 7), metaTensor(0, 10) - metaTensor(0, 8);
      // denorm_boxes
      Eigen::MatrixXf shiftNorm_rMat(1, 4);
      Eigen::MatrixXf scaleNorm_rMat(1, 4);
      shiftNorm_rMat << 0, 0, 1, 1;
      scaleNorm_rMat << metaTensor(0, 1) - 1, metaTensor(0, 2) - 1, metaTensor(0, 1) - 1, metaTensor(0, 2) - 1;
      Eigen::MatrixXf shiftNormMat = shiftNorm_rMat.colwise().replicate(boxMat.rows());
      Eigen::MatrixXf scaleNormMat = scaleNorm_rMat.colwise().replicate(boxMat.rows());
      boxMat = boxMat.cwiseProduct(scaleNormMat);
      boxMat = boxMat + shiftNormMat;
      mFinalboxMat = boxMat;
      // std::cout<<"final box mat is "<<finalboxMat<<std::endl;
      struct ImageDetectInfo imageDetectInfoTmp;
      for (int i = 0; i < mFinalboxMat.rows(); i++)
      {
        struct BoxInfo boxInfoTmp;
        boxInfoTmp.y1 = (int)(mFinalboxMat(i, 0));
        boxInfoTmp.x1 = (int)(mFinalboxMat(i, 1));
        boxInfoTmp.y2 = (int)(mFinalboxMat(i, 2));
        boxInfoTmp.x2 = (int)(mFinalboxMat(i, 3));
        boxInfoTmp.classId = (int)(classSoresMat(i, 0));
        boxInfoTmp.scores = classSoresMat(i, 1);
        boxInfoTmp.boxNum = i;
        imageDetectInfoTmp.detectInfo.push_back(boxInfoTmp);
      }
      imageDetectInfoTmp.imageNum = imgNum;
      outputVec[imgNum] = imageDetectInfoTmp;

      // outputsInfo.push_back(imageDetectInfoTmp);
    }
  }
};

}  // namespace tf_mask_rcnn_detector
#endif