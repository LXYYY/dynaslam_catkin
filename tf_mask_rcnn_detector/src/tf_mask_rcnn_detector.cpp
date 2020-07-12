#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <utility>
#include "tf_mask_rcnn_detector/tf_mask_rcnn_detector.hpp"
namespace tf_mask_rcnn_detector
{
TensorFlowMaskRCNNDetector::TensorFlowMaskRCNNDetector(const MaskRCNNParameters &parameters) : mParameters(parameters)
{
  ComposeImageMeta();
  GeneratePyramidAnchors();
}

TensorFlowMaskRCNNDetector::~TensorFlowMaskRCNNDetector()
{
  std::vector<tensorflow::Tensor>().swap(mvtOutputs);
  std::vector<ImageDetectInfo>().swap(mvOutputsInfo);
  if (mbSession_open)
  {
    mpSession->Close();
    std::cout << "Session closed." << std::endl;
    mbSession_open = false;
  }
}

void TensorFlowMaskRCNNDetector::LoadModel(std::string modelPath)
{
  std::cout << "Loading model from " << modelPath << std::endl;

  tensorflow::SessionOptions opts;
  tensorflow::Status status = NewSession(opts, &mpSession);

  tensorflow::Status status_load = ReadBinaryProto(tensorflow::Env::Default(), modelPath, &mGraphDef);
  if (!status_load.ok())
  {
    throw std::runtime_error("Error: Loading model failed! " + status_load.ToString());
  }

  tensorflow::Status status_create = mpSession->Create(mGraphDef);
  if (!status_create.ok())
  {
    throw std::runtime_error("Error: Creating graph in session failed! " + status_create.ToString());
  }

  mbSession_open = true;
  std::cout << "Successfully created session and load graph!" << std::endl;
}

// void TensorFlowMaskRCNNDetector::InitConfig(int inputW, int inputH)
// {
//   mParameters.inputTnsW = inputW;
//   mParameters.inputTnsH = inputH;
//   ComposeImageMeta();
//   GetAnchors();
// }

void TensorFlowMaskRCNNDetector::BuildTensors()
{
  inputTensor =
      tensorflow::Tensor(tensorflow::DT_FLOAT, { mParameters.batchSize, mParameters.resizedImageShape[0],
                                                 mParameters.resizedImageShape[1], mParameters.resizedImageShape[2] });
}

void TensorFlowMaskRCNNDetector::ComposeImageMeta()
{
  std::vector<float> imageMeta;
  imageMeta.insert(imageMeta.begin(), 0);
  imageMeta.insert(imageMeta.begin(), mParameters.inputImageShape.begin(), mParameters.inputImageShape.end());
  cv::Mat tImage = cv::Mat::zeros(cv::Size(mParameters.inputImageShape[0], mParameters.inputImageShape[1]), CV_8UC3);
  std::vector<int> tWindow;
  ResizeImage(tImage, tWindow);
  imageMeta.insert(imageMeta.begin(), tWindow.begin(), tWindow.end());
  std::vector<int> tClassIds(mParameters.numInputImage, 0);
  imageMeta.insert(imageMeta.begin(), tClassIds.begin(), tClassIds.end());

  mtInputMetaDataTensor =
      tensorflow::Tensor(tensorflow::DT_FLOAT, { mParameters.batchSize, static_cast<long long>(imageMeta.size()) });
  auto inputMetadataTensorData = mtInputMetaDataTensor.flat<float>().data();
  inputMetadataTensorData[0] = 0;
  for (int i = 0; i < mParameters.batchSize; i++)
    std::copy_n(imageMeta.begin(), imageMeta.size(), inputMetadataTensorData + i * imageMeta.size());
}

void TensorFlowMaskRCNNDetector::GeneratePyramidAnchors()
{
  int nFinalBoxesRows = 0;  //用于统计五个mParameters.rpnAnchorScales尺度对应的所有boxes的行数,可以先不看这个
  std::vector<Eigen::MatrixXf> vemBoxes;
  for (int i = 0; i < mParameters.rpnAnchorScales.size(); i++)
  {
    int scale = mParameters.rpnAnchorScales.at(i);
    int numRpnAnchorRatios = mParameters.rpnAnchorRatios.size();
    int numRpnAnchorScales = mParameters.rpnAnchorScales.size();
    Eigen::RowVectorXf evScales(1);  //遍历并且临时存储mParameters.rpnAnchorScales[5]={32, 64, 128, 256,
    Eigen::VectorXf evRatios(mParameters.rpnAnchorRatios.size());
    Eigen::MatrixXf emScales = Eigen::MatrixXf(numRpnAnchorRatios, 1);  //();
    Eigen::MatrixXf emRatios = Eigen::MatrixXf(numRpnAnchorRatios, 1);  //();
    Eigen::MatrixXf emHeights;  //=Eigen::MatrixXf(mParameters.numRpnAnchorRatios, 1);//();
    Eigen::MatrixXf emWidths;   //=Eigen::MatrixXf(mParameters.numRpnAnchorRatios, 1);//();
    evScales(0) = mParameters.rpnAnchorScales.at(i);

    //构造np.array(ratios)
    for (int nR = 0; nR < numRpnAnchorRatios;nR++)
      evRatios(nR) = mParameters.rpnAnchorRatios.at(nR);
    for (int nRc = 0; nRc < emRatios.cols(); nRc++)
      emRatios.col(nRc) << evRatios;
    for (int nS = 0; nS < emScales.rows(); nS++)
      emScales.row(nS) << evScales;

    emHeights = emScales.cwiseQuotient(emRatios.cwiseSqrt());
    emWidths = emScales.cwiseProduct(emRatios.cwiseSqrt());

    int step = mParameters.rpnAnchorStride;
    int low = 0;
    int hightY = mParameters.backboneShapes.at(i).at(0);
    int hightX = mParameters.backboneShapes.at(i).at(1);
    Eigen::RowVectorXf evShiftX, evShiftY;
    int realsizeY = ((hightY - low) / step);
    int realsizeX = ((hightX - low) / step);
    evShiftX.setLinSpaced(realsizeX, low, low + step * (realsizeX - 1));
    evShiftY.setLinSpaced(realsizeY, low, low + step * (realsizeY - 1));
    evShiftX *= mParameters.backboneStrides.at(i);
    evShiftY *= mParameters.backboneStrides.at(i);

    Eigen::MatrixXf emShiftX(evShiftY.cols(), evShiftX.cols());
    Eigen::MatrixXf emShiftY(evShiftY.cols(), evShiftX.cols());
    for (int nrow = 0; nrow < emShiftX.rows(); nrow++)
      emShiftX.row(nrow) = evShiftX;
    for (int ncol = 0; ncol < emShiftY.cols(); ncol++)
      emShiftY.col(ncol) = evShiftY;
    Eigen::RowVectorXf emHeightsFlat(
        Eigen::Map<Eigen::VectorXf>(emHeights.data(), emHeights.rows() * emHeights.cols()));
    Eigen::RowVectorXf emWidthsFlat(Eigen::Map<Eigen::VectorXf>(emWidths.data(), emWidths.rows() * emWidths.cols()));

    emShiftX.transposeInPlace();
    emShiftY.transposeInPlace();
    Eigen::RowVectorXf emShiftYFlat(Eigen::Map<Eigen::VectorXf>(emShiftY.data(), emShiftY.rows() * emShiftY.cols()));
    // Eigen::RowVectorXf
    // shifts_xMatFlat(Eigen::Map<Eigen::VectorXf>(shifts_xMat.data(),shifts_xMat.rows()*shifts_xMat.cols(),Eigen::ColMajor));
    Eigen::RowVectorXf emShiftXFlat(Eigen::Map<Eigen::VectorXf>(emShiftX.data(), emShiftX.rows() * emShiftX.cols()));
    Eigen::MatrixXf emBoxWidths = Eigen::MatrixXf(emShiftXFlat.cols(), emWidthsFlat.cols());    //();
    Eigen::MatrixXf emBoxCenterX = Eigen::MatrixXf(emShiftXFlat.cols(), emWidthsFlat.cols());   //();
    Eigen::MatrixXf emBoxHeights = Eigen::MatrixXf(emShiftYFlat.cols(), emHeightsFlat.cols());  //();
    Eigen::MatrixXf emBoxCenterY = Eigen::MatrixXf(emShiftYFlat.cols(), emHeightsFlat.cols());  //();
    for (int nrow = 0; nrow < emBoxWidths.rows(); nrow++)
    {
      emBoxWidths.row(i) = emWidthsFlat;
      emBoxHeights.row(i) = emHeightsFlat;
    }
    for (int ncol = 0; ncol < emBoxHeights.cols(); ncol++)
    {
      emBoxCenterX.col(ncol) = emShiftXFlat;
      emBoxCenterY.col(ncol) = emShiftYFlat;
    }
    Eigen::MatrixXf emY1 = emBoxCenterY - emBoxHeights * 0.5;
    Eigen::MatrixXf emX1 = emBoxCenterX - emBoxWidths * 0.5;
    Eigen::MatrixXf emY2 = emBoxCenterY + emBoxHeights * 0.5;
    Eigen::MatrixXf emX2 = emBoxCenterX + emBoxWidths * 0.5;
    emY1.transposeInPlace();
    emX1.transposeInPlace();
    emY2.transposeInPlace();
    emX2.transposeInPlace();
    Eigen::RowVectorXf emY1Flat(Eigen::Map<Eigen::VectorXf>(emY1.data(), emY1.rows() * emY1.cols()));
    Eigen::RowVectorXf emX1Flat(Eigen::Map<Eigen::VectorXf>(emX1.data(), emX1.rows() * emX1.cols()));
    Eigen::RowVectorXf emY2Flat(Eigen::Map<Eigen::VectorXf>(emY2.data(), emY2.rows() * emY2.cols()));
    Eigen::RowVectorXf emX2Flat(Eigen::Map<Eigen::VectorXf>(emX2.data(), emX2.rows() * emX2.cols()));
    Eigen::MatrixXf emBoxes(emY1.rows() * emY1.cols(), 4);  //注意这里的boxes不是python代码里面对应的boxes
    emBoxes.col(0) = emY1Flat;
    emBoxes.col(1) = emX1Flat;
    emBoxes.col(2) = emY2Flat;
    emBoxes.col(3) = emX2Flat;
    vemBoxes.push_back(emBoxes);
    nFinalBoxesRows += emBoxes.rows();
  }
  mFinalBox = Eigen::MatrixXf(nFinalBoxesRows, 4);
  int nBeginX = 0;
  for (int i = 0; i < vemBoxes.size(); i++)
  {
    mFinalBox.block(nBeginX, 0, vemBoxes[i].rows(), vemBoxes[i].cols()) = vemBoxes[i];
    nBeginX += vemBoxes[i].rows();
  }

  Eigen::MatrixXf scaleMat_1r(1, mFinalBox.cols());
  Eigen::MatrixXf shiftMat_1r(1, mFinalBox.cols());
  scaleMat_1r << float(mParameters.resizedImageShape[1] - 1), float(mParameters.resizedImageShape[0] - 1),
      float(mParameters.resizedImageShape[1] - 1), float(mParameters.resizedImageShape[0] - 1);
  shiftMat_1r << 0.f, 0.f, 1.f, 1.f;
  //因为上一步得到是scaleMat_1r,shiftMat_1r是向量,接下来创建对应的矩阵,该矩阵与finalBox有相同的
  //形状
  Eigen::MatrixXf scaleMat =
      scaleMat_1r.colwise().replicate(mFinalBox.rows());  //通过重复与finalBox同样的行数构建scaleMat
  Eigen::MatrixXf shiftMat = shiftMat_1r.colwise().replicate(mFinalBox.rows());  //同上
  Eigen::MatrixXf tmpMat = mFinalBox - shiftMat;   // finalBox对应位置元素减去偏移量
  mFinalBoxNorm = tmpMat.cwiseQuotient(scaleMat);  // finalBox对应位置元素处以scale
  //至此完成了python代码中的boxes(mFinalBoxNorm),下一步把mFinalBoxNorm矩阵弄成Eigen::tensor类型的inputAnchorsTensor
  //再通过inputAnchorsTensor填充到tensorflow::tensor类型的mtInputAnchorsTensor构建最后送入模型的anchor boxes

  mtInputAnchorsTensor =
      tensorflow::Tensor(tensorflow::DT_FLOAT, { mParameters.batchSize, mFinalBoxNorm.rows(),
                                                 mFinalBoxNorm.cols() });  //初始化mtInputAnchorsTensor
  // float *p=mtInputAnchorsTensor.flat<float>().data();
  //通mFinalBoxNorm矩阵构建Eigen::tensor类型的inputAnchorsTensor
  Eigen::Tensor<float, 3> inputAnchorsTensor(1, mFinalBoxNorm.rows(), mFinalBoxNorm.cols());
  for (int i = 0; i < mFinalBoxNorm.rows(); i++)
  {
    Eigen::Tensor<float, 1> eachrow(mFinalBoxNorm.cols());  //用于临时存储mFinalBoxNorm矩阵的的每一行
    //把mFinalBoxNorm矩阵的一行放进eachrow
    eachrow.setValues(
        { mFinalBoxNorm.row(i)[0], mFinalBoxNorm.row(i)[1], mFinalBoxNorm.row(i)[2], mFinalBoxNorm.row(i)[3] });
    //把eachrow放进inputAnchorsTensor的每一行
    inputAnchorsTensor.chip(i, 1) = eachrow;
  }
  //把inputAnchorsTensor赋值给mtInputAnchorsTensor,注意它们两个的类型是不同的
  auto showMap = mtInputAnchorsTensor.tensor<float, 3>();
  for (int b = 0; b < showMap.dimension(0); b++)
  {
    for (int r = 0; r < showMap.dimension(1); r++)
    {
      for (int c = 0; c < showMap.dimension(2); c++)
      {
        showMap(b, r, c) = inputAnchorsTensor(0, r, c);  //这里为0是因为
        //我的batch里面的图片都是同样尺寸的,所以它们最终的anchor boxes都是一样,
        //只要赋值一个就行了,建议batch里面图片尺寸都是一样的,这样好处理
      }
    }
  }
}

void TensorFlowMaskRCNNDetector::GetAnchors()
{
  // tensorflow::Tensor flatTensor(tensorflow::DT_FLOAT,{1,2,4,5});
  // auto f=flatTensor.shape();

  // compute_mBackboneShape
  // Cache anchors and reuse if image shape is the same
  //计算图像mBackboneShapes
  if (mParameters.resizedImageShape[1] != mAnchorCache[0] ||
      mParameters.resizedImageShape[0] != mAnchorCache[1])  //如果之前计算过就不用重新计算了,相当于cache
  {
    for (int i = 0; i < mParameters.numBackboneStrides; i++)
    {
      mBackboneShape[i][0] = ceil(mParameters.resizedImageShape[1] / mParameters.backboneStrides[i]);
      mBackboneShape[i][1] = ceil(mParameters.resizedImageShape[0] / mParameters.backboneStrides[i]);
    }
    // std::vector<tensorflow::Tensor> anchors;
    std::vector<Eigen::MatrixXf> vemBoxes;  // eigen矩阵类型容器,用于存储anchors
    // vemBoxes作用类似于python代码中的
    /*
       for i in range(len(scales)):
            anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                    feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)



     */

    int finalBoxesRows = 0;  //用于统计五个mParameters.rpnAnchorScales尺度对应的所有boxes的行数,可以先不看这个

    // generate_pyramid_anchors //生成不同尺度(配置参数中是5个)的anchor
    for (int j = 0; j < mParameters.numRpnAnchorScales; j++)
    {
      // generate_anchors

      // Get all combinations of scales and ratios
      Eigen::RowVectorXf scalesVec(1);  //遍历并且临时存储mParameters.rpnAnchorScales[5]={32, 64, 128, 256,
                                        // 512}的每个元素,主要给scalesMat赋值用
      Eigen::VectorXf ratiosVec(mParameters.numRpnAnchorRatios);
      Eigen::MatrixXf scalesMat = Eigen::MatrixXf(mParameters.numRpnAnchorRatios, 1);  //();
      Eigen::MatrixXf ratiosMat = Eigen::MatrixXf(mParameters.numRpnAnchorRatios, 1);  //();
      Eigen::MatrixXf heightsMat;  //=Eigen::MatrixXf(mParameters.numRpnAnchorRatios, 1);//();
      Eigen::MatrixXf widthsMat;   //=Eigen::MatrixXf(mParameters.numRpnAnchorRatios, 1);//();

      //以下步骤主要是实现python中的
      /*
       scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
       */

      scalesVec(0) = (mParameters.rpnAnchorScales[j]);

      //构造np.array(ratios)
      for (int i = 0; i < mParameters.numRpnAnchorRatios; i++)
      {
        ratiosVec(i) = mParameters.rpnAnchorRatios[i];
      }
      for (int i = 0; i < ratiosMat.cols(); i++)
      {
        ratiosMat.col(i) << ratiosVec;
      }

      //构造np.array(scales)
      // std::cout<<"scalesMat is <<"<<scalesMat.cols()<<std::endl;
      for (int i = 0; i < scalesMat.rows(); i++)
      {
        scalesMat.row(i) << scalesVec;
      }

      //构造heights,widths,这两个在python里面是长度为3的向量,但为了后面的点乘等操作换成了3*1的矩阵
      // python代码如下
      /*
          heights = scales / np.sqrt(ratios)
          widths = scales * np.sqrt(ratios)
       */

      // Enumerate heights and widths from scales and ratios
      heightsMat = scalesMat.cwiseQuotient(ratiosMat.cwiseSqrt());
      widthsMat = scalesMat.cwiseProduct(ratiosMat.cwiseSqrt());

      //构造shifts_x, shifts_y
      // python代码如下
      /*
      shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
      shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
      shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
       */
      // Enumerate shifts in feature space
      //先进行   shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
      //        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride

      int step = mParameters.rpnAnchorStride, low = 0, hight_y = mBackboneShape[j][0],
          hight_x = mBackboneShape[j][1];  //获取shape[0],shape[1],anchor_stride,
      Eigen::RowVectorXf shifts_y;         //行向量
      Eigen::RowVectorXf shifts_x;
      int realsize_y = ((hight_y - low) / step);
      int realsize_x = ((hight_x - low) / step);
      shifts_y.setLinSpaced(realsize_y, low, low + step * (realsize_y - 1));
      shifts_x.setLinSpaced(realsize_x, low, low + step * (realsize_x - 1));
      shifts_y *=
          mParameters.backboneStrides
              [j];  //获取feature_stride,这里的feature_stride其实是python代码中外围循环送进的参数mParameters.backboneStrides[j]
      shifts_x *=
          mParameters.backboneStrides
              [j];  //获取feature_stride,这里的feature_stride其实是python代码中外围循环送进的参数mParameters.backboneStrides[j]

      /*再进行   shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y),
      构造出最终的shifts_x,shifts_y矩阵,注意经过np.meshgrid后shifts_x,shifts_y是二维的矩阵
      */
      //构造shifts_x,shifts_y矩阵
      Eigen::MatrixXf shifts_xMat(shifts_y.cols(), shifts_x.cols()), shifts_yMat(shifts_y.cols(), shifts_x.cols());
      for (int i = 0; i < shifts_xMat.rows(); i++)
      {
        shifts_xMat.row(i) = shifts_x;
      }
      for (int i = 0; i < shifts_yMat.cols(); i++)
      {
        shifts_yMat.col(i) = shifts_y;
      }

      //进行python代码
      /*
          box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
          box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

          # Reshape to get a list of (y, x) and a list of (h, w)
          box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
          box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

          # Convert to corner coordinates (y1, x1, y2, x2)
          boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                      box_centers + 0.5 * box_sizes], axis=1)
          return boxes

       */
      // Enumerate combinations of shifts, widths, and heights
      //先进行 box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
      //      box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
      //先把heightsMat,widthsMat换成行向量方便赋值,
      Eigen::RowVectorXf heightsMatFlat(
          Eigen::Map<Eigen::VectorXf>(heightsMat.data(), heightsMat.rows() * heightsMat.cols()));
      Eigen::RowVectorXf widthsMatFlat(
          Eigen::Map<Eigen::VectorXf>(widthsMat.data(), widthsMat.rows() * widthsMat.cols()));

      /*因为上面的np.meshgrid(widths, shifts_x)
      中widths是长度为3的向量,shifts_x是二维矩阵,所以np.meshgrid(widths, shifts_x)生成的矩阵列数是widths的长度
      生成的矩阵行数是--shifts_x按照行方向平铺后的长度,假如shifts_x是2*3矩阵,那么就是6.而后面
      box_widths, box_centers_x = np.meshgrid(widths,
      shifts_x)生成的box_centers_x的行数是shifts_x的行数*列数,box_centers_x每一列是shifts_x矩阵的元素按照行方向平铺后构成的,
      但是因为eigen里面的矩阵是列优先存储,所以要在c++代码中对shifts_xMat(shift_x)进行转置,这样通过Eigen::Map映射到shifts_yMatFlat就是相当于把shifts_x矩阵的元素按照行方向平铺后构成的向量
      同理对shifts_yMat进行同样的操作得到shifts_yMatFlat.
      而box_widths,box_heights可以通过widthsMatFlat,和heightsMatFlat赋值得到,因为heightsMatFlat和box_heights可以通过widthsMatFlat
      本身是一维的向量
      */
      shifts_xMat.transposeInPlace();
      shifts_yMat.transposeInPlace();
      Eigen::RowVectorXf shifts_yMatFlat(
          Eigen::Map<Eigen::VectorXf>(shifts_yMat.data(), shifts_yMat.rows() * shifts_yMat.cols()));
      // Eigen::RowVectorXf
      // shifts_xMatFlat(Eigen::Map<Eigen::VectorXf>(shifts_xMat.data(),shifts_xMat.rows()*shifts_xMat.cols(),Eigen::ColMajor));
      Eigen::RowVectorXf shifts_xMatFlat(
          Eigen::Map<Eigen::VectorXf>(shifts_xMat.data(), shifts_xMat.rows() * shifts_xMat.cols()));
      Eigen::MatrixXf emBoxWidths = Eigen::MatrixXf(shifts_xMatFlat.cols(), widthsMatFlat.cols());    //();
      Eigen::MatrixXf emBoxCenterX = Eigen::MatrixXf(shifts_xMatFlat.cols(), widthsMatFlat.cols());   //();
      Eigen::MatrixXf emBoxHeights = Eigen::MatrixXf(shifts_yMatFlat.cols(), heightsMatFlat.cols());  //();
      Eigen::MatrixXf emBoxCenterY = Eigen::MatrixXf(shifts_yMatFlat.cols(), heightsMatFlat.cols());  //();
      for (int i = 0; i < emBoxWidths.rows(); i++)
      {
        emBoxWidths.row(i) = widthsMatFlat;
        emBoxHeights.row(i) = heightsMatFlat;
      }
      for (int i = 0; i < emBoxHeights.cols(); i++)
      {
        emBoxCenterX.col(i) = shifts_xMatFlat;
        emBoxCenterY.col(i) = shifts_yMatFlat;
      }

      // Convert to corner coordinates (y1, x1, y2, x2)
      // 'e for 's element abbreviation
      // note that ,in the bellow,matrix's element which to be add or substract, is In the corresponding position
      // python method: box_centers_y mat ,box_centers_x mat  stack to  mat A whose unit format is
      // (box_center_y'e,box_center_x'e) then reshape to [-1,2],so the result is mat whose  col format is
      // (box_center_y'e,box_center_x'e),box_sizes mat B is the same,col format is (box_height'e,box_width'e) then  A-B
      // ,A+B get the mat C,D whose col format are  respectively
      // (box_center_y'e-box_height'e,box_center_x'e-box_width'e) and
      // (box_center_y'e+box_height'e,box_center_x'e+box_width'e) then concat C and D get mat E whose col format is
      // (box_center_y'e-box_height'e,box_center_x'e-box_width'e
      // ,box_center_y'e+box_height'e,box_center_x'e+box_width'e) and that is (y1,x1,y2,x2) in eigen3,different to
      // python first we have got the matrix emBoxCenterY emBoxCenterX emBoxHeights emBoxWidths for
      // abbreviation is center_yMat,center_xMat,heightMat,widthMat center_yMat-0.5*heightMat=emY1
      // center_yMat+0.5*heightMat=emY2
      // center_xMat-0.5*widthMat=emX1
      // center_xMat+0.5*widthMat=emX2
      // then generate the matrix boxes whose col format is (emY1's e,emX1's e,emY2's e ,emX2's e),rows in the num

      //进行如下操作
      // boxes = np.concatenate([box_centers - 0.5 * box_sizes,
      // box_centers + 0.5 * box_sizes], axis=1)
      // boxes形式如[(y1, x1, y2, x2),...,...]
      Eigen::MatrixXf emY1 = emBoxCenterY - emBoxHeights * 0.5;
      Eigen::MatrixXf emX1 = emBoxCenterX - emBoxWidths * 0.5;
      Eigen::MatrixXf emY2 = emBoxCenterY + emBoxHeights * 0.5;
      Eigen::MatrixXf emX2 = emBoxCenterX + emBoxWidths * 0.5;
      emY1.transposeInPlace();
      emX1.transposeInPlace();
      emY2.transposeInPlace();
      emX2.transposeInPlace();
      Eigen::RowVectorXf emY1Flat(Eigen::Map<Eigen::VectorXf>(emY1.data(), emY1.rows() * emY1.cols()));
      Eigen::RowVectorXf emX1Flat(Eigen::Map<Eigen::VectorXf>(emX1.data(), emX1.rows() * emX1.cols()));
      Eigen::RowVectorXf emY2Flat(Eigen::Map<Eigen::VectorXf>(emY2.data(), emY2.rows() * emY2.cols()));
      Eigen::RowVectorXf emX2Flat(Eigen::Map<Eigen::VectorXf>(emX2.data(), emX2.rows() * emX2.cols()));
      Eigen::MatrixXf boxes(emY1.rows() * emY1.cols(), 4);  //注意这里的boxes不是python代码里面对应的boxes
      boxes.col(0) = emY1Flat;
      boxes.col(1) = emX1Flat;
      boxes.col(2) = emY2Flat;
      boxes.col(3) = emX2Flat;
      //到此已经完成单独一个mParameters.rpnAnchorScales[i]尺度对应的boxes了
      //下一步把它放进容器里

      vemBoxes.push_back(boxes);
      finalBoxesRows += boxes.rows();  //统计五个mParameters.rpnAnchorScales尺度对应的所有boxes的行数
                                       // break;
    }
    //以上一步得到的boxes的finalBoxesRows为行数,4为列数创建二维矩阵finalBox(对应python代码的boxes),
    //其实就是用上面所有的boxes构建形式如[(y1, x1, y2, x2),...,...]的矩阵
    mFinalBox = Eigen::MatrixXf(finalBoxesRows, 4);
    // Eigen::VectorXf a(3);
    // Eigen::VectorXf b(4);
    // Eigen::VectorXf c(7);
    //取出vemBoxes容器里面每个boxes构建最终的finalBox矩阵(对应boxes)
    //至此完成了boxes的构建
    int beginX = 0;
    for (int i = 0; i < vemBoxes.size(); i++)
    {
      // mat1.block<rows,cols>(i,j)
      //矩阵块赋值
      mFinalBox.block(beginX, 0, vemBoxes[i].rows(), vemBoxes[i].cols()) = vemBoxes[i];
      beginX += vemBoxes[i].rows();
      // tensorflow::Tensor matTensor(tensorflow::DT_FLOAT,{vemBoxes[i].rows(),vemBoxes[i].cols()});
    }

    /*get normalization finalbox
    归一化finalBox
    python代码如下:
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)
    */

    //先创建scale,shift两个向量
    Eigen::MatrixXf scaleMat_1r(1, mFinalBox.cols());
    Eigen::MatrixXf shiftMat_1r(1, mFinalBox.cols());
    scaleMat_1r << float(mParameters.resizedImageShape[1] - 1), float(mParameters.resizedImageShape[0] - 1),
        float(mParameters.resizedImageShape[1] - 1), float(mParameters.resizedImageShape[0] - 1);
    shiftMat_1r << 0.f, 0.f, 1.f, 1.f;
    //因为上一步得到是scaleMat_1r,shiftMat_1r是向量,接下来创建对应的矩阵,该矩阵与finalBox有相同的
    //形状
    Eigen::MatrixXf scaleMat =
        scaleMat_1r.colwise().replicate(mFinalBox.rows());  //通过重复与finalBox同样的行数构建scaleMat
    Eigen::MatrixXf shiftMat = shiftMat_1r.colwise().replicate(mFinalBox.rows());  //同上
    Eigen::MatrixXf tmpMat = mFinalBox - shiftMat;   // finalBox对应位置元素减去偏移量
    mFinalBoxNorm = tmpMat.cwiseQuotient(scaleMat);  // finalBox对应位置元素处以scale
    //至此完成了python代码中的boxes(mFinalBoxNorm),下一步把mFinalBoxNorm矩阵弄成Eigen::tensor类型的inputAnchorsTensor
    //再通过inputAnchorsTensor填充到tensorflow::tensor类型的mtInputAnchorsTensor构建最后送入模型的anchor boxes

    mtInputAnchorsTensor =
        tensorflow::Tensor(tensorflow::DT_FLOAT, { mParameters.batchSize, mFinalBoxNorm.rows(),
                                                   mFinalBoxNorm.cols() });  //初始化mtInputAnchorsTensor
    // float *p=mtInputAnchorsTensor.flat<float>().data();
    //通mFinalBoxNorm矩阵构建Eigen::tensor类型的inputAnchorsTensor
    Eigen::Tensor<float, 3> inputAnchorsTensor(1, mFinalBoxNorm.rows(), mFinalBoxNorm.cols());
    for (int i = 0; i < mFinalBoxNorm.rows(); i++)
    {
      Eigen::Tensor<float, 1> eachrow(mFinalBoxNorm.cols());  //用于临时存储mFinalBoxNorm矩阵的的每一行
      //把mFinalBoxNorm矩阵的一行放进eachrow
      eachrow.setValues(
          { mFinalBoxNorm.row(i)[0], mFinalBoxNorm.row(i)[1], mFinalBoxNorm.row(i)[2], mFinalBoxNorm.row(i)[3] });
      //把eachrow放进inputAnchorsTensor的每一行
      inputAnchorsTensor.chip(i, 1) = eachrow;
    }
    //把inputAnchorsTensor赋值给mtInputAnchorsTensor,注意它们两个的类型是不同的
    auto showMap = mtInputAnchorsTensor.tensor<float, 3>();
    for (int b = 0; b < showMap.dimension(0); b++)
    {
      for (int r = 0; r < showMap.dimension(1); r++)
      {
        for (int c = 0; c < showMap.dimension(2); c++)
        {
          showMap(b, r, c) = inputAnchorsTensor(0, r, c);  //这里为0是因为
          //我的batch里面的图片都是同样尺寸的,所以它们最终的anchor boxes都是一样,
          //只要赋值一个就行了,建议batch里面图片尺寸都是一样的,这样好处理
        }
      }
    }
  }
}
}  // namespace tf_mask_rcnn_detector