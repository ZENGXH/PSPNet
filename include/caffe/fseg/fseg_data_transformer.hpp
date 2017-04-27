#ifndef CAFFE_FSEG_DATA_TRANSFORMER_HPP
#define CAFFE_FSEG_DATA_TRANSFORMER_HPP

#include <vector>
#include <random>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/half/half.hpp"

namespace caffe {

template <typename Dtype>
class FSEGDataTransformer {
 public:
  explicit FSEGDataTransformer(const FSEGTransformationParameter& param, Phase phase);
  virtual ~FSEGDataTransformer() {}


  void InitRand(int rand_seed);

  struct AugmentSelection {
    bool flip;
    float degree;
    Size crop;
    float scale;
  };

  void TransformImgAndSeg(Mat& cv_img, Mat& cv_seg, Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob);

  void visualize(const cv::Mat& img, const cv::Mat& mask, AugmentSelection& as);
  bool augmentation_flip(const Mat& img_src, const Mat& label_src, Mat& img_aug, Mat& label_aug);
  float augmentation_rotate(const Mat& img_src, const Mat& label_src, Mat& img_aug, Mat& label_aug);
  float augmentation_scale(const Mat& img_src, const Mat& label_src, Mat& img_aug, Mat& label_aug);
  Size augmentation_croppad(const Mat& img_src, const Mat& label_src, Mat& img_aug, Mat& label_aug);

 protected:
  bool onPlane(cv::Point p, Size img_size); 
  FSEGTransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
