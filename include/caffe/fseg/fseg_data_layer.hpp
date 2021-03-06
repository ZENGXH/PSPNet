#ifndef CAFFE_FSEG_DATA_LAYER_HPP_
#define CAFFE_FSEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/fseg/fseg_data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FSEGDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit FSEGDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~FSEGDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  FSEGTransformationParameter fseg_transform_param_;
  shared_ptr<FSEGDataTransformer<Dtype> > fseg_data_transformer_;

  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  // vector<std::pair<std::string, vector<vector<float> > > > lines_;
  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;

  // Blob<Dtype> prefetch_data_;
  // Blob<Dtype> prefetch_label_;
  // Blob<Dtype> transformed_data_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  Blob<Dtype> transformed_label_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
