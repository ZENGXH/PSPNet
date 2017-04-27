#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/data_layer.hpp"

#include "caffe/fseg/fseg_data_transformer.hpp"  
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/fseg/fseg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
FSEGDataLayer<Dtype>::~FSEGDataLayer<Dtype>() {
  this->StopInternalThread();
  // this->JoinPrefetchThread();
}

template <typename Dtype>
void FSEGDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  FSEGTransformationParameter transform_param = this->layer_param_.fseg_transform_param();
  const int new_height = transform_param.new_height();
  const int new_width  = transform_param.new_width();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int label_type = this->layer_param_.image_data_param().label_type();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  CHECK(transform_param.has_crop_size_x() && transform_param.has_crop_size_y())
    << "Must either specify crop_size or both crop_height and crop_width.";

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
    if (label_type != ImageDataParameter_LabelType_NONE) {
      iss >> segfn;
    }
    lines_.push_back(std::make_pair(imgfn, segfn));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder+lines_[lines_id_].first, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  const int channels = cv_img.channels();
  // image
  int crop_width = transform_param.crop_size_x();
  int crop_height = transform_param.crop_size_y();

  int height = crop_height; 
  int width = crop_width;
  // resize, no cropping 
  if(!(crop_width > 0 && crop_height > 0)){
    height = cv_img.rows;
    width = cv_img.cols;
  }

  // img
  top[0]->Reshape(batch_size, channels, height, width);
  this->transformed_data_.Reshape(batch_size, channels, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
  }

  //label
  top[1]->Reshape(batch_size, 1, height, width);
  this->transformed_label_.Reshape(batch_size, 1, height, width);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, height, width);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
}


template <typename Dtype>
void FSEGDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
// void FSEGDataLayer<Dtype>::InternalThreadEntry() {
void FSEGDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  FSEGTransformationParameter transform_param = this->layer_param_.fseg_transform_param();
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  // 
  CHECK(batch->data_.count() && batch->label_.count());
  CHECK(this->transformed_data_.count());

  const int batch_size = this->layer_param_.fseg_data_param().batch_size(); 
  const int crop_width = transform_param.crop_size_x();
  const int crop_height = transform_param.crop_size_y();
  const int new_height = transform_param.new_height();
  const int new_width  = transform_param.new_width();
  const int label_type = this->layer_param_.fseg_data_param().label_type();
  const int ignore_label = transform_param.ignore_label();
  const bool is_color  = this->layer_param_.fseg_data_param().is_color();
  const string root_folder = this->layer_param_.fseg_data_param().root_folder();
  const int lines_size = lines_.size();

  // reshape data and label
  vector<int> top_shape;//  = this->data_transformer_->InferBlobShape(cv_img);
  top_shape.push_back(batch_size);
  top_shape.push_back(3);
  top_shape.push_back(crop_height);
  top_shape.push_back(crop_width);
  batch->data_.Reshape(top_shape);

  top_shape[1] = this->transformed_label_.channels();
  batch->label_.Reshape(top_shape);

  VLOG(8) << "reshap as "  << batch->data_.shape_string() << " & " << batch->label_.shape_string();

  string image_name = lines_[lines_id_].first;
  VLOG(3) << "linesid " << lines_id_ << "/" << lines_.size();
 
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  
  for (int item_id = 0; item_id < batch_size; item_id++) {
      timer.Start();
      CHECK_GT(lines_.size(), lines_id_);
      CHECK(this->output_labels_);
      const int offset_data = batch->data_.offset(item_id);
      const int offset_label = batch->label_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset_data);
      this->transformed_label_.set_cpu_data(prefetch_label + offset_label);

      image_name = root_folder + lines_[lines_id_].first;
      // LOG(INFO) << "lines_id " << lines_id_  << "/" << lines_.size() 
      // << image_name;
      
      // transform and counting
      read_time += timer.MicroSeconds();
      timer.Start();
      // load img
      vector<cv::Mat> cv_img_seg;
      cv_img_seg.push_back(ReadImageToCVMat(image_name, is_color));
      CHECK(cv_img_seg[0].data) << "Fail to load seg: " << image_name;

      if (label_type == ImageDataParameter_LabelType_PIXEL) {
          cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].second, false));
          CHECK(cv_img_seg[1].data) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
      }else if (label_type == ImageDataParameter_LabelType_IMAGE) {
          const int label = atoi(lines_[lines_id_].second.c_str());
          cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
                  CV_8UC1, cv::Scalar(label));
          cv_img_seg.push_back(seg);      
      }else {
          cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
                  CV_8UC1, cv::Scalar(ignore_label));
          cv_img_seg.push_back(seg);
      }

      this->fseg_data_transformer_->TransformImgAndSeg(cv_img_seg[0], cv_img_seg[1], &(this->transformed_data_), &(this->transformed_label_));
      trans_time += timer.MicroSeconds();
      lines_id_++;
      if (lines_id_ >= lines_.size()) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          lines_id_ -= lines_.size();
          if (this->layer_param_.fseg_data_param().shuffle()) {
              ShuffleImages();
          }
      }
  }

  batch_timer.Stop();
  timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(FSEGDataLayer);
REGISTER_LAYER_CLASS(FSEGData);

}  // namespace caffe
