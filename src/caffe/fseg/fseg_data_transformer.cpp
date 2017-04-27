#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"


#include <string>
#include <vector>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
using namespace cv;
using namespace std;

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/fseg/fseg_data_transformer.hpp"

namespace caffe {

    template<typename Dtype>
        FSEGDataTransformer<Dtype>::FSEGDataTransformer(const FSEGTransformationParameter& param, Phase phase): param_(param), phase_(phase) {
            CHECK(param_.mean_value_size() == 3); 
                for (int c = 0; c < param_.mean_value_size(); ++c) {
                    mean_values_.push_back(param_.mean_value(c));
                }
        }

    template<typename Dtype>
        bool FSEGDataTransformer<Dtype>::onPlane(Point p, Size img_size) {
            if(p.x < 0 || p.y < 0) return false;
            if(p.x >= img_size.width || p.y >= img_size.height) return false;
            return true;
        }

    // subcall by Transform_bottomup
    template<typename Dtype>
        void FSEGDataTransformer<Dtype>::TransformImgAndSeg(Mat& cv_img, Mat& cv_seg, Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob) {
            Dtype* transformed_data = transformed_data_blob->mutable_cpu_data();
            Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();
            AugmentSelection as = {
                false,
                0.0,
                Size(),
                0,
            };
            // int crop_x = this->param_.crop_size_x();
            // int crop_y = this->param_.crop_size_y();
            Mat img = cv_img.clone();
            Mat mask = cv_seg.clone();
            //visualize original
            if(this->param_.visualize()) {
                visualize(img, mask);
            }

            //Start transforming
            Mat img_temp1, img_temp2, img_temp3, img_temp4; //size determined by scale
            Mat mask_temp1, mask_temp2, mask_temp3, mask_temp4; //size determined by scale

            // We only do random transform as augmentation when training.
            CHECK(this->phase_ == TRAIN);
            as.scale = augmentation_scale(img, mask, img_temp1, mask_temp1);
            if(this->param_.visualize())
                visualize(img_temp1, mask_temp1, as);

            as.degree = augmentation_rotate(img_temp1, mask_temp1, img_temp2, mask_temp2);
            if(this->param_.visualize())
                visualize(img_temp2, mask_temp2, as);

            as.crop = augmentation_croppad(img_temp2, mask_temp2, img_temp3,mask_temp3);
            if(this->param_.visualize())
                visualize(img_temp3, mask_temp3, as);

            as.flip = augmentation_flip(img_temp3, mask_temp3, img_temp4, mask_temp4);
            if(this->param_.visualize())
                visualize(img_temp4, mask_temp4, as);

            Mat img_aug = img_temp4.clone();
            Mat mask_aug = mask_temp4.clone();
            int offset = img_aug.rows * img_aug.cols;
            for (int i = 0; i < img_aug.rows; ++i) {
                for (int j = 0; j < img_aug.cols; ++j) {
                    Vec3b& rgb = img_aug.at<Vec3b>(i, j);
                    transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - mean_values_[0]);
                    transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - mean_values_[1]);
                    transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - mean_values_[2]);
                    transformed_label[i*img_aug.cols + j] = float(mask_aug.at<uchar>(i, j));
                }
            }
        }


    template<typename Dtype>
        float FSEGDataTransformer<Dtype>::augmentation_scale(const Mat& img_src, const Mat& mask_src, Mat& img_out, Mat& mask_out) {
            float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
            float scale = (this->param_.scale_max() - this->param_.scale_min()) * dice2 + this->param_.scale_min(); //linear shear into [scale_min, scale_max]
            resize(img_src, img_out, Size(), scale, scale, INTER_CUBIC);
            resize(mask_src, mask_out, Size(), scale, scale, INTER_CUBIC);

            return scale;
        }

    template<typename Dtype>
        Size FSEGDataTransformer<Dtype>::augmentation_croppad(
                const Mat& img_src, const Mat& mask_src, Mat& img_out, Mat& mask_out) {
            float dice_x =static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
            float dice_y =static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]

            int crop_x = this->param_.crop_size_x();
            int crop_y = this->param_.crop_size_y();

            float x_offset = int((dice_x - 0.5) * 2 * img_src.cols*0.5);
            float y_offset = int((dice_y - 0.5) * 2 * img_src.rows*0.5); // [-h/2, h/2]

            // int offset_left = -(center.x - (crop_x/2));
            // int offset_up = -(center.y - (crop_y/2));
            Point2f center = Point2f(img_src.rows*0.5, img_src.cols*0.5) + Point2f(x_offset, y_offset);

            img_out = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(mean_values_[0], mean_values_[1], mean_values_[2]);
            mask_out = Mat::zeros(crop_y, crop_x, CV_8UC1) + Scalar(this->param_.ignore_label);
            for(int i=0;i<crop_y;i++){
                for(int j=0;j<crop_x;j++){ //i,j on cropped
                    int coord_x_on_img = center.x - crop_x/2 + j;
                    int coord_y_on_img = center.y - crop_y/2 + i;
                    if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
                        img_out.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
                        mask_out.at<uchar>(i,j) = mask_src.at<uchar>(coord_y_on_img, coord_x_on_img);
                    }
                }
            }

            return Size(x_offset, y_offset);
        }

    template<typename Dtype>
        bool FSEGDataTransformer<Dtype>::augmentation_flip(
                const Mat& img_src, const Mat& mask_src, Mat& img_out, Mat& mask_out) {
            float dice =  static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            bool doflip = (dice <= 0.5);

            if(doflip){
                flip(img_src, img_out, 1);
                flip(mask_src, mask_out, 1);
            }else{
                img_out = img_src.clone();
                mask_out = mask_src.clone();
            }
            return doflip;
        }

    template<typename Dtype>
        float FSEGDataTransformer<Dtype>::augmentation_rotate(
                const Mat& img_src, const Mat& mask_src, Mat& img_out, Mat& mask_out) {
            float dice =  static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            float degree = (dice - 0.5) * 2 * this->param_.max_rotate_degree();
            Point2f center(img_src.cols/2.0, img_src.rows/2.0);
            Rect bbox = cv::RotatedRect(center, img_src.size(), degree).boundingRect();
            Mat R = getRotationMatrix2D(center, degree, 1.0);
            warpAffine(img_src, img_out, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));
            warpAffine(mask_src, mask_out, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(this->param_.ignore_label));
            return degree;
        }
    // end here

    template<typename Dtype>
        void FSEGDataTransformer<Dtype>::visualize(const cv::Mat& img, const cv::Mat& mask, AugmentSelection& as){
            static int count = 0;
            cv::Mat label_map = mask.clone();
            cv::applyColorMap(label_map, label_map, COLORMAP_JET);
            addWeighted(label_map, 0.5, img, 0.5, 0.0, label_map);
            char imagename[50];
            std::stringstream ss;
             ss << "Augmenting with:" << (as.flip ? "flip" : "no flip") << "; Rotate " << as.degree << " deg; scaling: "<< as.crop.height << "," << as.crop.width << " scale: " << as.scale;
             // setLabel(label_map, ss.str(), Point(0, 20));
             printf(imagename, "v_%d.jpg", count++);

             imwrite(imagename, label_map);
}
template <typename Dtype>
void FSEGDataTransformer<Dtype>::InitRand(int rng_seed) {
    rng_.reset(new Caffe::RNG(rng_seed));
}
} // namespace
