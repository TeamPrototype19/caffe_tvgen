#include <vector>

#include "caffe/util/tvgen.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

#ifdef TVGEN_EN
  std::ofstream tvi, tvw, tvb, tvo;
  string tvi_fname = "tv_" + this->layer_param().name() + "_i.dat";
  string tvw_fname = "tv_" + this->layer_param().name() + "_w.dat";
  string tvb_fname = "tv_" + this->layer_param().name() + "_b.dat";
  string tvo_fname = "tv_" + this->layer_param().name() + "_o.dat";
  tvi.open( tvi_fname.c_str(), std::ios::binary );
  tvw.open( tvw_fname.c_str(), std::ios::binary );
  tvb.open( tvb_fname.c_str(), std::ios::binary );
  tvo.open( tvo_fname.c_str(), std::ios::binary );
  //tvgen << "name = " << this->layer_param().name() << "\n";
  //tvgen << "num_ = " << this->num_ << "\n";
  //tvgen << "bottom_dim_ = " << this->bottom_dim_ << "\n";
  //tvgen << "top_dim     = " << this->top_dim_ << "\n";
#endif

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    for (int n = 0; n < this->num_; ++n) {

#ifdef TVGEN_EN
      {
        // IFM data dump
        //const Dtype* bottom_data = bottom[i]->cpu_data();
        const Dtype* weight_data = this->blobs_[0]->cpu_data();
        tvi.write( (char*)(bottom_data + n * this->bottom_dim_) , sizeof( Dtype ) * this->bottom_dim_ );
        for(int g = 0; g < this->group_; g++)
          tvw.write( (char*)(weight_data + g * this->weight_offset_), sizeof( Dtype ) * this->weight_offset_ );
        if( this->bias_term_ ) {
          const Dtype* bias_data = this->blobs_[1]->cpu_data();
          tvb.write( (char*)bias_data, sizeof( Dtype ) * this->num_output_);
        }
      }
#endif

      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }

#ifdef TVGEN_EN
      {
        // IFM data dump
        //const Dtype* top_data = top[i]->cpu_data();
        tvo.write( (char*)(top_data + n * this->top_dim_) , sizeof( Dtype ) * this->top_dim_ );
      }
#endif

    }
  }

#ifdef TVGEN_EN
  tvi.close();
  tvw.close();
  tvb.close();
  tvo.close();
#endif


}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
