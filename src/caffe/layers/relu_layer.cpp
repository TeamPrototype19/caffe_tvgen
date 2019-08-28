#include <algorithm>
#include <vector>

#include "caffe/util/tvgen.hpp"
#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

#ifdef TVGEN_EN
  std::ofstream tvi, tvo;
  string tvi_fname = "tv_" + this->layer_param().name() + "_i.dat";
  string tvo_fname = "tv_" + this->layer_param().name() + "_o.dat";
  tvi.open( tvi_fname.c_str(), std::ios::binary );
  tvo.open( tvo_fname.c_str(), std::ios::binary );

  tvi.write( (char*) bottom[0]->cpu_data(), sizeof(Dtype) * bottom[0]->count() );
  tvi.close();
#endif


  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }

#ifdef TVGEN_EN
  tvo.write( (char*) top[0]->cpu_data(), sizeof(Dtype) * bottom[0]->count() );
  tvo.close();
#endif
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
