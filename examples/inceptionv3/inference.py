import sys
import caffe
from PIL import Image
import numpy as np

pimg = Image.open('cat.jpg')
rimg = pimg.resize((299,299))
nimga = np.array(rimg).reshape(1,299,299,3).transpose(0,3,1,2)

caffe.set_mode_cpu()
net = caffe.Net("deploy_inception-v3-merge.prototxt", "inception-v3-merge.caffemodel", 0)
print net.forward_all(**{"data":nimga})
