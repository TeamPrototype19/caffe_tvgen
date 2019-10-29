import sys
import caffe
from PIL import Image
import numpy as np

pimg = Image.open('lena_256.png')
rimg = pimg.resize((256,256))
nimga = np.array(rimg).reshape(1,256,256,3).transpose(0,3,1,2)

caffe.set_mode_cpu()
net = caffe.Net("ResRBXRB5.prototxt", "ResRBX_sim02-RB5-07_ep05_final.caffemodel", 0)
print net.forward_all(**{"data":nimga})
