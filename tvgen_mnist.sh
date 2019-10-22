./build/tools/caffe test -model ./examples/mnist/lenet_train_test.prototxt -weights ./examples/mnist/lenet_iter_1000.caffemodel
mkdir -p tv_mnist
mv tv_*.dat tv_mnist
