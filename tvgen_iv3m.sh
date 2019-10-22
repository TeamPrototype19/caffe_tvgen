# remove tv folder
rm -rf tv_iv3

# set PYTHONPATH to use pycaffe
export PYTHONPATH=${PWD}/python:$PYTHONPATH

# run inference python script
cd examples/inceptionv3
python inference.py

# move test vector to folder 'tv'
mkdir -p tv_iv3
mv tv_*.dat tv_iv3
mv tv_iv3 ../../
