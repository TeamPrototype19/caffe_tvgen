# remove tv folder
rm -rf tv

# set PYTHONPATH to use pycaffe
export PYTHONPATH=${PWD}/python:$PYTHONPATH

# run inference python script
cd examples/inceptionv3
python inference.py

# move test vector to folder 'tv'
mkdir -p tv
mv tv_*.dat tv
mv tv ../../
