# remove tv folder
rm -rf tv_picasso

# set PYTHONPATH to use pycaffe
export PYTHONPATH=${PWD}/python:$PYTHONPATH

# run inference python script
cd examples/picasso
python inference.py

# move test vector to folder 'tv'
mkdir -p tv_picasso
mv tv_*.dat tv_picasso
mv tv_picasso ../../
