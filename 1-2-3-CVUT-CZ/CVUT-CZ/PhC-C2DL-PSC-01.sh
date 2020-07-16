#!/bin/bash

#This is a probabilistic method, the results may vary slightly if run multiple times

#Version of Python: 3
#Python packages:
#  PyTorch (0.2.0_3 or newer)
#  numpy
#  scipy
#  PIL (for scipy.misc.imread and scipy.misc.imsave)
#Dependencies for the main executable:
#  ImageMagick

#Generate detection candidates using a neural network:
PYTHON="python"
if [[ ! -z $(command -v python3) ]]; then
  PYTHON="python3"
fi
$PYTHON "python/main.py" --cuda --dataset_root "../PhC-C2DL-PSC" --images_idx '{"01":[]}' --output_dir "." --resolution_levels "[-2,-1,0]" --dt_bound 2 --model_file "python/networks/PhC-C2DL-PSC"

#Run the tracker
LD_LIBRARY_PATH=. ./tracker -o config "PhC-C2DL-PSC" "../PhC-C2DL-PSC/01" "./models/PhC-C2DL-PSC" 2 .

#Clean up
mv ./*.tif "../PhC-C2DL-PSC/01_RES"
mv ./res_track.txt "../PhC-C2DL-PSC/01_RES"
rm -f ./*.png
rm -rf ./0

