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
$PYTHON "python/main.py" --cuda --dataset_root "../Fluo-N2DL-HeLa" --images_idx '{"02":[]}' --output_dir "." --resolution_levels "[-2,-1,0]" --dt_bound 2 --model_file "python/networks/Fluo-N2DL-HeLa"

#Run the tracker
LD_LIBRARY_PATH=. ./tracker -o config "Fluo-N2DL-HeLa" "../Fluo-N2DL-HeLa/02" "./models/Fluo-N2DL-HeLa" 2 .

#Clean up
mv ./*.tif "../Fluo-N2DL-HeLa/02_RES"
mv ./res_track.txt "../Fluo-N2DL-HeLa/02_RES"
rm -f ./*.png
rm -rf ./0

