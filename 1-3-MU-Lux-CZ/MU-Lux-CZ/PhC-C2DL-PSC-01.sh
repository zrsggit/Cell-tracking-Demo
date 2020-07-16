#!/bin/bash
# stores result for the "PhC-C2DL-PSC" sequences

rm -rf '../PhC-C2DL-PSC/01_RES'
python3 predict_dataset.py --name='PhC-C2DL-PSC' --sequence='01' --no-viz
python3 create_tracking.py --path='../PhC-C2DL-PSC/01_RES' --output_path='../PhC-C2DL-PSC/01_RES' 

