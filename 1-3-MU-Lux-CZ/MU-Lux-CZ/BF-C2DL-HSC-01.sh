#!/bin/bash
# stores result for the "BF-C2DL-HSC" sequences

rm -rf '../BF-C2DL-HSC/01_RES'
python3 predict_dataset.py --name='BF-C2DL-HSC' --sequence='01' --no-viz
python3 create_tracking.py --path='../BF-C2DL-HSC/01_RES' --output_path='../BF-C2DL-HSC/01_RES' 

