#!/bin/bash
# stores result for the "Fluo-N2DH-SIM+" sequences

rm -rf '../DIC-C2DH-HeLa/02_RES'
python3 predict_dataset.py --name='DIC-C2DH-HeLa' --sequence='02' --no-viz
python3 create_tracking.py --path='../DIC-C2DH-HeLa/02_RES' --output_path='../DIC-C2DH-HeLa/02_RES' 

