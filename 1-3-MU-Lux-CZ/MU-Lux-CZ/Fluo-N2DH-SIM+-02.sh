#!/bin/bash
# stores result for the "Fluo-N2DH-SIM+" sequences

rm -rf '../Fluo-N2DH-SIM+/02_RES'
python3 predict_dataset.py --name='Fluo-N2DH-SIM+' --sequence='02' --no-viz
python3 create_tracking.py --path='../Fluo-N2DH-SIM+/02_RES' --output_path='../Fluo-N2DH-SIM+/02_RES' 

