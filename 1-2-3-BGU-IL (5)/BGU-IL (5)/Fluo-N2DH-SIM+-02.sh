#!/bin/sh
# Requierments python3.5 or greater
# Python requirements in requests.txt
python3 Inference2D.py --gpu_id 0 --model_path ./Models/Fluo-N2DH-SIM+/ --sequence_path '../Fluo-N2DH-SIM+/02' --output_path '../Fluo-N2DH-SIM+/02_RES' --edge_dist 10  --edge_thresh=0.3 --min_cell_size 100 --max_cell_size 1000000 --fov 0 --centers_sigmoid_threshold 0.8 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-N2DH-SIM+/02



