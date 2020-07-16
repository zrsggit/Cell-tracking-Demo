#!/bin/sh
# Requierments python3.5
python3 Inference3DSlice.py --gpu_id 0 --model_path './Models/Fluo-C3DL-MDA231/' --sequence_path '../Fluo-C3DL-MDA231/02' --output_path '../Fluo-C3DL-MDA231/02_RES' --edge_dist 2 --min_cell_size 100 --max_cell_size 10000 --fov 10 --centers_sigmoid_threshold 0.4 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-C3DL-MDA231/02



