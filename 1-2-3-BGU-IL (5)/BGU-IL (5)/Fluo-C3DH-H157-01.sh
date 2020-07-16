#!/bin/sh
# Requierments python3.5
python3 Inference3DSlice.py --gpu_id 0 --model_path './Models/Fluo-C3DH-H157/' --sequence_path '../Fluo-C3DH-H157/01' --output_path '../Fluo-C3DH-H157/01_RES' --edge_dist 2 --min_cell_size 20000 --max_cell_size 100000000 --fov 0 --centers_sigmoid_threshold 0.1 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-C3DH-H157/01



