#!/bin/sh
# Requierments python3.5
python3 Inference2D.py --gpu_id 0 --model_path './Models/Fluo-N2DL-HeLa/' --sequence_path '../Fluo-N2DL-HeLa/02' --output_path '../Fluo-N2DL-HeLa/02_RES' --edge_dist 2 --min_cell_size 10 --max_cell_size 1000000 --fov 10 --centers_sigmoid_threshold 0.4 --min_center_size 5 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-N2DL-HeLa/02


