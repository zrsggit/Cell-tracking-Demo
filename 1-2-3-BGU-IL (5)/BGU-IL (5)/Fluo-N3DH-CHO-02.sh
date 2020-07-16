#!/bin/sh
# Requierments python3.5
python3 Inference3DSlice.py --gpu_id 0 --model_path './Models/Fluo-N3DH-CHO/' --sequence_path '../Fluo-N3DH-CHO/02' --output_path '../Fluo-N3DH-CHO/02_RES' --edge_dist 2 --min_cell_size 400 --max_cell_size 1000000 --fov 10 --centers_sigmoid_threshold 0.4 --min_center_size 50 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-N3DH-CHO/02



