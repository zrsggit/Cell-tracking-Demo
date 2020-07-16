#!/bin/sh
# Requierments python3.5
python3 Inference3DSlice.py --gpu_id 0 --model_path './Models/Fluo-N3DH-SIM+/' --sequence_path '../Fluo-N3DH-SIM+/02' --output_path '../Fluo-N3DH-SIM+/02_RES' --edge_dist 2 --min_cell_size 100 --max_cell_size 1000000 --fov 0 --centers_sigmoid_threshold 0.4 --min_center_size 50 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ./Fluo-N3DH-SIM+/02



