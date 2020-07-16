#!/bin/sh
# Requierments python3.5
python3 Inference3DSlice.py --gpu_id 0 --model_path './Models/Fluo-C3DH-A549-SIM/' --sequence_path '../Fluo-C3DH-A549-SIM/02' --output_path '../Fluo-C3DH-A549-SIM/02_RES' --edge_dist 2 --min_cell_size 0 --max_cell_size 1000000000000 --fov 10 --centers_sigmoid_threshold 0.4 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --one_object --save_intermediate --save_intermediate_path ./Fluo-C3DH-A549-SIM/02



