#!/bin/sh
# Requierments python3.5
cd SourceCode
python3 eval_SegNetLSTM_ISBI.py -m ../model_DIC-C2DH-HeLa.ckpt --data_root_dir '../..' --tmp_output_dir '..' --output_dir '../..' -s 2 -d 'DIC-C2DH-HeLa' --multi_split



