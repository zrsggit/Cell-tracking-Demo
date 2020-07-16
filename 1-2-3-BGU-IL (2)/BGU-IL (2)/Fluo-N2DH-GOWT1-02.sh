#!/bin/sh
# Requierments python3.5
cd SourceCode
python3 eval_SegNetLSTM_ISBI.py -m ../model_N2DH_GOWT1.ckpt --data_root_dir '../..' --tmp_output_dir '..' --output_dir '../..' -s 2 -d 'Fluo-N2DH-GOWT1' --multi_split


