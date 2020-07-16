#!/bin/sh
# Requierments python3.5
cd SourceCode
python3 eval_SegNetLSTM_ISBI.py -m ../model_PhC-C2DL-PSC.ckpt --data_root_dir '../..' --tmp_output_dir '..' --output_dir '../..' -s 1 -d 'PhC-C2DL-PSC' --multi_split --min_size 20 --max_size 1000


