#!/bin/sh
# Requierments python3.5
cd SourceCode
python3 eval_SegNetLSTM_ISBI.py -m ../model_Fluo-N2DL-HeLa.ckpt --data_root_dir '../..' --tmp_output_dir '..' --output_dir '../..' -s 1 -d 'Fluo-N2DL-HeLa' --multi_split --min_size 50
