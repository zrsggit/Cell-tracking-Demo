#!/bin/sh
# Requierments python3.5
cd SourceCode
python3 eval_SegNetLSTM_ISBISIM.py --cpu -m ../model_SIM02.ckpt --data_root_dir '../..' --tmp_output_dir '..' --output_dir '../..' -s 2 -d 'Fluo-N2DH-SIM+'




