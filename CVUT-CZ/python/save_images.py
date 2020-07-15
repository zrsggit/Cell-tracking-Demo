import os
import argparse
import ast
import sys
from dataset import get_isbi_filenames,save_images_for_tracker
    
  
  
parser = argparse.ArgumentParser(description='Cell detection')
parser.add_argument('--dataset_root', required=True, type=str,help='Directory with the dataset (it contains subdirectories 01, 02 and optionally 01_GT and 02_GT)') 
parser.add_argument('--images_idx', required=True, type=str,help='Dictionary with image ids. The keys are datasets (01 or 02), the values lists with three digit indices. Example: {"01":["002","005"],"02":["006","007"]}') 
parser.add_argument('--output_dir', required=True, type=str,help='Output directory')
args = parser.parse_args()
print("==================================args=============================")
print(args)
print("================================end args===========================")

if not os.path.isdir(args.dataset_root):
  raise Exception("Unable to load images from "+args.dataset_root+": not a directory")

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)
if not os.path.isdir(args.output_dir):
  raise Exception("Unable to save results to "+args.output_dir+": not a directory")

  

#Generate filenames of images in the dataset_root dataset
image_filenames,gt_filenames=get_isbi_filenames(args.dataset_root,ast.literal_eval(args.images_idx))
save_images_for_tracker(image_filenames,args)

