import os
import argparse
import ast
import sys
import torch
from dataset import Cells,get_isbi_filenames,is_iterable,save_images_for_tracker
from nets import Model,load_model_from_file
from eval import eval
    
  
  
parser = argparse.ArgumentParser(description='Cell detection')
parser.add_argument('--cuda',  action="store_true",help='Use GPU if available')
parser.add_argument('--dataset_root', required=True, type=str,help='Directory with the dataset (it contains subdirectories 01, 02 and optionally 01_GT and 02_GT)') 
parser.add_argument('--images_idx', required=True, type=str,help='Dictionary with image ids. The keys are datasets (01 or 02), the values lists with three digit indices. Example: {"01":["002","005"],"02":["006","007"]}') 
parser.add_argument('--output_dir', required=True, type=str,help='Output directory')
parser.add_argument('--resolution_levels', required=True, type=str,help='List of resolutions in the pipeline. 0 means the original resolution, -1 downscale by factor 2, -2 downscale by factor 4 etc.')
parser.add_argument('--dt_bound',default=9,type=int,help='Bound for the distance transform')
parser.add_argument('--model_file', required=True, type=str,help='Filename of loaded model')
parser.add_argument('--num_workers', default=0, type=int,help='Number of workers for the dataloader. If -1, data augmentation is done on GPU')
args = parser.parse_args()
print("==================================args=============================")
print(args)
print("================================end args===========================")
args.cuda=torch.cuda.is_available() and args.cuda
augmentation_cuda=args.cuda if args.num_workers==-1 else False
if args.num_workers<0:
  args.num_workers=0
print("CUDA:",args.cuda,augmentation_cuda,sep=" ")


if not os.path.isdir(args.dataset_root):
  raise Exception("Unable to load images from "+args.dataset_root+": not a directory")

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)
if not os.path.isdir(args.output_dir):
  raise Exception("Unable to save results to "+args.output_dir+": not a directory")

  
#Load the network
model=load_model_from_file(args.model_file)

#Generate filenames of images in the dataset_root dataset
image_filenames,gt_filenames=get_isbi_filenames(args.dataset_root,ast.literal_eval(args.images_idx))


save_images_for_tracker(image_filenames,args)


#Run the network
dataset=Cells(image_filenames,[],args.dt_bound,ast.literal_eval(args.resolution_levels))
eval(model,dataset,args)

