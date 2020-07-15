import os
import torch
from torch.utils.data import DataLoader
from nets import batch_images_labels
from dataset import is_torch_none
from scipy.misc import imsave
from PIL import Image
import numpy as np

def eval(model,dataset,args,save_images=True,print_output=True):
  dataloader=DataLoader(dataset, batch_size=1,num_workers=args.num_workers,shuffle=False)
  model.eval()
  use_cuda=torch.cuda.is_available() and args.cuda
  model.cuda() if use_cuda else model.cpu()
  if len(dataloader)>0:
    for i,batch in enumerate(dataloader):
      image,labels=batch_images_labels(batch,use_cuda)
      outputs=model(image)
      if print_output:
        print(dataset.get_filename_basis(i),end=', ',flush=True)
      vis=outputs.cpu().data.squeeze().numpy()
      if save_images:
        seg=vis[0,:,:]
        dist=vis[1,:,:]
        dist[dist>args.dt_bound]=args.dt_bound
        dist/=args.dt_bound
        imsave(os.path.join(args.output_dir,"seg"+dataset.get_filename_basis(i)+".png"),Image.fromarray(seg))
        imsave(os.path.join(args.output_dir,"bound"+dataset.get_filename_basis(i)+".png"),Image.fromarray(dist))
  if print_output:
      print("")

