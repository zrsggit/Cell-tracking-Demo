import glob
import os
import numpy as np
import scipy
import scipy.ndimage
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn.functional import avg_pool2d




def get_isbi_filenames(root_dir,images_idx={"01":[],"02":[]}):
  images=[]
  gt=[]
  for sequence in sorted(images_idx):
    images_root=os.path.join(root_dir,sequence)
    seg_root=os.path.join(root_dir,sequence+"_GT","SEG")
    if len(images_idx[sequence])==0:
      images+=glob.glob(os.path.join(images_root,"*.tif"))
    else:
      idx=images_idx[sequence]
      images+=[os.path.join(images_root,"t"+x+".tif") for x in idx]
      gt+=[os.path.join(seg_root,"man_seg"+x+".tif") for x in idx]
  return(sorted(images),sorted(gt))

def get_isbi_dataset_intensity_min_max(dataset_name):
  retmin=0.0
  retmax=255.0
  desired_histogram_mean=-1
  if 'DIC-C2DH-HeLa' in dataset_name:
    retmin=0.0
    retmax=255.0
    desired_histogram_mean=118.0
  if 'Fluo-C2DL-MSC' in dataset_name:
    retmin=2000.0
    retmax=65535.0
  if 'Fluo-N2DH-GOWT1' in dataset_name:
    retmin=0.0
    retmax=255.0
  if 'Fluo-N2DH-SIM' in dataset_name:
    retmin=60.0
    retmax=1000.0
  if 'Fluo-N2DL-HeLa' in dataset_name:
    retmin=32768.0
    retmax=36863.0
  if 'PhC-C2DH-U373' in dataset_name:
    retmin=20.0
    retmax=200.0
    desired_histogram_mean=88.0
  if 'PhC-C2DL-PSC' in dataset_name:
    retmin=0.0
    retmax=255.0
  return(retmin,retmax,desired_histogram_mean)

def is_torch_none(object):
  ret=object is None
  if object is not None:
    try:
      l=len(object)
      ret=l<=0
    except:
      pass
  return(ret)
  
def is_iterable(x):
  try:
    _=iter(x)
    return(True)
  except TypeError:
    return(False)
    
def rescale_tensor(tensor,factor):
  if factor<0:
    return(avg_pool2d(tensor,2**(-factor)))
  else:
    return(tensor)


class Cells(Dataset):
  """Loads images, gt segmentation and gt boundary from a dataset."""
  def __init__(self,image_filenames,gt_filenames=[],dt_bound=1,resolution_levels=[0],use_cuda=False):
    self.image_filenames=image_filenames
    self.dt_bound=dt_bound
    self.use_cuda=torch.cuda.is_available() and use_cuda
    if is_iterable(resolution_levels):
      self.resolution_levels=resolution_levels
    else:
      self.resolution_levels=[0]

  def load(self,index):
    image=scipy.misc.imread(self.image_filenames[index],flatten=False).astype(float)
    gt_segmentation=None
    gt_boundary=None
    return(image,gt_segmentation,gt_boundary)
  def save(self,filename_base,image,gt_segmentation,gt_boundary,use_scipy_clever_convert=True):
    if use_scipy_clever_convert:
      scipy.misc.imsave(filename_base+"_img.png",image)
    else:
      histmin,histmax=get_isbi_dataset_intensity_min_max(self.image_filenames[0])
      int_image=np.copy(image)*255
      int_image[int_image<0]=0
      int_image[int_image>255]=255
      int_image=int_image.astype(np.uint8)
      scipy.misc.imsave(filename_base+"_img.png",int_image)
    scipy.misc.imsave(filename_base+"_gtseg.png", gt_segmentation)
    scipy.misc.imsave(filename_base+"_gtbound.png",gt_boundary)
  def get_filename_basis(self,index):
    base_name=os.path.splitext(os.path.basename(self.image_filenames[index]))[0]
    return(base_name[1:])


  def __getitem__(self,index):
    images=[None]*len(self.resolution_levels)
    image,gt_segmentation,gt_boundary=self.load(index)
    gt_labels=[]
    
    image=np.expand_dims(np.expand_dims(image,0),0)
    image=Variable(torch.from_numpy(image).float(),requires_grad=False)
    if self.use_cuda:
      image=image.cuda()
      
    for i,level in enumerate(self.resolution_levels):
      images[i]=rescale_tensor(image,level)
      images[i]=images[i]-images[i].mean()
      images[i]=images[i]/(3*images[i].std())
      images[i]=images[i].squeeze(0).data
    return(images,gt_labels)
    
  def __len__(self):
    return(len(self.image_filenames))
    
  def __debug_tensors_manual__(self,image,gt_labels,filename_base):
    image=np.squeeze(image.numpy())
    gt_segmentation=np.squeeze(gt_labels[0,:,:].numpy())
    gt_boundary=np.squeeze(gt_labels[1,:,:].numpy())
    print("Image min,max,mean:",image.min(),image.max(),image.mean(),sep=' ')
    print("gt_segmentation min,max,mean:",gt_segmentation.min(),gt_segmentation.max(),gt_segmentation.mean(),sep=' ')
    print("gt_boundary min,max,mean:",gt_boundary.min(),gt_boundary.max(),gt_boundary.mean(),sep=' ')
    image=image/8
    image=image+0.5
    self.save(filename_base,image,gt_segmentation,gt_boundary,use_scipy_clever_convert=False)

  def __debug_tensors__(self,index,filename_base):
    index/=self.len_multiplier
    images,gt_labels=self.__getitem__(index)
    self.__debug_tensors_manual__(images[-1],gt_labels,filename_base)






def save_images_for_tracker(filenames,args):
  for filename in filenames:
    im=scipy.misc.imread(filename,flatten=False).astype(float)
    min_brightness,max_brightness,desired_histogram_mean=get_isbi_dataset_intensity_min_max(args.dataset_root)
    im-=min_brightness
    im*=1.0/(max_brightness-min_brightness)
    if desired_histogram_mean>0:
      im*=(desired_histogram_mean/255.0)/im.flatten().mean()
    im[im<0.0]=0.0
    im[im>1.0]=1.0 
    
    base_name=os.path.splitext(os.path.basename(filename))[0]
    scipy.misc.imsave(args.output_dir+'/img'+base_name[1:]+".png", im)

