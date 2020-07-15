import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import is_torch_none, is_iterable


def get_nested_structure_level(x):
  if not is_iterable(x):
    return(0)
  else:
    return(get_nested_structure_level(x[0])+1)

def batch_images_labels(batch,use_cuda):
  batch_images=[Variable(x) for x in batch[0]]
  batch_labels=None
  if use_cuda:
    for i in range(len(batch_images)):
      batch_images[i]=batch_images[i].cuda()
  return(batch_images,batch_labels)

class LastActivation(nn.Module):
  def __init__(self):
    super(LastActivation, self).__init__()
    self.sigmoid=nn.Sigmoid()
    self.relu=nn.ReLU()
  def forward(self, input):
    return(torch.cat((self.sigmoid.forward(input[:,0:1,:,:]),self.relu.forward(input[:,1:,:,:])),dim=1))
    
class Model(nn.Module):
  def __init__(self,structure,batchnorm_momentum):
    super(Model, self).__init__()
    self.batchnorm_momentum=batchnorm_momentum
    self.structure=structure
    if get_nested_structure_level(self.structure)==1:
      self.structure=[self.structure]
    print("net structure: ",  self.structure)
    
    self.segments=nn.ModuleList()
    in_channels=1
    cumulative_out_channels=0
    for i,(numof_layers,out_channels,kernel_size) in enumerate(structure):
      cumulative_out_channels+=out_channels
      if kernel_size % 2 != 1:
        raise Exception("Size of the receptive field must be an odd number")
      self.segments.append(nn.ModuleList())
      if i>0:
        in_channels=cumulative_out_channels
      for l in range(numof_layers):
        print("Layer",l,"segment",i,"channels",cumulative_out_channels,sep=" ")
        self.segments[i].append(
          nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1)/2)),
            nn.Conv2d(in_channels, cumulative_out_channels, kernel_size=kernel_size),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(cumulative_out_channels,momentum=batchnorm_momentum)
            )
          )
        in_channels=cumulative_out_channels
      #Last or upsampling layer
      if i==len(structure)-1:
        self.segments[i].append(
          nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size-1)/2)),
            nn.Conv2d(in_channels, 2, kernel_size=kernel_size),
            LastActivation()))    
      else:
        self.segments[i].append(
          nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(cumulative_out_channels,momentum=batchnorm_momentum)))
  
  def forward(self,x):
    for i,segment in enumerate(self.segments):
      if i==0:
        out=x[i]
      else:
        out=torch.cat((out,x[i].repeat(1,self.structure[i][1],1,1)),dim=1)
      for layer in segment:
        out=layer(out)
    return(out)  

  
def load_model_from_file(filename):
  checkpoint = torch.load(filename,map_location=lambda storage, loc: storage)
  model=Model(checkpoint["structure"],checkpoint["batchnorm_momentum"])
  model.load_state_dict(checkpoint["state_dict"])
  return(model)





