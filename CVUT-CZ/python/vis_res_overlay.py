import os
import sys
import glob
import scipy.misc
#python3 vis_res_overlay.py input_output_directory

def sgn(n):
  if n<0:
    return(-1)
  if n>0:
    return(1)
  else:
    return(0)
class Palette:
  def __init__(self):
    self.data=[(0,0,0),
               (255,0,0),
               (0,255,0),
               (0,0,255),
               (255,255,0),
               (255,0,255),
               (0,255,255),
               (255,255,255),
               (255,127,0),
               (255,0,127),
               (0,255,127),
               (127,255,0),
               (127,0,255),
               (0,127,255),
               (255,127,127),
               (127,255,127),
               (127,127,255),
               (255,255,127),
               (255,127,255),
               (127,255,255)]
  def __call__(self,i):
    if i==0:
      return(self.data[i])
    else:
      return(self.data[(i % (len(self.data)-1))+1])




input_output_directory=sys.argv[1]
masks=sorted(glob.glob(os.path.join(input_output_directory,"mask*.tif")))
images=sorted(glob.glob(os.path.join(input_output_directory,"img*.png")))
print("Found "+str(len(masks))+" masks and "+str(len(images))+" images")
p=Palette()
for i in range(len(masks)):
  mask=scipy.misc.imread(masks[i],flatten=False)
  vis=scipy.misc.imread(images[i],mode='RGB')
  for j in range(1,vis.max()+1):
    #for c in range(vis.shape[2]):
    overlay=[0.15*x for x in p(j)]
    vis[mask==j]=0.85*vis[mask==j]+overlay
  out_filename="vis"+str(i).zfill(3)+".png"
  print(out_filename, end=', ',flush=True)
  scipy.misc.imsave(os.path.join(input_output_directory,out_filename),vis)
print("")
