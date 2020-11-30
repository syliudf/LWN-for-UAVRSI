import matplotlib.pyplot as plt	# 绘图
import matplotlib.image as mpimg # 显示图像
import numpy as np
# from torch._C import dtype, uint8	# 处理数据
import scipy
from scipy.stats import mode	# 统计操作
from imageio import imwrite
import os
import os.path
from scipy import misc 
root = "./data/UDD6/val/gt"
l=os.listdir(root)
# l.remove("show.py")
l.remove("label")

def get_Vaihingen_label():
    return np.asarray(
                        [
                            
                                [107, 142,  35],  # vegetation
                                [102, 102, 156],  # building
                                [128,  64, 128],  # road
                                
                                [  0,   0, 142],  # vehicle
                                [ 70,  70,  70],  # roof
                                [  0,   0,   0]   # other 
                            
                        ]
                        )

def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
        the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
        in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 6
    label_colours = get_Vaihingen_label()
    # print(label_mask)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    # print(r.size())
    rgb[:, :, 0] = r 
    rgb[:, :, 1] = g 
    rgb[:, :, 2] = b 
    return rgb 

for i in l:
    p=mpimg.imread(os.path.join(root, i))
    l,h=p.shape
    p=(p*255)
    p=np.array(p, dtype='uint8')
    p=np.around(p)
    
    m = decode_segmap(p)
    # print(m)
    # print(m)
    # m=np.zeros((l,h,3), dtype=uint8)
    # m[p==0]=[107,142,35]
    # m[p==1]=[102,102,156]
    # m[p==2]=[128,64,128]
    # m[p==3]=[0,0,142]
    # m[p==4]=[70,70,70]
    # m[p==5]=[0,0,0]
    imwrite(os.path.join(root,"label",i), m) 

