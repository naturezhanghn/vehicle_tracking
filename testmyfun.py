# import numpy as np 

# def get_variance(det_object):
#   len_no0 = np.sum( det_object[:,:,0] != 0) 
#   gray_average = np.sum(det_object[:,:,0]) / (len_no0 + 0.000001)
#   gray_variance = 0
#   gray_list = det_object[:,:,0].flatten()
#   for i in gray_list:
#     if i != 0 :
#       gray_variance += (i - gray_average)*(i - gray_average)/ (len_no0 + 0.000001)
#   return gray_variance

# a = np.array([[[10,1,1 ], [1,1,1 ], [1,1,1 ]],[[1,1,1 ], [1,1,1 ], [1,1,1 ]],[[1,1,1 ], [1,1,1 ], [1,1,1 ]]])
# b = np.array( [[[27,0,0], [0,0,0] ,[0,0,0] ] , [[0,0,0], [0,0,0] ,[0,0,0]  ] ,[[0,0,0], [0,0,0] ,[0,0,0]  ] ] )


# result = get_variance(a)
# print(result,np.var(a[:,:,0]))

from skimage.feature import hog
from skimage import io
im = io.imread('/workspace/mnt/storage/zhangziran/zhangzr4/pythontraffictracking/a1_output_40172/1.png')
print(im.shape)
normalised_blocks = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8))
print(normalised_blocks)