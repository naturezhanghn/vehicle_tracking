from shapely.geometry import box 

# make some rectangles (for demonstration purposes and intersect with each other) 
rect1 = box(0,0,5,2) 
rect2 = box(0.5,0.5,3,3) 
rect3 = box(1.5,1.5,4,6) 

rect_list = [rect1, rect2, rect3] 

# find intersection of rectangles (probably a more elegant way to do this) 
for rect in rect_list[1:]: 
    rect1 = rect1.intersection(rect) 
intersection = rect1 

from matplotlib import pyplot as plt 
from matplotlib.collections import PatchCollection 
from matplotlib.patches import Polygon 

# plot the rectangles before and after merging 

patches = PatchCollection([Polygon(a.exterior) for a in rect_list], facecolor='red', linewidth=.5, alpha=.5) 
intersect_patch = PatchCollection([Polygon(intersection.exterior)], facecolor='red', linewidth=.5, alpha=.5) 

# make figure 
fig, ax = plt.subplots(1,2, subplot_kw=dict(aspect='equal')) 
ax[0].add_collection(patches, autolim=True) 
ax[0].autoscale_view() 
ax[0].set_title('separate polygons') 
ax[1].add_collection(intersect_patch, autolim=True) 
ax[1].set_title('intersection = single polygon') 
ax[1].set_xlim(ax[0].get_xlim()) 
ax[1].set_ylim(ax[0].get_ylim()) 
plt.show() 
plt.savefig('pic.jpg',dpi=360) 