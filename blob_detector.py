import numpy as np
import cv2
import matplotlib.pyplot as plt
"""
parameter chosen
image       threshold   n       initial     k       d       blob number  evaluate
dots        0.035       6       1.6         1.414   0.5     7            perfect
butterfly   0.2         5       1.6         1.414   2       6            perfect
dice        0.13        6       1.6         1.414   2       35           not bad
bonus       0.13        7       0.8         1.414   2       126          bad
"""
# configurations
name = "butterfly.jpeg" # relative path for the image
threshold = 0.2         # threshold for maximum peak
n = 5                   # number of scale
initial_sigma = 1.6     # first scale
k = 2**0.5              # exponential increase in sigma
d = 2                   # define overlap as radius(1)+radius(2) >= d*distance (d decrease blob merge)


# convolve the image with different LoG filter(normalized)
all_img = []
for i in range(n):   # scale space
    sigma = initial_sigma*k**i
    size  = 6*sigma  # filter size
    y,x   = np.ogrid[-size//2 : size//2+1 , -size//2 : size//2+1]
    LoG_y = np.exp( -(y*y/(2*sigma**2)) )
    LoG_x = np.exp( -(x*x/(2*sigma**2)) )
    LoG_filter = ( -(2*sigma**2) + (x*x + y*y) ) * (LoG_x*LoG_y) * ( 1/(2*np.pi*sigma**4) ) # normalized LoG

    image = cv2.imread(name, 0)
    img   = image/255
    img   = cv2.filter2D(img, -1, LoG_filter)
    img   = np.square(img)
    all_img.append(img)                         # (9, 540, 640) <class 'list'>
all_img_np = np.array([i for i in all_img])     # (9, 540, 640) <class 'numpy.ndarray'>


# find maximum peak
coord = []      # store blob coordinates
(h,w) = image.shape
for i in range(1, h):
    for j in range(1, w):
        slice = all_img_np[ : , i-1:i+2, j-1:j+2] # perform slice image(9, 3, 3) <class 'numpy.ndarray'>
        maxi = np.max(slice) # finding maximum
        if maxi >= threshold:
            z_sigma_index, y_height, x_width = np.unravel_index(np.argmax(slice),slice.shape)  # zyx correspond to slice maximum
            coord.append((j+x_width-1, i+y_height-1, z_sigma_index, maxi))    # inference original image coordinates
coord = list(set(coord))      # discard repeat coordinates

coord_list = []      # from tuple to list(x, y, sigma_index, maximum_peak)
for i in range(len(coord)):
    coord_list.append([coord[i][0],coord[i][1], coord[i][2], coord[i][3]])


# dealing with overlap
for i in range(len(coord_list)):                     # coord_list[i]
    if coord_list[i] == -1:                           
        continue
    for j in range(1, len(coord_list)-(i+1)+1):      # coord_list[i+j]
        if coord_list[i+j] == -1:
            continue
        distance = ( (coord_list[i][0]-coord_list[i+j][0])**2 + (coord_list[i][0]-coord_list[i+j][0])**2 )**0.5  #distance
        if initial_sigma*(k**coord_list[i][2] + k**coord_list[i+j][2])*1.414 >= d*distance:   # radius(1)+radius(2) >= d*distance
            if coord_list[i][3] < coord_list[i+j][3]:
                coord_list[i] = coord_list[i+j]
            coord_list[i+j] = -1
discard_overlap_coord = list(filter((-1).__ne__, coord_list))


print("{}".format(name))
# print(discard_overlap_coord)         # (x, y, sigma_index, maximum_peak)
print("Blob number = ", len(discard_overlap_coord))


# plot the blobs
fig, ax = plt.subplots()
ax.imshow(image, cmap = 'gray', aspect = 'auto')
for blob in discard_overlap_coord:
    _x, _y, _z, _m = blob
    ax.add_patch(plt.Circle((_x, _y), initial_sigma*k**_z*1.414, color='red', fill=False))   #radius = sigma*1.414 
plt.title(name)
plt.savefig("{}(blobs).jpeg".format(name))
plt.show()

