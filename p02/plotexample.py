from  skimage import io
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from scipy.signal import convolve2d

img = io.imread("heart-1776746_960_720.jpg")
gimg = np.mean(img, axis=2)
print ("dimensiones", img.shape, "max", np.max(img), "min", np.min(img))
plt.scatter(320,60, marker="x", s=200, linewidth=5, c="b")
plt.scatter(150,230, marker="x", s=200, linewidth=5, c="g")
print ("pixel at blue marker ", gimg[60,320])
print ("pixel at green marker", gimg[230,150])
plt.grid() # remove gridlines
#Show the normal Image
plt.imshow(gimg, cmap = plt.cm.Greys_r, vmin=0, vmax=255)
portion = gimg[50:200,300:350]
portiondark = portion/2
fig = plt.figure(figsize=(3,3))
fig.add_subplot(121); plt.grid();
plt.imshow(portion, cmap=plt.cm.Greys_r, vmin=0, vmax=255)
fig.add_subplot(122); plt.grid();
plt.imshow(portiondark, cmap=plt.cm.Greys_r, vmin=0, vmax=255)
fig2 = plt.figure(figsize=(4,5))
#fig2.add_subplot(121)
plt.grid()
plt.imshow(img)
plt.show()
type(img)
