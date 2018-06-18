import numpy as np
from skimage import io
import matplotlib.pyplot as plt
def convolution(A, f):
    rows = A.shape[0]-f.shape[0]+1
    cols = A.shape[1]-f.shape[1]+1
    r = np.ones((rows,cols))
    if rows<=0 or cols<=0:
        raise ValueError("filter must be smaller than matrix")
    for i in range(rows):
        for j in range(cols):
            sub = A[i:i+f.shape[0],j:j+f.shape[1]]
            v = sub*f
            r[i,j]=r[i,j]*np.sum(v)
    result = r
    return result


def convolution_sad(A, f):
    rows = A.shape[0]-f.shape[0]+1
    cols = A.shape[1]-f.shape[1]+1
    r = np.ones((rows, cols))
    if rows <= 0 or cols <= 0:
        raise ValueError("filter must be smaller than matrix")
    for i in range(rows):
        for j in range(cols):
            sub = A[i:i+f.shape[0], j:j+f.shape[1]]
            v = abs(sub-f)
            r[i, j] = r[i, j]*abs(np.sum(v))
    result = r
    return result

def min_sad(A, f):
    c = convolution_sad(A, f)
    if(A.shape[0] == f.shape[0]):
        return np.argmin(c)
    return np.min(c)

def index_min_sad_in_band(A,B,k,r,c):
    p = A[r:r+k,c:c+k]
    h = B[r:r+k,:]
    c = convolution_sad(h,p)
    return np.argmin(c)
def min_sad_matrix (A, B, k):
    cols = A.shape[1]-k+1
    rows = A.shape[0]-k+1
    r = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            r[i, j] = np.abs(index_min_sad_in_band(A, B, k, i, j) - j)
    return r
def main():
    img_l = io.imread("leaf-blue.png")
    img_r = io.imread("leaf-blue.png")
    img_l = np.mean(img_l, axis=2)
    img_r = np.mean(img_r, axis=2)
    fig=plt.figure(figsize=(10,20))
    fig.add_subplot(121)
    plt.imshow(img_l, cmap = plt.cm.Greys_r)
    fig.add_subplot(122)
    plt.imshow(img_r, cmap = plt.cm.Greys_r)
    fig2=plt.figure(figsize=(10,20))
    fig2.add_subplot(121)
    r = min_sad_matrix(img_l, img_r, 10)
    plt.imshow(r, cmap = plt.cm.Greys_r)
    w = np.copy(r)
    w[w>20]=0
    fig2.add_subplot(122)
    plt.imshow(w, cmap = plt.cm.Greys_r)
    plt.show()

if __name__ == '__main__':
    main()
