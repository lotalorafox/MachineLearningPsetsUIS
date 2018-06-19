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
