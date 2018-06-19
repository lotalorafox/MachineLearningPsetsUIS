from IPython.display import Image
import sys
import numpy as np
sys.path.append("code")
from haar import *
# EX 1
haar_1 = [ {"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 1.0},
           {"op": "sub", "topleft_row_rel": 0.5, "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 1.0}]

haar_2 = [{"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.0, "height_rel": 1.0, "width_rel": 0.5},
          {"op": "sub", "topleft_row_rel": 0.0, "topleft_col_rel": 0.5, "height_rel": 1.0, "width_rel": 0.5}]

haar_3 = [ {"op": "add", "topleft_row_rel": 0.0,   "topleft_col_rel": 0.0, "height_rel": 0.3, "width_rel": 1.0},
           {"op": "add", "topleft_row_rel": 0.7,   "topleft_col_rel": 0.0, "height_rel": 0.3, "width_rel": 1.0},
           {"op": "sub", "topleft_row_rel": 0.3,   "topleft_col_rel": 0.0, "height_rel": 0.4, "width_rel": 1.0}]

haar_4 = [ {"op": "add", "topleft_row_rel": 0.0,   "topleft_col_rel": 0.0, "height_rel": 1.0, "width_rel": 0.3},
           {"op": "add", "topleft_row_rel": 0.0,   "topleft_col_rel": 0.7, "height_rel": 1.0, "width_rel": 0.3},
           {"op": "sub", "topleft_row_rel": 0.0,   "topleft_col_rel": 0.3, "height_rel": 1.0, "width_rel": 0.4}]

haar_5 = [ {"op": "add", "topleft_row_rel": 0.0,   "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 0.5},
           {"op": "add", "topleft_row_rel": 0.5,   "topleft_col_rel": 0.5, "height_rel": 0.5, "width_rel": 0.5},
           {"op": "sub", "topleft_row_rel": 0.0,   "topleft_col_rel": 0.5, "height_rel": 0.5, "width_rel": 0.5},
           {"op": "sub", "topleft_row_rel": 0.5,   "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 0.5}]

haar_features = [haar_1, haar_2, haar_3, haar_4, haar_5 ]
M = create_sample_matrix(10)
print (M)
print (extract_haar(haar_1, M))
print (extract_haar(haar_2, M))
print (extract_haar(haar_3, M))
print (extract_haar(haar_4, M))
print (extract_haar(haar_5, M))

# EX 2

def get_submatrix_sum_using_integral(image, integral, topleft_row, topleft_col, height, width):
    a = integral[topleft_row-1,topleft_col-1] if topleft_row-1 >=0 and topleft_col-1>=0 else 0
    b = integral[topleft_row-1,topleft_col+width-1] if topleft_row-1>=0 and topleft_col+width-1>=0 else 0
    c = integral[topleft_row+height-1,topleft_col+width-1] if topleft_row+height-1>=0 and topleft_col+width-1>=0 else 0
    d = integral[topleft_row+height-1,topleft_col-1] if topleft_row+height-1>=0 and topleft_col-1>=0 else 0
    result = a+c-b-d
    return result
# prove
M = create_sample_matrix(10)
Mi = get_integral(M)
print (M)
print (Mi)

print (get_submatrix_sum_using_integral(M, Mi, 1,2,4,6))
print (np.sum(M[1:5,2:8]))
print (get_submatrix_sum_using_integral(M, Mi, 0,4,1,3))
print (np.sum(M[0:1,4:7]))
print (get_submatrix_sum_using_integral(M, Mi, 1,6,4,3))
print (np.sum(M[1:5,6:9]))
print (get_submatrix_sum_using_integral(M, Mi, 0,0,3,1))
print (np.sum(M[0:3,0:1]))

print (extract_haar(haar_1, M, M))
print (extract_haar(haar_1, M, Mi, submatrix_sum_function=get_submatrix_sum_using_integral))


# EX 3
mnist = np.loadtxt("data.csv", delimiter=",")
d=mnist[:,1:785]
c=mnist[:,0]
haar_1 = [ {"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 1.0},
           {"op": "sub", "topleft_row_rel": 0.5, "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 1.0}]

haar_2 = [ {"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.0, "height_rel": 1.0, "width_rel": 0.5},
           {"op": "sub", "topleft_row_rel": 0.0, "topleft_col_rel": 0.5, "height_rel": 1.0, "width_rel": 0.5}]

haar_features = [ haar_1, haar_2 ]


def make_haar_dataset_for_MNIST(haar_features, d, N):
    M  = d[0].reshape(28,28)
    Mi = get_integral(M)
    num_feats = len(get_haar_features(haar_features, M, Mi, nb_scales=N, nb_shifts=N))
    print "number of haar features", num_feats
    d_haar = np.zeros((d.shape[0], num_feats))
    for i in range(d.shape[0]):
        if i%(d.shape[0]/20)==0:
            print "{0:2.0f}% completed".format(i*1./d.shape[0] * 100)
            
        M  = d[i].reshape(28,28)
        Mi = get_integral(M)
        res = get_haar_features(haar_features, tem, intem, nb_scales=N, nb_shifts=N)
        d_haar[i] = res             
    return d_haar