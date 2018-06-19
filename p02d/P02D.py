#libraries
import sys
sys.path.append("code")
from haar import *
import os
from skimage import io
import numpy as np

#import the images
base_dir = "faces/"
dir_positives = base_dir+"positives"
dir_negatives = base_dir+"negatives"
positive_filenames = os.listdir(dir_positives)
negative_filenames = os.listdir(dir_negatives)

pos_imgs = []
neg_imgs = []

for i in positive_filenames:
    img = io.imread(dir_positives+"/"+i).astype(int)
    pos_imgs.append(img)

for i in negative_filenames:
    img = io.imread(dir_negatives+"/"+i).astype(int)
    neg_imgs.append(img)
    
#print ("loaded", len(pos_imgs), "positive images and", len(neg_imgs), "negative images")
# Set the haar features
haar_1 = [ {"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 1.0},
           {"op": "sub", "topleft_row_rel": 0.5, "topleft_col_rel": 0.0, "height_rel": 0.5, "width_rel": 1.0}]

haar_2 = [ {"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.0, "height_rel": 1.0, "width_rel": 0.5},
           {"op": "sub", "topleft_row_rel": 0.0, "topleft_col_rel": 0.5, "height_rel": 1.0, "width_rel": 0.5}]

haar_3 = [ {"op": "add", "topleft_row_rel": 0.0,   "topleft_col_rel": 0.0, "height_rel": 0.3, "width_rel": 1.0},
           {"op": "add", "topleft_row_rel": 0.7,   "topleft_col_rel": 0.0, "height_rel": 0.3, "width_rel": 1.0},
           {"op": "sub", "topleft_row_rel": 0.3,   "topleft_col_rel": 0.0, "height_rel": 0.4, "width_rel": 1.0}]

haar_4 = [ {"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.0, "height_rel": 1.0, "width_rel": 0.3},
           {"op": "add", "topleft_row_rel": 0.0, "topleft_col_rel": 0.7, "height_rel": 1.0, "width_rel": 0.3},
           {"op": "sub", "topleft_row_rel": 0.0, "topleft_col_rel": 0.3, "height_rel": 1.0, "width_rel": 0.4}]

show_haar_features([haar_1, haar_2, haar_3, haar_4])
# get the harr
def make_haar_dataset_for_MNIST(haar_features, d, N):
    M  = d[0].reshape(28,28)
    Mi = get_integral(M)
    num_feats = len(get_haar_features(haar_features, M, Mi, nb_scales=N, nb_shifts=N))
    print ("number of haar features", num_feats)
    d_haar = np.zeros((d.shape[0], num_feats))
    for i in range(d.shape[0]):
        if i%(d.shape[0]/20)==0:
            print ("{0:2.0f}% completed".format(i*1./d.shape[0] * 100))
            
        M  = d[i].reshape(28,28)
        Mi = get_integral(M)
        res = get_haar_features(haar_features, M, Mi, nb_scales=N, nb_shifts=N)
        d_haar[i] = res             
    return d_haar
# charge the harr features
haar_features = [ haar_1, haar_2, haar_3, haar_4 ]
def make_haar_dataset_for_faces(haar_features, positive_images, negative_images, N):
    datos     = []
    etiquetas = []
    # positive
    i =0
    while i<len(positive_images):
        datos.append(get_haar_features(haar_features, positive_images[i], get_integral(positive_images[i]), nb_scales=N, nb_shifts=N))
        etiquetas.append(1)
        i+=1
    j = i
    r = j+len(negative_images)
    while j<len(negative_images):
        datos.append(get_haar_features(haar_features, negative_images[j], get_integral(negative_images[j]), nb_scales=N, nb_shifts=N))
        etiquetas.append(0)
        j+=1
    datos = np.array(datos)
    etiquetas = np.array(etiquetas)
    return datos, etiquetas
        
d,c = make_haar_dataset_for_faces(haar_features, pos_imgs, neg_imgs, 6)
print (d.shape, c.shape)

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

est = RandomForestClassifier(n_estimators=50)
sc = cross_val_score(est, d, c, cv=10)
print(np.mean(sc), np.std(sc))