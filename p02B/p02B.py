import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from sklearn.cross_validation import train_test_split
#EX 1
def load_mnist():
    mnist = np.loadtxt("data/mnist1.5k.cvs",delimeter=",")
    d = mnist[:,1:785]
    c = mnist[:,0]
    return d,c

#EX 2
def convolve_mnist(d,f):
    r = convolve2d(d,f,mode="valid")
    return r

#EX 3
def classifyGNB(d,c, classifier, test_size=0.2):
    d_train,d_test, c_train, c_test = cross_validation.train_test_split(d, c, test_size=test_size, random_state=np.random.randint(1,100))
    classifier.fit(d_train,c_train)
    score = classifier.score(d_test, c_test)
    return score

def classifyDT(d,c, classifier, test_size=0.2):
    d_train,d_test, c_train, c_test = cross_validation.train_test_split(d, c, test_size=test_size, random_state=np.random.randint(1,100))
    classifier.fit(d_train,c_train)
    score = classifier.score(d_test, c_test)
    return score
#EX 4
def classify_avg_GNB(d,c,classifier,repeat=10, test_size=0.2):
    values = np.ones((repeat))
    for i in range(repeat):
        values[i] = classifyGNB(d,c, classifier, test_size)
    mean_performance = np.mean(values)
    std_performance  = np.std(values)
    return mean_performance, std_performance

def classify_avg_DT(d,c,classifier,repeat=10, test_size=0.2):
    values = np.ones((repeat))
    for i in range(repeat):
        values[i] = classifyDT(d,c, classifier, test_size)
    mean_performance = np.mean(values)
    std_performance  = np.std(values)
    return mean_performance, std_performance
#EX 5
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
def NBDT_MNIST():
    d,c = load_mnist()
    g1 = GaussianNB()
    g2 = DecisionTreeClassifier()
    meanNB, stdNB = classify_avg_GNB(d,c,g1)
    meanDT, stdDT = classify_avg_DT(d,c,g2)
    return meanNB, stdNB, meanDT, stdDT

#EX 6
def NBDT_MNIST_convolution1():
    d,c = load_mnist()
    f   = np.array([[-1,1]])
    dc  = convolve_mnist(d,f)
    g1 = GaussianNB()
    g2 = DecisionTreeClassifier()
    meanNB, stdNB = classify_avg(d,c,g1)
    meanDT, stdDT = classify_avg(d,c,g2)
    return meanNB, stdNB, meanDT, stdDT

#EX 7
def NBDT_MNIST_convolution2():
    d,c = load_mnist()
    f   = np.ones((4,4))
    dc  = convolve_mnist(d,f)
    g1 = GaussianNB()
    g2 = DecisionTreeClassifier()
    meanNB, stdNB = classify_avg(d,c,g1)
    meanDT, stdDT = classify_avg(d,c,g2)

    return meanNB, stdNB, meanDT, stdDT
# EX 8
def NBDT_MNIST_convolution3():

    d,c = load_mnist()
    d[d>=1] = 1
    f   = np.ones((4,4))
    dc  = convolve_mnist(d,f)
    g1 = GaussianNB()
    g2 = DecisionTreeClassifier()
    meanNB, stdNB = classify_avg(d,c,g1)
    meanDT, stdDT = classify_avg(d,c,g2)

    return meanNB, stdNB, meanDT, stdDT
