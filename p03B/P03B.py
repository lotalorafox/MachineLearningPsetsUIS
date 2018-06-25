from math import *
import matplotlib as plt
from random import gauss
import numpy as np
import bisect
from filter import *

class RobotPosition:
    def __init__(self, landmarks, world_size):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0;
        self.turn_noise    = 0.0;
        self.sense_noise   = 1.0;
        self.landmarks  = landmarks
        self.world_size = world_size
    
    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= self.world_size:
            raise (ValueError, 'X coordinate out of bound')
        if new_y < 0 or new_y >= self.world_size:
            raise (ValueError, 'Y coordinate out of bound')
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise (ValueError, 'Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);
        self.sense_noise   = float(new_s_noise);
    
    
    def sense(self):
        Z = []
        for i in range(len(self.landmarks)):
            dist = sqrt((self.x - self.landmarks[i][0]) ** 2 + (self.y - self.landmarks[i][1]) ** 2)
            dist += gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z
    
    
    def move(self, turn, forward):
        if forward < 0:
            raise (ValueError, 'Robot cant move backwards')       
        
        # turn, and add randomness to the turning command
        orientation = (self.orientation + float(turn) + gauss(0.0, self.turn_noise)) % (2*pi)
        
        # move, and add randomness to the motion command
        dist = float(forward) + gauss(0.0, self.forward_noise)
        x = ( self.x + (cos(orientation) * dist) ) % self.world_size
        y = ( self.y + (sin(orientation) * dist) ) % self.world_size
        
        # set particle
        res = RobotPosition(self.landmarks, self.world_size)
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        res.landmarks = self.landmarks
        res.world_size = self.world_size
        return res
    
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))

def Gaussian(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return np.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * pi * (sigma ** 2))


def measurement_prob(particle, measurement):

    # calculates how likely a measurement should be
    prob = 1.0;
    particlepos = particle.sense()
    sigma = np.std(measurement)
    for i in range(len(particle.landmarks)):
        pro_tem = Gaussian(particlepos[i],sigma,measurement[i]) 
        prob = prob*pro_tem
    return prob

def weighted_sample(weight_list, n_samples=None):    
    if n_samples is None:
        n_samples=len(weight_list)
    
    proba = weight_list*n_samples
    result = []
    print(proba)
    # -- TU CODIGO AQUI ---
    i=n_samples
    while i>0:
        a = random
        j = a.randint(0,len(weight_list))
        if proba[j] >0:
            result.append(j)
            proba[j]= proba[j]-1
            i-=1
    # -----
    return result


def main():
    lmarks  = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]
    wsize   = 100.0

    myrobot = RobotPosition(lmarks, wsize)
    myrobot.x, myrobot.y = 70,70
    myrobot.sense_noise  = 5.
    measurement = myrobot.sense()
    print (myrobot)

    p = RobotPosition(lmarks, wsize)
    p.x, p.y = myrobot.x, myrobot.y
    p.sense_noise = myrobot.sense_noise
    print (p)
    print (measurement_prob(p,measurement))

    for i in range(10):
        p = RobotPosition(lmarks, wsize)
        p.sense_noise = myrobot.sense_noise
        prob = measurement_prob(p,measurement)
        print (p, prob)

if __name__ == "__main__":
    main()