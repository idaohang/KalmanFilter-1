'''
Created on Nov 27, 2012

@author: hitokazu
'''

import sys
import math
import time
import random
import threading

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pylab import *

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from bzrc import BZRC, Command, Answer

import numpy as np
cimport numpy as np

DOUBLE = np.float64

ctypedef np.float64_t DOUBLE_t

import pprint

from kalmancalc import predict, update, future_prediction
from anglecalc import computeAngle, calc_timing

class Agent(object):
    """Class handles all command and control logic for a teams tanks."""

    def __init__(self, bzrc):
        #self.pp = pprint.PrettyPrinter()
        #self.debug = False
        self.bzrc = bzrc
        self.constants = self.bzrc.get_constants()
        self.speed = 0
        self.tick_count = 0
        self.bullet_speed = 100  
        self.sec_per_tick = 0.1 # approximate
        self.burn_in = 50
        self.Sx_pos = 0.5
        self.Sx_accel = 0.01
        self.c = 0 # friction constant (can be 0)
        self.figure_counter = 1
        self.delta_t = 0.5
        self.step_size = 50
        self.buffer = 40
        self.plot = False
        self.fire = False
        # experimental: do we need vel and accel??
        #self.xpos = 0
        #self.ypos = 0
        #self.xvel = 0
        #self.yvel = 0
        #self.xaccel = 0
        #self.yaccel = 0
        # self.accel = 0

        cdef int msize = 6
        cdef int i, j

        self.mu = np.zeros(6).reshape(6,1)
        self.mu.dtype = np.float64

        self.predicted_mu = np.zeros(6).reshape(6,1) # for shooting
        self.mu.dtype = np.float64

        self.St = np.zeros(msize*msize).reshape((msize,msize))
        for i in xrange(msize):
            for j in xrange(msize):
                if i == j:
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        self.St[i, j] = 100
                    else:
                        self.St[i, j] = 0.1
                else:
                    continue
                
        # self.St = [[100.0 , 0   , 0   , 0   , 0   , 0    ],
        #            [0     , 0.1 , 0   , 0   , 0   , 0    ],
        #            [0     , 0   , 0.1 , 0   , 0   , 0    ],
        #            [0     , 0   , 0   , 100 , 0   , 0    ], 
        #            [0     , 0   , 0   , 0   , 0.1 , 0    ],
        #            [0     , 0   , 0   , 0   , 0   , 0.1  ]], dtype=np.float64)
        
        cdef double dt, c
        
        dt = self.delta_t
        c = self.c
                
        self.F = np.zeros(msize*msize).reshape((6,6))
        for i in xrange(msize):
            for j in xrange(msize):
                if i == j:
                    self.F[i, j] = 1.0
                elif j == i + 1 and i != 2:
                    self.F[i, j] = dt
                elif j == i + 2 and (i == 0 or i == 3):
                    self.F[i, j] = (dt**2)/2.0
                elif i == j + 1  and (i == 2 or i == 5):
                    self.F[i, j] = -c
                else:
                    continue

        self.FT = self.F.T

        # self.F = [[1.0 , dt  , dt**2/2.0 , 0   , 0   , 0        ],
        #          [0   , 1.0 , dt        , 0   , 0   , 0        ],
        #          [0   , -c  , 1.0       , 0   , 0   , 0        ],
        #          [0   , 0   , 0         , 1.0 , dt  , dt**2/2.0],
        #          [0   , 0   , 0         , 0   , 1.0 , dt       ],
        #          [0   , 0   , 0         , 0   , -c  , 1.0      ]], dtype=np.float64)
        
        self.Sx = np.zeros(msize*msize).reshape((6,6))
        for i in xrange(msize):
            for j in xrange(msize):
                if i == j:
                    if i == 2 or i == 5:
                        self.Sx[i, j] = self.Sx_accel
                    else:
                        self.Sx[i, j] = self.Sx_pos
                else:
                    continue
        
        #self.Sx = [[0.1 , 0   , 0   , 0   , 0   , 0  ],
        #           [0   , 0.1 , 0   , 0   , 0   , 0  ],
        #           [0   , 0   , 100 , 0   , 0   , 0  ],
        #           [0   , 0   , 0   , 0.1 , 0   , 0  ],
        #           [0   , 0   , 0   , 0   , 0.1 , 0  ],
        #           [0   , 0   , 0   , 0   , 0   , 100]], dtype=np.float64)
        
        
        self.H = np.zeros(2*msize).reshape((2,msize))
        self.H[0,0] = self.H[1,3] = 1.0
        #self.H = [[1,0,0,0,0,0],
        #          [0,0,0,0,0,1], dtype=np.float64])
        
        self.HT = self.H.T
        
        self.Sz = np.array([25.0,0,0,25.0]).reshape((2,2))
        
        self.K = np.zeros(msize*msize).reshape((6,6))
        self.K.dtype = np.float64

        # self.constants = self.bzrc.get_constants()
        # constants values:
        #{'shotspeed': '100', 'tankalive': 'alive', 'truepositive': '0.97', 'worldsize': '800', 
        # 'explodetime': '5', 'truenegative': '0.9', 'shotrange': '350', 'flagradius': '2.5', 
        # 'tankdead': 'dead', 'tankspeed': '25', 'shotradius': '0.5', 
        # 'tankangvel': '0.785398163397', 'linearaccel': '0.5', 'team': 'red', 
        # 'tankradius': '4.32', 'angularaccel': '0.5', 'tankwidth': '2.8', 'tanklength': '6'}

        mytanks, self.othertanks, flags, shots = self.bzrc.get_lots_o_stuff()

        # initial observation
        self.z = np.zeros(2).reshape((2,1))
        self.get_observation()
        # self.x = np.zeros((6)).reshape(6,1)
        # self.x.dtype=np.float64
        # self.x[0,0] = self.xpos = othertanks[0].x
        # self.x[2,0] = self.accel
        # self.x[3,0] = self.ypos = othertanks[0].y
        # self.x[5,0] = self.accel

        # self.mu = self.x

        #print "S_0:"
        #self.pp.pprint(self.St)

        self.commands = []
        
        if self.plot:
            self.density_plot()

#        print "Sigma_x"
#        self.pp.pprint(self.Sx)
#        print "F"
#        self.pp.pprint(self.F)
#        print "H"
#        self.pp.pprint(self.H)
#        print "Sigma_z"
#        self.pp.pprint(self.Sz)


    def get_observation(self):
        """ obtain observation z_t"""
        self.z[0,0] = self.othertanks[0].x
        self.z[1,0] = self.othertanks[0].y 

         
    def tick(self):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks
        self.othertanks = othertanks

        self.commands = []

#        if self.othertanks[0].status == 'dead':
#            self.end = time.time()
#            print "other x: %d" % self.enemy[0]
#            print "other y: %d" % self.enemy[1]
#            print "my x: %.2f" % self.mytanks[0].x
#            print "my y: %.2f" % self.mytanks[0].y
#            print "distance: %.2f" % self.d
#            print "time: %.2f" % (self.end-self.start)
#            print "bullet velocity: %.2f" % (self.d/(self.end-self.start))


        for tank in self.othertanks:
            self.enemy = (tank.x, tank.y)
        #    print "enemy position: %d, %d" % (tank.x, tank.y)

        if self.othertanks[0].status != 'dead':
            for tank in self.mytanks:
                self.mu_hat, self.S_hat = predict(self.F, self.mu, self.St, self.FT, self.Sx)
                #self.predict()
                #self.update()
                self.get_observation()
                self.K, self.mu, self.St = update(self.z, \
                                                  self.H, self.mu_hat, self.S_hat, \
                                                  self.HT, self.Sz)
    
                self.predicted_mu = future_prediction(self.F, self.mu.copy(), self.step_size - self.tick_count)
                self.get_direction()
    
                if self.plot:
                    if self.tick_count % 50 == 0:
                        self.density_plot()
                
                #self.tick_count += 1
    
            if self.plot:
                if self.othertanks[0].status == 'dead':
                    plt.show()

        #for tank in self.mytanks:
            #self.prev_t = time.time()
            #t = threading.Thread(target=self.update_probabilities(tank))
            #self.update_probabilities(tank)
            #threads.append(t)
            #t.start()
            #self.prev[tank.index] = (tank.x, tank.y)
        
        #self.d = math.sqrt((self.enemy[0] - self.mytanks[0].x)**2 + (self.enemy[1] - self.mytanks[0].y)**2)
        #self.start = time.time()
        results = self.bzrc.do_commands(self.commands)

    def density_plot(self):
        x = np.arange(-400.0, 400.0, 1)
        y = np.arange(-400.0,400.0, 1)
        X, Y = np.meshgrid(x, y)
        Z = mlab.bivariate_normal(X, Y, math.sqrt(self.St[0,0]), math.sqrt(self.St[3,3]), \
                                  self.mu[0,0], self.mu[3,0])

        plt.figure(self.figure_counter)

        im = plt.imshow(Z, interpolation='bilinear', origin='lower',
                        cmap=get_cmap('binary'), extent=(-400,400,-400,400))
        levels = np.arange(-1.2, 1.6, 0.2)
        CS = plt.contour(Z, levels,
                         origin='lower',
                         linewidths=2,
                         extent=(-3,3,-2,2))
        
        #Thicken the zero contour.
        zc = CS.collections[6]
        plt.setp(zc, linewidth=4)
        
        plt.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%1.1f',
                   fontsize=14)
        
        plt.title('Kalman Filter')
        plt.flag()
        
        # We can still add a colorbar for the image, too.
        CBI = plt.colorbar(im, orientation='vertical', shrink=0.8)
        
        # This makes the original colorbar look a bit out of place,
        # so let's improve its position.
        l,b,w,h = plt.gca().get_position().bounds
        
        plt.plot(self.z[0,0], self.z[1,0], 'ro', markersize=6) # observed location
        plt.plot(self.mu[0,0], self.mu[3,0], 'bo', markersize=6) # kalman filter estimate
    
        self.figure_counter += 1
        

#    def compute_obs(self):
#        """ compte velocity and acceleration """
#        newx = self.othertanks[0].x
#        newy = self.othertanks[0].y
#        
#        newxvel = (newx - self.xpos) / self.delta_t
#        newyvel = (newy - self.ypos) / self.delta_t
#        
#        print "new xvel: %.2f    new yvel: %.2f" % (newxvel, newyvel)
#        
#        newxaccel = (newxvel - self.xaccel) / self.delta_t
#        newyaccel = (newyvel - self.yaccel) / self.delta_t  
#        
#        print "new xaccel: %.2f    new yaccel: %.2f" % (newxaccel, newyaccel)
#
#        self.x[0,0] = newx
#        self.x[1,0] = newxvel
#        self.x[2,0] = newxaccel
#        self.x[3,0] = newy
#        self.x[4,0] = newyvel
#        self.x[5,0] = newyaccel
#        
#        self.x = np.random.multivariate_normal(self.F.dot(self.x).flatten(), self.Sx, 1).T
#        self.mu = self.x
        
#        print "mu_t:"
#        self.pp.pprint(self.mu)
#        print "S_t:"
#        self.pp.pprint(self.St)

#    def predict(self):
#        """ predict current state and covariance """
#        self.mu_hat = self.F.dot(self.mu)
#        #print "mu_hat"
#        #self.pp.pprint(self.mu_hat)
#        self.S_hat = (self.F.dot(self.St)).dot(self.FT) + self.Sx
#
#    def update(self):
#        """ update the current state and covariance at time t with Kalman update equation. """
#        self.get_observation()
#        z_tilda = self.z - self.H.dot(self.mu_hat) # observation error
#        S = (self.H.dot(self.S_hat)).dot(self.HT) + self.Sz # covariance of observation residue
#        self.K = (self.S_hat.dot(self.HT)).dot(np.linalg.inv(S)) # Kalman gain
#        self.mu = self.mu_hat + self.K.dot(z_tilda) # updated current estimation (mu)
##        print 'mu'
##        self.pp.pprint(self.mu)
#        self.St = self.S_hat - (self.K.dot(self.H)).dot(self.S_hat) # updated current error matrix
        #self.compute_obs()
        #self.x[0,0] = self.othertanks[0].x
        #self.x[3,0] = self.othertanks[0].y
        #self.x = np.random.multivariate_normal(self.F.dot(self.x).flatten(), self.Sx, 1).T
        #if not self.observed:
        #    self.mu = np.random.multivariate_normal(self.F.dot(self.x).flatten(), self.Sx, 1).T
        #    self.observed = True
        #self.mu = self.x
        #self.z = np.random.multivariate_normal(self.H.dot(self.x).flatten(), self.Sz, 1).T
        #print self.z
        #factor = self.F.dot(self.St).dot(self.FT)
        #self.K = ((factor + self.Sx).dot(self.HT)).dot(np.linalg.inv(self.H.dot(factor + self.Sx).dot(self.HT) + self.Sz))
        #self.K = factor.dot(self.HT).dot(np.linalg.inv(self.H.dot(factor).dot(self.HT) + self.Sz))
        #self.mu = self.F.dot(self.mu) + self.K.dot(self.z - self.H.dot(self.F.dot(self.mu)))
        #self.mu = self.F.dot(self.mu) #+ self.K.dot(self.z - self.H.dot(self.F.dot(self.mu)))
        #self.St = (np.identity(6, dtype=np.float64) - self.K.dot(self.H)).dot(factor + self.Sx)

    def get_direction(self):
        """ Get the moving direction based on the combined vector """
        command = self.create_move_forward_command()
        self.commands.append(command)

    def create_move_forward_command(self):
        """ produce move forward command """
        #cdef double angle = self.compute_angle()
        cdef double target_y, target_x 

        #if self.fire:
            #target_y = self.mu[3,0] 
            #target_x = self.mu[0,0]
            #self.tick_count = 0 
        if not self.fire:
            #print "mu.x: %.2f   mu.y: %.2f   p_mu.x: %.2f   p_mu.y: %.2f" % (self.mu[0,0], self.mu[3,0], self.predicted_mu[0,0], self.predicted_mu[3,0])
            target_y = self.predicted_mu[3,0] 
            target_x = self.predicted_mu[0,0] 
            self.buffer, self.step_size = calc_timing(self.step_size, self.sec_per_tick, self.bullet_speed, \
                                      target_x, target_y, self.mytanks[0].x, \
                                      self.mytanks[0].y)
            print "buffer: %d" % self.buffer
            print "tick: %d" % self.tick_count
            print "step_size - buffer: %d" % (self.step_size - self.buffer)
#        cdef double target_y = self.mu[3,0] + self.mu[4,0] * self.delta_t * 22.0
#        cdef double target_x = self.mu[0,0] + self.mu[1,0] * self.delta_t * 22.0
        cdef double angle = computeAngle(target_y, self.mytanks[0].y, 
                                         target_x, self.mytanks[0].x, \
                                         self.mytanks[0].angle)
        #print "angvel: %f" % (2.0*angle)
        command = Command(self.mytanks[0].index, self.speed, 2*angle, self.fire)
        return command

#    def compute_angle(self):
#        #print "(mu.x: %.2f   mu.y: %.2f    obs.x: %.2f   obs.y: %.2f)" % (self.mu[0,0], self.mu[3,0], self.z[0,0], self.z[1,0])
#        cdef double angle = math.atan2(self.mu[3,0] - self.mytanks[0].y, 
#                                       self.mu[0,0] - self.mytanks[0].x)
#        cdef double relative_angle = self.normalize_angle(angle - self.mytanks[0].angle)
#        #print "relative angle: %.2f" % relative_angle
#        return relative_angle
#            
#    def normalize_angle(self, angle):
#        """Make any angle be between +/- pi."""
#        cdef double angle -= 2 * math.pi * int (angle / (2 * math.pi))
#        if angle <= -math.pi:
#            angle += 2 * math.pi
#        elif angle > math.pi:
#            angle -= 2 * math.pi                    
#        return angle
