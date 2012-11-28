'''
Created on Nov 27, 2012

@author: hitokazu
'''

import sys
import math
import time
import random
import threading

from bzrc import BZRC, Command, Answer

import numpy as np
#cimport numpy as np

#DOUBLE = np.float64

#ctypedef np.float64_t DOUBLE_t

class Agent(object):
    """Class handles all command and control logic for a teams tanks."""

    def __init__(self, bzrc):
        self.debug = False
        self.bzrc = bzrc
        self.constants = self.bzrc.get_constants()
        self.attractive_s = 10
        self.attractive_alpha = 0.7
        self.speed = 0
        self.delta_t = 0.5
        self.c = 0.1 # friction constant (can be 0)
        msize = 6

        self.mut = np.zeros(6)
        self.mut.dtype = np.float64
        
        self.St = np.zeros(msize*msize).reshape((msize,msize))
        self.St.dtype = np.float64
        for i in xrange(msize):
            for j in xrange(msize):
                if i == j:
                    if (i == 0 and j == 0) or (i == 5 and j == 5):
                        self.St[i, j] = 100
                    else:
                        self.St[i, j] = 0.1
                else:
                    continue
                
        # self.St = [[100.0 , 0   , 0   , 0   , 0   , 0    ],
        #            [0     , 0.1 , 0   , 0   , 0   , 0    ],
        #            [0     , 0   , 0.1 , 0   , 0   , 0    ],
        #            [0     , 0   , 0   , 0.1 , 0   , 0    ], 
        #            [0     , 0   , 0   , 0   , 0.1 , 0    ],
        #            [0     , 0   , 0   , 0   , 0   , 100.0]], dtype=np.float64)
        
        dt = self.delta_t
        c = self.c
                
        self.F = np.zeros(msize*msize).reshape((6,6))
        self.F.dtype = np.float64
        for i in xrange(msize):
            for j in xrange(msize):
                if i == j:
                    self.F[i, j] = 1.0
                elif j == i + 1 and i != 2:
                    self.F[i, j] = dt
                elif j == i + 2 and (i == 0 or i == 3):
                    self.F[i, j] = dt**2/2.0
                elif i == j + 1  and (i == 2 or i == 5):
                    self.F[i, j] = -c
                else:
                    continue

        # self.F = [[1.0 , dt  , dt**2/2.0 , 0   , 0   , 0        ],
        #          [0   , 1.0 , dt        , 0   , 0   , 0        ],
        #          [0   , -c  , 1.0       , 0   , 0   , 0        ],
        #          [0   , 0   , 0         , 1.0 , dt  , dt**2/2.0],
        #          [0   , 0   , 0         , 0   , 1.0 , dt       ],
        #          [0   , 0   , 0         , 0   , -c  , 1.0      ]], dtype=np.float64)
        
        self.Sx = np.zeros(msize*msize).reshape((6,6))
        self.Sx.dtype = np.float64
        for i in xrange(msize):
            for j in xrange(msize):
                if i == j:
                    if i == 2 or i == 5:
                        self.Sx[i, j] = 100
                    else:
                        self.Sx[i, j] = 0.1
                else:
                    continue
        
        #self.Sx = [[0.1 , 0   , 0   , 0   , 0   , 0  ],
        #           [0   , 0.1 , 0   , 0   , 0   , 0  ],
        #           [0   , 0   , 100 , 0   , 0   , 0  ],
        #           [0   , 0   , 0   , 0.1 , 0   , 0  ],
        #           [0   , 0   , 0   , 0   , 0.1 , 0  ],
        #           [0   , 0   , 0   , 0   , 0   , 100]], dtype=np.float64)
        
        self.H = np.zeros(msize*msize).reshape((2,6))
        self.H.dtype = np.float64
        self.H[0,0] = self.H[1,5] = 1.0
        #self.H = [[1,0,0,0,0,0],
        #          [0,0,0,0,0,1], dtype=np.float64])
        
        self.Sz = np.array([25,0,0,25]).reshape((2,2))
        self.Sz.dtype = np.float64
        
        self.K = np.zeros(msize*msize).reshape((6,6))
        self.K.dtype = np.float64

        self.fire = True
        self.infinity = float("inf")
        self.colors = ['red', 'blue', 'green', 'purple']
        # self.constants = self.bzrc.get_constants()
        # constants values:
        #{'shotspeed': '100', 'tankalive': 'alive', 'truepositive': '0.97', 'worldsize': '800', 
        # 'explodetime': '5', 'truenegative': '0.9', 'shotrange': '350', 'flagradius': '2.5', 
        # 'tankdead': 'dead', 'tankspeed': '25', 'shotradius': '0.5', 
        # 'tankangvel': '0.785398163397', 'linearaccel': '0.5', 'team': 'red', 
        # 'tankradius': '4.32', 'angularaccel': '0.5', 'tankwidth': '2.8', 'tanklength': '6'}

        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()

        # initial observation
        self.x = np.matrix(np.zeros((6,1)), dtype=np.float64)
        self.x[0,0] = othertanks[0].x
        self.x[3,0] = othertanks[0].y

        self.commands = []
         
    def tick(self):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks
        self.othertanks = othertanks

        self.commands = []

        for tank in self.othertanks:
            print "enemy position: %d, %d" % (tank.x, tank.y)

        #for tank in self.mytanks:
            #self.prev_t = time.time()
            #t = threading.Thread(target=self.update_probabilities(tank))
            #self.update_probabilities(tank)
            #threads.append(t)
            #t.start()
            #self.prev[tank.index] = (tank.x, tank.y)
        
        #results = self.bzrc.do_commands(self.commands)

    def set_attractive_points(self, tank):
        """ set the positions to be visited"""
        self.grids = []
        size = self.bzrc.get_occgrid(tank.index)[2]
        occgrid_width = max(size[0], size[1])
        self.occgrid_width = occgrid_width         

        for y in xrange(int(occgrid_width/self.grid_size), self.worldsize, occgrid_width/int(self.grid_size*self.grid_step)):
            grids = []
            for x in xrange(int(occgrid_width/self.grid_size), self.worldsize, occgrid_width/int(self.grid_size*self.grid_step)):
                grids.append((x-400, y-400))
            self.grids.append(grids)
            
        self.grids = self.separate_gridlist()
        if len(self.grids) > 1:
            for i in xrange(len(self.grids)):
                if i+1 > int(len(self.grids)*0.5):
                    #print "%d is reversed" % i
                    self.grids[i].reverse()

        self.flags = []

        for grids in self.grids: # iterate over flag list separated by # of tanks
            self.flags.append(self.set_separated_flags(grids))

        if self.debug == True:
            print "%d x %d x %d" % (len(self.flags), len(self.flags[0]), len(self.flags[0][0]))

    def create_flag(self, x, y):
        """ create a new flag """
        flag = Answer()
        for color in self.colors:
            if color != self.constants['team']:
                break
        flag.color = color
        flag.poss_color = color
        flag.x = float(x)
        flag.y = float(y)
        
        return flag
        
    def set_flags(self, grids):
        """ set a list of new flags """
        flags = []
        for grid in grids:
            flag = self.create_flag(grid[0], grid[1])
            flags.append(flag)
        
        return flags
    
    def set_separated_flags(self, grids):
        """ set flag lists separeted by # of tanks"""
        flaglist = []
        for gridlist in grids:
            flaglist.append(self.set_flags(gridlist))
        return flaglist

    def add_new_attractive_point(self):
        """ if the tank seems to be stuck, throw in a new attractive point to get it to move again"""
        found = False
        for k in xrange(len(self.grids)): # grid chunk divided by # of tanks
            for j in xrange(len(self.grids[k])): # rows
                for i in xrange(len(self.grids[k][j])): # cols
                    if self.flags[k][j][i].poss_color != self.constants['team']:
                        new_x = self.cap_max_and_min(random.randint(self.flags[k][j][i].x-(self.occgrid_width/2+1), self.flags[k][j][i].x+(self.occgrid_width/2+1)))
                        new_y = self.cap_max_and_min(random.randint(self.flags[k][j][i].y-(self.occgrid_width/2+1), self.flags[k][j][i].y+(self.occgrid_width/2+1)))           
                        new_point = (new_x, new_y)
                        self.flags[k][j].insert(i, self.create_flag(new_point[0], new_point[1]))
                        if self.debug == True:
                            print "New point at (%d, %d) is added." % new_point
                        #self.prev = (self.mytanks[0].x, self.mytanks[0].y)
                        found = True
                        break
                if found == True:
                    break
            if found == True:
                break

    def swap_flag_list(self):
        """ put the current front flags at the last of the list """
        found = False
        for k in xrange(len(self.grids)): # grid chunk divided by # of tanks
            for j in xrange(len(self.grids[k])):
                for i in xrange(len(self.grids[k][j])):
                    if self.flags[k][j][i].poss_color != self.constants['team']:
                        lflags = self.flags[k].pop(j)
                        self.flags[k].append(lflags)
                        if self.debug == True:
                            print "Send flag list (%d, %d, %d) to the last" % (i, j, k) 
                        found = True
                        break
                if found == True:
                    break
            if found == True:
                break

    def get_direction(self, tank):
        """ Get the moving direction based on the combined vector """
        cdef double delta_x, delta_y 
        delta_x, delta_y = self.compute_attractive_vectors(tank) # compute the strongest attractive vector and the target flag
        command = self.create_move_forward_command(tank, delta_x, delta_y)
        self.commands.append(command)

    def compute_attractive_x_and_y(self, flag, d, tank, r):
        cdef double theta, cos, sin, const, delta_x, delta_y
        
        if d == 0:
            d = math.sqrt((flag.x - tank.x)**2 + (flag.y-tank.y)**2)
        else:
            d = math.sqrt(d)
        
        if flag != None and flag.poss_color != self.constants['team']:
            theta = math.atan2(flag.y-tank.y, flag.x-tank.x)
        else:
            theta = 0
        if d < r:
            delta_x = delta_y = 0
        else:
            cos = math.cos(theta)
            sin = math.sin(theta)
            if r <= d and d <= self.attractive_s + r:
                const = self.attractive_alpha * (d - r)
                delta_x = const * cos
                delta_y = const * (d - r) * sin
            elif d > self.attractive_s + r:
                const = self.attractive_alpha * self.attractive_s
                delta_x = const * cos
                delta_y = const * sin        
        return delta_x, delta_y

    def count_nonvisited_flags(self):
        """ count the number of flags that haven't been visited"""
        cdef int counter = 0
        for flaglist in self.flags:
            for flags in flaglist:
                for flag in flags:
                    if flag.poss_color != self.constants['team']:
                        counter += 1
        return counter

    def compute_attractive_vectors(self, tank):
        """ compute the strongest attractive vector and return the direction and the angle """        

        cdef double min_d

        min_d, best_flag = self.find_best_flag(tank)

        delta_x, delta_y = self.compute_attractive_x_and_y(best_flag, min_d, tank, 0)

        return delta_x, delta_y

    def loop_over_flaglist(self, tank):
        """ loop over flag list and returns min_d, best_flag, and found/not found."""
        cdef double min_d
        
        min_d = self.infinity
        best_flag = None
        separated_flag_list = self.flags[tank.index]
        found = False
        
        for flags in separated_flag_list:
            for flag in flags:
                #print "flag: (%d, %d)" % (flag.x, flag.y)
                if flag.poss_color != self.constants['team']:
                    d = ((flag.x - tank.x)**2 + (flag.y - tank.y)**2) # get distance between tank and flag
                    if d < min_d:
                        min_d = d
                        best_flag = flag
                        found = True
                        break
            if found == True:
                break

        return found, min_d, best_flag

    def find_best_flag(self, tank):
        """ find best flag and its mininum distance"""
        cdef double min_d
        
        found, min_d, best_flag = self.loop_over_flaglist(tank)
        
        if found == False:
            for mytank in self.mytanks:
                if mytank.index != tank.index:
                    found, min_d, best_flag = self.loop_over_flaglist(mytank)
                    if found == True:
                        return min_d, best_flag
        
        return min_d, best_flag
        
    def create_move_forward_command(self, tank, delta_x, delta_y):
        """ produce move forward command """
        cdef double angle = self.compute_angle(tank, delta_x, delta_y)
        self.stuck = self.is_stuck(tank)
        #if self.debug == True:
        #    print "Prev: %d, %d  Cur: %d, %d  Stuck? %s" % (self.prev[tank.index][0], self.prev[tank.index][1], tank.x, tank.y, self.stuck)
        if self.stuck == False:
            command = Command(tank.index, self.speed, 2*angle, self.fire)
        else:
            #command = Command(tank.index, self.speed, 2*angle, self.fire)
            #angle = angle + random.uniform(-5, 0)
            command = Command(tank.index, self.speed, angle-5, self.fire)                    
        return command

    def compute_angle(self, tank, delta_x, delta_y):
        cdef double angle = math.atan2(delta_y, delta_x)
        cdef double relative_angle = self.normalize_angle(angle - tank.angle, tank)
        return relative_angle
            
    def normalize_angle(self, angle, tank):
        """Make any angle be between +/- pi."""
        angle -= 2 * math.pi * int (angle / (2 * math.pi))
        if angle <= -math.pi:
            angle += 2 * math.pi
        elif angle > math.pi:
            angle -= 2 * math.pi                    
        return angle

