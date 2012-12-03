'''
Created on Sep 25, 2012

@author: hitokazu

Agent only with attractive potential fields
This will be merged with agents with other potential fields later
'''

import sys
import math
import time
import random

from bzrc import BZRC, Command

class Agent(object):
    """Class handles all command and control logic for a teams tanks."""

    def __init__(self, bzrc):
        self.bzrc = bzrc
        self.constants = self.bzrc.get_constants()
        self.commands = []
        self.s = 10
        self.attractive_alpha = 1
        self.repulsive_radius = 10
        self.speed = 1
        self.tick_count = 0
        #self.elapsed_time = 0
        #self.moving_time = self.set_moving_time()
        #self.shooting_time = self.set_shooting_time()
        
    def tick(self, time_diff):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks
        self.othertanks = othertanks
        self.flags = flags
        self.shots = shots
        self.enemies = [tank for tank in othertanks if tank.color !=
                        self.constants['team']]

        self.commands = []

        if self.tick_count > 95 and self.tick_count < 300:
            self.speed = random.uniform(-0.1, 2)
        else:
            self.speed = 1

        #print "tank position: (%f, %f)" % (mytanks[0].x, mytanks[0].y)
        flag = None
        for flag in self.flags:
            if flag.color == 'blue':
                break
        #print "closest flag position: (%f, %f)" % (flag.x, flag.y)
        
        for tank in self.mytanks:
            if self.tick_count > 95 and self.tick_count < 300:
                for i in xrange(3000000):
                    x = 100 * math.sqrt(i) + i ** 3.5 
            self.get_direction(tank)
        
        self.tick_count += 1
        
#        if shoot == True:
#            if angle == True:
#                for tank in mytanks:
#                    self.shoot(tank)
#                    self.change_angle(tank)
#            else:
#                for tank in mytanks:
#                    self.shoot(tank)
#                    self.move_forward(tank)
#        else:
#            if angle == True:
#                for tank in mytanks:
#                    self.change_angle(tank)
#            else:
#                for tank in mytanks:
#                    self.move_forward(tank)

        results = self.bzrc.do_commands(self.commands)
        
    def get_direction(self, tank):
        """ Get the moving direction based on the strongest attractive vector """
        delta_x, delta_y = self.compute_attractive_vectors(tank) # compute the strongest attractive vector and the target flag
        angle = math.atan2(delta_y, delta_x)
        relative_angle = self.normalize_angle(angle - tank.angle)
        #print "relative angle: %f" % relative_angle
        #print "delta_x: %f \t delta_y: %f" % (delta_x, delta_y)
        if tank.index == 1:        
            print "delta_x: %f \t delta_y: %f \t angle: %f \t tank angle: %f \t relative angle: %f" % (delta_x, delta_y, angle, tank.angle, relative_angle)
        
        command = Command(tank.index, self.speed, 2*relative_angle, False)
        self.commands.append(command)
        
    def compute_attractive_vectors(self, tank):
        """ computer the strongest attractive vector and return the direction and the angle """
        
        min_d = float("inf")
        best_flag = None

        for flag in self.flags:
            if flag.color == 'blue':
                d = (flag.x - tank.x)**2 + (flag.x - tank.y)**2 # get distance between tank and flag
                if d < min_d:
                    min_d = d
                    best_flag = flag

        theta = math.atan2(best_flag.y-tank.y, best_flag.x-tank.x) # compute the angle between tank and flag
        if min_d >= 0 and d <= self.s:
            delta_x = self.attractive_alpha * self.s * min_d * math.cos(theta)
            delta_y = self.attractive_alpha * self.s * min_d * math.sin(theta)
        elif min_d > self.s:
            delta_x = self.attractive_alpha * self.s * math.cos(theta)
            delta_y = self.attractive_alpha * self.s * math.sin(theta)
                
        return (delta_x, delta_y)
            
    def shoot(self, tank):
        command = Command(tank.index, self.speed, 0, False)
        self.commands.append(command)

    def move_to_position(self, tank, target_x, target_y):
        """Set command to move to given coordinates."""
        target_angle = math.atan2(target_y - tank.y,
                                  target_x - tank.x)
        relative_angle = self.normalize_angle(target_angle - tank.angle)
        command = Command(tank.index, self.speed, 2 * relative_angle, False)
        self.commands.append(command)
    
    def normalize_angle(self, angle):
        """Make any angle be between +/- pi."""
        angle -= 2 * math.pi * int (angle / (2 * math.pi))
        if angle <= -math.pi:
            angle += 2 * math.pi
        elif angle > math.pi:
            angle -= 2 * math.pi
        return angle

def main():
    # Process CLI arguments.
    try:
        execname, host, port = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print >>sys.stderr, '%s: incorrect number of arguments' % execname
        print >>sys.stderr, 'usage: %s hostname port' % sys.argv[0]
        sys.exit(-1)

    # Connect.
    #bzrc = BZRC(host, int(port), debug=True)
    bzrc = BZRC(host, int(port))

    agent = Agent(bzrc)

    #agent.elapsed_time = prev_time = time.time()
    time_diff = 0

    #print "Moving Time: %d" % agent.moving_time

    # Run the agent
    try:
        while True: 
            #print "Elapsed Time: %f" % time_diff
            agent.tick(time_diff)
            #for flag in agent.flags:
                #print flag.x, flag.y, flag.color, flag.poss_color
#            time_diff = time.time() - prev_time
#            if time.time() - agent.elapsed_time > agent.shooting_time:
#                print "Shoot!"
#                agent.tick(time_diff, False, True)
#                agent.shooting_time = agent.set_shooting_time()
#                agent.elapsed_time = time.time()
#            if time_diff > agent.moving_time:
#                print "Turning 60 degrees." 
#                if time_diff < agent.moving_time + 0.83:
#                    agent.tick(time_diff, True, False)
#                else:
#                    agent.moving_time = agent.set_moving_time()
#                    prev_time = time.time()
#                    print "Moving Time: %d" % agent.moving_time
#            else:
#                agent.tick(time_diff, False, False)
    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()


if __name__ == '__main__':
    main()

