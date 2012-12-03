'''
Created on Nov 27, 2012

@author: hitokazu
'''

import sys
import math
import time
import random

from bzrc import BZRC, Command

from anglecalc import computeAngle

class Agent(object):
    """Class handles all command and control logic for a teams tanks."""

    def __init__(self, bzrc):
        self.bzrc = bzrc
        self.constants = self.bzrc.get_constants()
        self.commands = []
        self.speed = 1#random.uniform(0.1, 1.5)
        
    def tick(self):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks

        if self.mytanks[0].flag != '-':
            self.speed = -1 
        else:
            if self.speed < 0:
                self.speed = 1#random.uniform(0.1, 1.5)

        if self.mytanks[0].status == 'dead':
            self.speed = 1#random.uniform(0.1, 1.5)
            

        self.commands = []

        for tank in mytanks:
            self.move_forward(tank)
        
        results = self.bzrc.do_commands(self.commands)
        
    def move_forward(self, tank):
        """ Tanks move forward. """
        angle = computeAngle(400, tank.y, -400, tank.x, tank.angle)
        command = Command(tank.index, self.speed, 2*angle, True)
        self.commands.append(command)

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

    time_diff = 0

    # Run the agent
    try:
        while True: 
            #print "Elapsed Time: %f" % time_diff
            agent.tick()

    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()


if __name__ == '__main__':
    main()

