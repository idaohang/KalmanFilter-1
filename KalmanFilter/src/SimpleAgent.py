'''
Created on Nov 27, 2012

@author: hitokazu
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
        self.speed = 1
        
    def tick(self):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks

        self.commands = []

        for tank in mytanks:
            self.move_forward(tank)
        
        results = self.bzrc.do_commands(self.commands)
        
    def move_forward(self, tank):
        """ Tanks move forward. """
        command = Command(tank.index, self.speed, 0, True)
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

