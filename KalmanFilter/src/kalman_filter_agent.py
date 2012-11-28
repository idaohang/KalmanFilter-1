'''
Created on Nov 27, 2012

@author: hitokazu
'''

import sys
import time
from bzrc import BZRC, Command, Answer

from kalmanagent import Agent

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

    # Run the agent
    try:
        while True: 
            agent.tick()
                    
    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()

if __name__ == '__main__':
    main()
