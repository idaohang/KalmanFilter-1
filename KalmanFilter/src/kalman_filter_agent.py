'''
Created on Nov 27, 2012

@author: hitokazu
'''

import sys
import time, math
from bzrc import BZRC, Command, Answer

from kalmanagent import Agent

from shootingcalc import check_criterion

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

    burn_in = 0

    # Run the agent
    try:
        while True:
            #start = time.time()

            if agent.tick_count > agent.step_size:
                agent.tick_count = 1

            if burn_in < agent.burn_in:
                burn_in += 1
            else:
                if check_criterion(agent.step_size, agent.buffer, agent.tick_count):
                    agent.fire = True
                    agent.tick_count = 1
                else:
                    agent.fire = False
                     
                agent.tick_count += 1
                
            agent.tick()

            #end = time.time()
            #diff.append(float(end - start))
            #if tick_count == 20:
            #    print "average time in a tick: %.2f" % (sum(diff)/len(diff))
        
    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()

if __name__ == '__main__':
    main()
