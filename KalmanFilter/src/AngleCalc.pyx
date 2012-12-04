from libc.math cimport atan2, sqrt
from math import pi

cpdef double computeAngle(double target_y, double tank_y, double target_x, double tank_x, \
                          double tank_angle):
    """ compute angle between tank and target """
    cdef double angle = atan2(target_y - tank_y, target_x - tank_x)
    cdef double relative_angle = normalizeAngle(angle - tank_angle)
    #print "relative angle: %.2f" % relative_angle
    return relative_angle
        
cdef double normalizeAngle(double angle) except? -1:
    """Make any angle be between +/- pi."""
    angle -= 2 * pi * int (angle / (2 * pi))
    if angle <= -pi:
        angle += 2 * pi
    elif angle > pi:
        angle -= 2 * pi                    
    return angle

cpdef calc_timing(int step_size, double sec_per_tick, double bullet_velocity, double target_x, \
                  double target_y, double tank_x, double tank_y):
                """ calcluate when to shoot """
                cdef double d = distance(target_x, target_y, tank_x, tank_y)
                cdef double time_needed = d/bullet_velocity
                cdef int tick_needed = int(time_needed/sec_per_tick)
                
                if tick_needed > 50:
                    return tick_needed, 70
                elif tick_needed > 40:
                    return tick_needed, 60
                elif tick_needed > 30:
                    return tick_needed, 50
                else:
                    return tick_needed, 40
    
                #return int(step_size - (d/bullet_velocity)/sec_per_tick)
                #return tick_needed 

cdef double distance(double target_x, double target_y, double tank_x, double tank_y):
    return sqrt((target_x - tank_x) ** 2 + (target_y - tank_y) ** 2)
