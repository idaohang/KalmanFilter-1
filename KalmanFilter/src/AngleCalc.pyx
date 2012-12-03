from libc.math cimport atan2
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
