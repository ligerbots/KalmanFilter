#!/usr/bin/env python3

# Taken from "Odometry Error Covariance Estimation for Two Wheel Robot Vehicles"
#  by Lindsay Kleeman, Monash University, Tech Report MECSE-95-1 (1995)
#  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.4728&rep=rep1&type=pdf

import numpy as np
from math import cos, sin


class TankEncoderTracker(object):
    def __init__(self, wheel_base, var_rate_l, var_rate_r):
        '''Init routine.
           wheel_base = distance between the wheel (perpendicular to forward)
           error_l and _r = error *rate* for the encoders
             sigma_l**2 = var_rate_l * distance

           WARNING: var_rate_l = k_l**2 from the paper!!!
        '''

        self.wheel_base = wheel_base
        self.var_rate_l = var_rate_l
        self.var_rate_r = var_rate_r
        return

    @staticmethod
    def t_matrix(theta):
        c = cos(theta)
        s = sin(theta)
        t = np.array([[  c,  -s, 0.0],
                      [  s,   c, 0.0],
                      [0.0, 0.0, 1.0]])
        return t

    # useful for test routines
    @staticmethod
    def dl_func(R, a, B):
        return a * (R - B / 2.0)

    @staticmethod
    def dr_func(R, a, B):
        return a * (R + B / 2.0)

    def compute_move(self, d_l, d_r, d_theta=None):
        # robot sim is backwards!! For now, make this agree
        if d_theta is None:
            d_theta = (d_l - d_r) / self.wheel_base

        d_avg = (d_l + d_r) / 2.0
        # print(d_theta, d_avg)
        if abs(d_theta) < 1e-4:
            d_s = d_avg
            d_p = 0
        else:
            r_c = d_avg / d_theta
            d_s = r_c * sin(d_theta)
            d_p = r_c * (1.0 - cos(d_theta))

        return d_s, d_p, d_theta

    def propagate_line(self, dist, P_old):
        C = self.cov_line(dist)

        phi = np.eye(3)
        phi[1, 2] = dist

        # NB '@' is matrix multiple in Python 3.5 and later
        # The first term is actually the state translation contribution. Should be handled
        #   elsewhere in a Kalman filter
        P_new = phi @ P_old @ phi.T + C

        return P_new

    def cov_line(self, dist):
        '''Covariance change for straight line motion. NB P is in (f,s) coord'''

        d_ksum2 = dist * (self.var_rate_l + self.var_rate_r)
        d_kdiff2 = dist * (self.var_rate_r - self.var_rate_l)
        B2 = self.wheel_base**2

        c00 = 0.25 * d_ksum2
        c01 = 0.25 * dist * d_kdiff2 / self.wheel_base
        c02 = 0.5 * d_kdiff2 / self.wheel_base
        c11 = dist**2 * d_ksum2 / (3.0*B2)
        c12 = 0.5 * dist * d_ksum2 / B2
        c22 = d_ksum2 / B2

        C = np.array([[c00, c01, c02],
                      [c01, c11, c12],
                      [c02, c12, c22]])
        return C

    def propagate_arc(self, dl, dr, P_old):
        C = self.cov_arc(dl, dr)

        a = (dr - dl) / self.wheel_base
        R = (dr + dl)/(2.0 * a)
        sinA = sin(a)
        cosA = cos(a)
        phi = np.array([[ cosA, sinA, R * (1.0 - cosA)],
                        [-sinA, cosA,         R * sinA],
                        [    0,    0,              1.0]])

        # NB '@' is matrix multiple in Python 3.5 and later
        # The first term is actually the state translation contribution. Should be handled
        #   elsewhere in a Kalman filter
        P_new = phi @ P_old @ phi.T + C
        return P_new

    def cov_arc(self, dl, dr, del_angle=None):
        '''Covariance change for curved motion. NB P is in (f,s) coord'''

        if del_angle is not None:
            a = del_angle
        else:
            a = (dr - dl) / self.wheel_base

        R = (dr + dl)/(2.0 * a)
        kRsr = abs(dr * self.var_rate_r / a)
        kRsl = abs(dl * self.var_rate_l / a)
        ksum = kRsr + kRsl
        kdiff = kRsr - kRsl
        RoB = R / self.wheel_base
        RoB2 = RoB**2
        B2 = self.wheel_base**2
        sin2A = sin(2.0 * a)
        sinA = sin(a)
        cos2A = cos(2.0 * a)
        cosA = cos(a)

        c00 = ksum * (RoB2 * (1.5 * a - 2.0 * sinA + 0.25*sin2A) + a / 8.0 + sin2A / 16.0) + \
              kdiff * RoB * (sinA - a / 2.0 - 0.25*sin2A)
        c11 = (a / 2.0 - 0.25*sin2A) * (ksum * (RoB2 + 0.25) - kdiff * RoB)
        c22 = a * ksum / B2

        # math and code in the paper don't agree!
        # this is from the code, which I suspect is correct.
        c01 = ksum * (RoB2 * 0.25*(3.0 - 4.0*cosA + cos2A) + (cos2A - 1)/16.0) + \
              0.25 * kdiff * RoB * (2.0*cosA - cos2A - 1.0)

        c02 = ksum * RoB / self.wheel_base * (a - sinA) + kdiff * sinA / (2.0*self.wheel_base)
        c12 = (1.0 - cosA) * (ksum * RoB / self.wheel_base - kdiff / (2.0*self.wheel_base))

        C = np.array([[c00, c01, c02],
                      [c01, c11, c12],
                      [c02, c12, c22]])
        if a < 0:
            C *= -1.0
        return C


if __name__ == '__main__':
    import math

    def cmp_covariance(P_calc, P_expect, step_num):
        dP = P_calc - P_expect
        dPmax = np.ndarray.max(dP)
        # print('10^4 * Pxy', 1e4 * P)
        assert dPmax < 5e-9, 'Pxy step {} does not agree. dPmax = {}'.format(step_num, dPmax)
        return

    # Test case from the paper
    kl = 1e-6
    kr = 1e-6
    B = 0.5

    results_10e4 = [
        np.array([[0.01, 0, 0], [0, 0.2133, 0.16], [0, 0.16, 0.16]]),
        np.array([[0.0110, 0.0006, 0], [0.0006, 0.2143, -0.16], [0, -0.16, 0.1914]]),
        np.array([[0.2370, -0.3883, -0.2421], [-0.3883, 0.7846, 0.4264], [-0.2421, 0.4264, 0.3171]]),
        np.array([[0.3032, -0.4763, 0.2817], [-0.4763, 0.8974, -0.4700], [0.2817, -0.4700, 0.3485]])
    ]

    tracker = TankEncoderTracker(B, kl, kr)
    print('kl, kr, B =', tracker.error_l, tracker.error_r, tracker.wheel_base)
    theta = 0.0

    # start with no error
    P = np.zeros((3, 3))

    # Step 1: move 2 meters
    d = 2.0
    P = tracker.propagate_line(d, P)  # this is Pxy, since angle = 0
    cmp_covariance(P, 1e-4 * results_10e4[0], 1)

    # Step 2: rotate 90deg to the left
    da = math.pi / 2.0
    dl = TankEncoderTracker.dl_func(0, da, B)
    dr = TankEncoderTracker.dr_func(0, da, B)
    P = tracker.propagate_arc(dl, dr, P)
    theta += da
    # rotate to (x,y)
    t = TankEncoderTracker.t_matrix(-theta)
    Pxy = t @ P @ t.T
    cmp_covariance(Pxy, 1e-4 * results_10e4[1], 2)

    # Step 3: an arc of 90 deg with radius of 1 (to the right)
    da = -math.pi / 2.0
    dl = TankEncoderTracker.dl_func(-1.0, da, B)
    dr = TankEncoderTracker.dr_func(-1.0, da, B)
    P = tracker.propagate_arc(dl, dr, P)
    theta += da
    # rotate to (x,y)
    t = TankEncoderTracker.t_matrix(-theta)
    Pxy = t @ P @ t.T
    cmp_covariance(Pxy, 1e-4 * results_10e4[2], 3)

    # Step 4: 90 deg turn, radius = 0.125 on the left    da = -math.pi / 2.0
    da = math.pi / 2.0
    dl = TankEncoderTracker.dl_func(0.125, da, B)
    dr = TankEncoderTracker.dr_func(0.125, da, B)
    P = tracker.propagate_arc(dl, dr, P)
    theta += da
    # rotate to (x,y)
    t = TankEncoderTracker.t_matrix(-theta)
    Pxy = t @ P @ t.T
    cmp_covariance(Pxy, 1e-4 * results_10e4[3], 4)

    print('Passed')
