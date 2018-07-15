#!/usr/bin/env python3

import math
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise

from tank_encoders import TankEncoderTracker


class TankKalman2(UnscentedKalmanFilter):
    STATE_LEN = 6
    STATE_X = 0
    STATE_Y = 1
    STATE_VS = 2
    STATE_VP = 3
    STATE_THETA = 4
    STATE_OMEGA = 5

    MEAS_LEN = 4
    MEAS_X = 0
    MEAS_Y = 1
    MEAS_THETA_ENC = 2
    MEAS_THETA_NAVX = 3

    def __init__(self, dt, wheel_base, var_rate_enc, meas_da):
        sp = MerweScaledSigmaPoints(n=self.STATE_LEN, alpha=0.3, beta=2., kappa=-1.)
        UnscentedKalmanFilter.__init__(self, dim_x=self.STATE_LEN, dim_z=self.MEAS_LEN, dt=dt,
                                       fx=self.f_func,
                                       residual_x=self.x_residual_fn,
                                       x_mean_fn=self.x_mean_fn,
                                       hx=self.h_func,
                                       residual_z=self.h_residual_fn,
                                       z_mean_fn=self.h_mean_fn,
                                       points=sp)
        self.wheel_base = wheel_base
        self.var_a_s = 1.0
        self.var_a_p = 0.25
        self.var_a_a = math.radians(20.0)**2

        # starting point and covariance
        self.x = np.array(self.STATE_LEN*[0., ])
        sig_a_start = math.radians(5.0)
        self.P = np.diag([0.5**2, 0.5**2, self.var_a_s**2, self.var_a_p**2, sig_a_start**2, self.var_a_a**2])

        self.meas_da = meas_da
        self.prev_x = None
        self.prev_encoders = [0, 0, 0]
        self.tank_encoder = TankEncoderTracker(wheel_base, var_rate_enc, var_rate_enc)
        self.prev_encoder_pos = [0, 0, 0]
        return

    def predict(self, dt):
        # save the previous state for use during update()
        self.prev_x = np.copy(self.x)

        # set Q and predict
        self.Q = self.Q_func(dt, self.x[self.STATE_THETA])

        return UnscentedKalmanFilter.predict(self)

    def update(self, enc_l, enc_r, yaw):
        # Compute position from encoders, use yaw
        del_l = enc_l - self.prev_encoders[0]
        del_r = enc_r - self.prev_encoders[1]
        # del_theta = yaw - self.prev_encoders[2]

        # ds, dp, dtheta_enc = self.tank_encoder.compute_move(del_l, del_r, del_theta)
        ds, dp, dtheta_enc = self.tank_encoder.compute_move(del_l, del_r)
        c = math.cos(yaw)
        s = math.sin(yaw)
        dx = c * ds - s * dp
        dy = s * ds + c * dp
        # print("move:", del_l, del_r, del_theta, ds, dp, dx, dy)
        z = np.array([self.prev_encoder_pos[0] + dx, self.prev_encoder_pos[1] + dy, self.prev_encoder_pos[2] + dtheta_enc, yaw])

        self.prev_encoder_pos = z

        if abs(dtheta_enc) < 0.001:
            d = 0.5 * (del_l + del_r)
            R3 = self.tank_encoder.cov_line(d)
        else:
            R3 = self.tank_encoder.cov_arc(del_l, del_r)

        R4 = np.eye(self.MEAS_LEN)
        R4[0:3, 0:3] = R3
        R4[3, 3] = self.meas_da**2

        self.prev_encoders[0] = enc_l
        self.prev_encoders[1] = enc_r
        self.prev_encoders[2] = yaw
        return UnscentedKalmanFilter.update(self, z, R=R4)

    def Q_func(self, dt, theta):
        # TODO: maybe expand this out for speed. For now, easier to program and debug
        q2 = Q_discrete_white_noise(2, dt=dt, var=1.0)

        q_sp = np.array([[self.var_a_s*q2[0, 0], 0, self.var_a_s*q2[0, 1], 0],
                         [0, self.var_a_p*q2[0, 0], 0, self.var_a_p*q2[0, 1]],
                         [self.var_a_s*q2[1, 0], 0, self.var_a_s*q2[1, 1], 0],
                         [0, self.var_a_p*q2[1, 0], 0, self.var_a_p*q2[1, 1]]])
        Q_spa = np.eye(self.STATE_LEN)
        Q_spa[0:4, 0:4] = q_sp
        Q_spa[4:self.STATE_LEN, 4:self.STATE_LEN] = self.var_a_a * q2

        c = math.cos(theta)
        s = math.sin(theta)
        T = np.eye(self.STATE_LEN)
        T[0, 0] = c
        T[0, 1] = -s
        T[1, 0] = s
        T[1, 1] = c

        Q_xya = T @ Q_spa @ T.T    # @ is matrix multiply; Python >=3.5
        return Q_xya

    @staticmethod
    def normalize_angle(x):
        x = x % (2 * np.pi)    # force in range [0, 2 pi)
        if x > np.pi:          # move to [-pi, pi)
            x -= 2 * np.pi
        return x

    @staticmethod
    def f_func(x, dt):
        theta = x[TankKalman2.STATE_THETA]
        c_t = math.cos(theta)
        s_t = math.sin(theta)

        d_s = dt * x[2]
        d_p = dt * x[3]
        d_theta = dt * x[5]

        if abs(d_theta) < 0.001:
            # straight line motion
            # print('straight', d_s, d_p, c_t, s_t)
            dx = np.array([d_s*c_t - d_p*s_t, d_s*s_t + d_p*c_t, 0, 0, 0, 0])

        else:
            # turning, circular motion
            r = d_s / d_theta

            s_t_dt = math.sin(theta + d_theta)
            c_t_dt = math.cos(theta + d_theta)
            dx = np.array([r*(s_t_dt - s_t) - d_p*(s_t + s_t_dt)/2.0,
                           r*(c_t - c_t_dt) + d_p*(c_t + c_t_dt)/2.0,
                           0, 0, d_theta, 0])

        return x + dx

    @staticmethod
    def x_residual_fn(a, b):
        y = a - b
        y[TankKalman2.STATE_THETA] = TankKalman2.normalize_angle(y[TankKalman2.STATE_THETA])
        return y

    @staticmethod
    def x_mean_fn(sigmas, Wm):
        x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

        # x[TankKalman2.STATE_THETA] is an angle, so it needs special treatment
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, TankKalman2.STATE_THETA]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, TankKalman2.STATE_THETA]), Wm))
        x[TankKalman2.STATE_THETA] = math.atan2(sum_sin, sum_cos)

        return x

    # ----------------------------------------

    @staticmethod
    def h_func(x):
        # angle is measured two ways
        return [x[TankKalman2.STATE_X], x[TankKalman2.STATE_Y],
                x[TankKalman2.STATE_THETA], x[TankKalman2.STATE_THETA]]

    @staticmethod
    def h_residual_fn(a, b):
        y = a - b
        y[TankKalman2.MEAS_THETA_ENC] = TankKalman2.normalize_angle(y[TankKalman2.MEAS_THETA_ENC])
        y[TankKalman2.MEAS_THETA_NAVX] = TankKalman2.normalize_angle(y[TankKalman2.MEAS_THETA_NAVX])
        return y

    @staticmethod
    def h_mean_fn(sigmas, Wm):
        z = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

        # z[TankKalman2.MEAS_THETA_ENC] is an angle, so it needs special treatment
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, TankKalman2.MEAS_THETA_ENC]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, TankKalman2.MEAS_THETA_ENC]), Wm))
        z[TankKalman2.MEAS_THETA_ENC] = math.atan2(sum_sin, sum_cos)

        # z[TankKalman2.MEAS_THETA_NAVX] is an angle, so it needs special treatment
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, TankKalman2.MEAS_THETA_NAVX]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, TankKalman2.MEAS_THETA_NAVX]), Wm))
        z[TankKalman2.MEAS_THETA_NAVX] = math.atan2(sum_sin, sum_cos)

        return z
