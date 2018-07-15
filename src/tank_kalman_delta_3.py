#!/usr/bin/env python3

# NOTE: comments like "noqa: xxx" turn off certain flakes8 warnings
# Stupidly, they never implemented a syntax for the whole file

# filterpy imports graphing. Not good
# import matplotlib
# matplotlib.use('PS')   # set renderer to something generic

import math  # noqa: E402
import numpy as np  # noqa: E402
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter  # noqa: E402
from filterpy.common import Q_discrete_white_noise  # noqa: E402

# import utils  # noqa: E402
from tank_encoders import TankEncoderTracker  # noqa: E402


class TankKalmanDelta3(UnscentedKalmanFilter):
    def __init__(self, dt, wheel_base, var_rate_enc, meas_da):
        sp = MerweScaledSigmaPoints(n=8, alpha=0.3, beta=2., kappa=-1.)
        UnscentedKalmanFilter.__init__(self, dim_x=8, dim_z=4, dt=dt,
                                       fx=TankKalmanDelta3.f_func,
                                       residual_x=TankKalmanDelta3.x_residual_fn,
                                       x_mean_fn=TankKalmanDelta3.x_mean_fn,
                                       hx=TankKalmanDelta3.h_func,
                                       residual_z=TankKalmanDelta3.h_residual_fn,
                                       z_mean_fn=TankKalmanDelta3.h_mean_fn,
                                       points=sp)
        self.wheel_base = wheel_base
        self.var_a_s = 4.0
        self.var_a_p = 1e-10
        self.var_a_a = math.radians(45.0)

        # starting point and covariance
        self.x = np.array(8*[0., ])
        sig_a_start = math.radians(5.0)
        self.P = np.diag([0.5**2, 0.5**2, dt**2*self.var_a_s, self.var_a_s, dt**2*self.var_a_p, sig_a_start**2, dt**2*self.var_a_a, self.var_a_a])

        self.meas_da = meas_da
        self.prev_x = None
        self.prev_encoders = [0, 0, 0]
        self.tank_encoder = TankEncoderTracker(wheel_base, var_rate_enc, var_rate_enc)
        return

    def predict(self, dt):
        # save the previous state for use during update()
        self.prev_x = np.copy(self.x)

        # set Q and predict
        self.Q = self.Q_func(dt, self.x[5])

        return UnscentedKalmanFilter.predict(self)

    def update(self, enc_l, enc_r, yaw):
        # Compute position from encoders, use yaw
        del_l = enc_l - self.prev_encoders[0]
        del_r = enc_r - self.prev_encoders[1]
        # del_theta = yaw - self.prev_encoders[2]

        # ds, dp, dtheta_enc = self.tank_encoder.compute_move(del_l, del_r, del_theta)
        ds, dp, dtheta_enc = self.tank_encoder.compute_move(del_l, del_r)
        c = math.cos(self.x[5])
        s = math.sin(self.x[5])
        dx = c * ds - s * dp
        dy = s * ds + c * dp
        # print("move:", del_l, del_r, del_theta, ds, dp, dx, dy)
        z = np.array([dx, dy, dtheta_enc, yaw])

        if abs(dtheta_enc) < 0.001:
            d = 0.5 * (del_l + del_r)
            R3 = self.tank_encoder.cov_line(d)
        else:
            R3 = self.tank_encoder.cov_arc(del_l, del_r)

        R4 = np.eye(4)
        R4[0:3, 0:3] = R3
        R4[3, 3] = self.meas_da**2

        self.prev_encoders[0] = enc_l
        self.prev_encoders[1] = enc_r
        self.prev_encoders[2] = yaw
        return UnscentedKalmanFilter.update(self, z, R=R4, prev_state=self.prev_x)

    def Q_func(self, dt, theta):
        # TODO: maybe expand this out for speed. For now, easier to program and debug
        q3 = Q_discrete_white_noise(3, dt=dt, var=1.0)

        q_sp_1 = np.zeros((5, 5))
        q_sp_1[0:3, 0:3] = self.var_a_s * q3
        q_sp_1[3:5, 3:5] = self.var_a_p * q3[0:2, 0:2]
        trans1 = np.array([[1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1]])
        q_sp = trans1 @ q_sp_1 @ trans1.T

        Q_spa = np.eye(8)
        Q_spa[0:5, 0:5] = q_sp
        Q_spa[5:8, 5:8] = self.var_a_a * q3

        c = math.cos(theta)
        s = math.sin(theta)
        T = np.eye(8)
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
        theta = x[5]
        c_t = math.cos(theta)
        s_t = math.sin(theta)

        d_s = x[2] * dt
        d_p = x[4] * dt
        d_theta = x[6] * dt

        if abs(d_theta) < 0.001:
            # straight line motion
            # print('straight', d_s, d_p, c_t, s_t)
            dx = d_s*c_t - d_p*s_t
            dy = d_s*s_t + d_p*c_t
        else:
            # turning, circular motion
            r = d_s / d_theta

            s_t_dt = math.sin(theta + d_theta)
            c_t_dt = math.cos(theta + d_theta)
            dx = r*(s_t_dt - s_t) - d_p*(s_t + s_t_dt)/2.0
            dy = r*(c_t - c_t_dt) + d_p*(c_t + c_t_dt)/2.0

        d_state = np.array([dx, dy, x[3] * dt, 0, 0, d_theta, x[7] * dt, 0])
        return x + d_state

    @staticmethod
    def x_residual_fn(a, b):
        y = a - b
        y[5] = TankKalmanDelta3.normalize_angle(y[5])
        return y

    @staticmethod
    def x_mean_fn(sigmas, Wm):
        x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

        # x[5] is an angle, so it needs special treatment
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 5]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 5]), Wm))
        x[5] = math.atan2(sum_sin, sum_cos)

        return x

    # ----------------------------------------

    @staticmethod
    def h_func(x, prev_state):
        # angle is measured two ways
        return [x[0] - prev_state[0], x[1] - prev_state[1], x[5] - prev_state[5], x[5]]

    @staticmethod
    def h_residual_fn(a, b):
        y = a - b
        # y[2] = TankKalmanDelta3.normalize_angle(y[2])
        y[3] = TankKalmanDelta3.normalize_angle(y[3])
        return y

    @staticmethod
    def h_mean_fn(sigmas, Wm):
        z = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

        # # z[2] is an angle, so it needs special treatment
        # sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        # sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        # z[2] = math.atan2(sum_sin, sum_cos)

        # z[3] is an angle, so it needs special treatment
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 3]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 3]), Wm))
        z[3] = math.atan2(sum_sin, sum_cos)

        return z
