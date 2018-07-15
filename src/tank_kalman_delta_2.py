#!/usr/bin/env python3

# Kalman filter for a tank drive robot.
# Do not track the acceleration terms, but use the "Control" values for that.

# Note that this is pretty similar to the non-delta filter. However, I am keeping them
#  separate because that one is just for historic/learning purposes.
#  This one will get more features.

import math
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise

from tank_encoders import TankEncoderTracker


class TankControlModel(object):
    def __init__(self, motor_intercept=0.1083, motor_slope=8.65, acceleration_factor=5.0,
                 omega_from_vel_factor=0.622, ang_accel_factor=4.8):
        self.motor_slope = motor_slope
        self.motor_intcpt = motor_intercept
        self.accel_factor = acceleration_factor
        self.omega_from_vel_factor = omega_from_vel_factor
        self.ang_accel_factor = ang_accel_factor
        return

    def max_velocity_from_motor(self, motor_frac):
        return max(0, self.motor_slope * (motor_frac - self.motor_intcpt))

    def max_omega_from_motor(self, motor_l, motor_r):
        v_max_l = self.max_velocity_from_motor(motor_l)
        v_max_r = self.max_velocity_from_motor(motor_r)
        return self.omega_from_vel_factor * (v_max_l - v_max_r)

    def linear_acceleration(self, motor, curr_vel):
        max_v = self.max_velocity_from_motor(motor)
        return self.acceleration_fraction * (max_v - curr_vel)

    def omega_acceleration(self, motor_l, motor_r, curr_omega):
        max_o = self.max_omega_from_motor(motor_l, motor_r)
        return self.ang_accel_factor * (max_o - curr_omega)

    def accelerations(self, motors, curr_vel_s, curr_omega):
        avg_m = 0.5 * (motors[0] + motors[1])
        a_s = self.linear_acceleration(avg_m, curr_vel_s)
        a_omega = self.omega_acceleration(motors[0], motors[1], curr_omega)
        return a_s, a_omega


class TankKalmanDelta2(UnscentedKalmanFilter):
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

    CTRL_MOTOR_L = 0
    CTRL_MOTOR_R = 1

    def __init__(self, dt, wheel_base, var_rate_enc, meas_da,
                 var_a_s=4.0, var_a_p=0.25, var_a_a=math.radians(45.0)**2,
                 control_model=None):

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
        self.var_a_s = var_a_s
        self.var_a_p = var_a_p
        self.var_a_a = var_a_a

        # starting point and covariance
        self.x = np.array(self.STATE_LEN*[0., ])
        sig_a_start = math.radians(5.0)
        self.P = np.diag([0.5**2, 0.5**2, self.var_a_s**2, self.var_a_p**2, sig_a_start**2, self.var_a_a**2])

        self.meas_da = meas_da
        self.prev_x = None
        self.prev_encoders = [0, 0, 0]
        self.tank_encoder = TankEncoderTracker(wheel_base, var_rate_enc, var_rate_enc)

        # control model
        # calculates linear and angular accelerations from motors
        self.control_model = control_model

        return

    def predict(self, dt, motors=None):
        # save the previous state for use during update()
        self.prev_x = np.copy(self.x)

        # set Q and predict
        self.Q = self.Q_func(dt, self.x[self.STATE_THETA])

        accelerations = None
        if motors:
            accelerations = self.control_model.accelerations(motors, self.x[self.STATE_VS], self.x[self.STATE_OMEGA])

        return UnscentedKalmanFilter.predict(self, accelerations=accelerations)

    def update(self, enc_l, enc_r, yaw):
        # Compute position from encoders, use yaw
        del_l = enc_l - self.prev_encoders[0]
        del_r = enc_r - self.prev_encoders[1]
        del_theta = yaw - self.prev_encoders[2]

        ds, dp, dtheta_enc = self.tank_encoder.compute_move(del_l, del_r, del_theta)
        # ds, dp, dtheta_enc = self.tank_encoder.compute_move(del_l, del_r)
        # a = self.x[self.STATE_THETA]
        a = yaw
        c = math.cos(a)
        s = math.sin(a)
        dx = c * ds - s * dp
        dy = s * ds + c * dp
        # print("move:", del_l, del_r, del_theta, ds, dp, dx, dy)
        z = np.array([dx, dy, dtheta_enc, yaw])

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
        return UnscentedKalmanFilter.update(self, z, R=R4, prev_state=self.prev_x)

    def Q_func(self, dt, theta):
        # TODO: expand this out for speed. For now, easier to program and debug
        q2 = Q_discrete_white_noise(2, dt=dt, var=1.0)

        Q_spa_1 = np.zeros((self.STATE_LEN, self.STATE_LEN))
        Q_spa_1[0:2, 0:2] = self.var_a_s * q2
        Q_spa_1[2:4, 2:4] = self.var_a_p * q2
        Q_spa_1[4:self.STATE_LEN, 4:self.STATE_LEN] = self.var_a_a * q2

        # swap row/columns 1 and 2 (starting from 0)
        trans1 = np.eye(self.STATE_LEN)
        trans1[1, 1] = trans1[2, 2] = 0
        trans1[1, 2] = trans1[2, 1] = 1
        Q_spa = trans1 @ Q_spa_1 @ trans1.T

        # rotate the (s, p) positions to (x, y) field positions
        T = np.eye(self.STATE_LEN)
        T[0, 0] = T[1, 1] = math.cos(theta)
        T[1, 0] = math.sin(theta)
        T[0, 1] = -T[1, 0]

        Q_xya = T @ Q_spa @ T.T    # @ is matrix multiply; Python >=3.5
        return Q_xya

    @staticmethod
    def normalize_angle(x):
        x = x % (2 * np.pi)    # force in range [0, 2 pi)
        if x > np.pi:          # move to [-pi, pi)
            x -= 2 * np.pi
        return x

    @staticmethod
    def f_func(x, dt, accelerations=None):
        theta = x[TankKalmanDelta2.STATE_THETA]
        c_t = math.cos(theta)
        s_t = math.sin(theta)

        d_s = dt * x[TankKalmanDelta2.STATE_VS]
        d_p = dt * x[TankKalmanDelta2.STATE_VP]
        d_theta = dt * x[TankKalmanDelta2.STATE_OMEGA]

        if abs(d_theta) < 0.001:
            # straight line motion
            dx = np.array([d_s*c_t - d_p*s_t, d_s*s_t + d_p*c_t, 0, 0, 0, 0])
        else:
            # turning, circular motion
            r = d_s / d_theta

            s_t_dt = math.sin(theta + d_theta)
            c_t_dt = math.cos(theta + d_theta)
            dx = np.array([r*(s_t_dt - s_t) - d_p*(s_t + s_t_dt)/2.0,
                           r*(c_t - c_t_dt) + d_p*(c_t + c_t_dt)/2.0,
                           0, 0, d_theta, 0])

        new_x = x + dx

        if accelerations is not None:
            dx = np.zeros((TankKalmanDelta2.STATE_LEN,))
            dx[TankKalmanDelta2.STATE_V_S] = dt * accelerations[0]
            dx[TankKalmanDelta2.STATE_OMEGA] = dt * accelerations[1]
            new_x += dx

        return new_x

    @staticmethod
    def x_residual_fn(a, b):
        y = a - b
        y[TankKalmanDelta2.STATE_THETA] = TankKalmanDelta2.normalize_angle(y[TankKalmanDelta2.STATE_THETA])
        return y

    @staticmethod
    def x_mean_fn(sigmas, Wm):
        x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

        # x[TankKalmanDelta2.STATE_THETA] is an angle, so it needs special treatment
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, TankKalmanDelta2.STATE_THETA]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, TankKalmanDelta2.STATE_THETA]), Wm))
        x[TankKalmanDelta2.STATE_THETA] = math.atan2(sum_sin, sum_cos)

        return x

    # ----------------------------------------

    @staticmethod
    def h_func(x, prev_state):
        # angle is measured two ways
        return [x[TankKalmanDelta2.STATE_X] - prev_state[TankKalmanDelta2.STATE_X],
                x[TankKalmanDelta2.STATE_Y] - prev_state[TankKalmanDelta2.STATE_Y],
                x[TankKalmanDelta2.STATE_THETA] - prev_state[TankKalmanDelta2.STATE_THETA],
                x[TankKalmanDelta2.STATE_THETA]]

    @staticmethod
    def h_residual_fn(a, b):
        y = a - b
        y[TankKalmanDelta2.MEAS_THETA_NAVX] = TankKalmanDelta2.normalize_angle(y[TankKalmanDelta2.MEAS_THETA_NAVX])
        return y

    @staticmethod
    def h_mean_fn(sigmas, Wm):
        z = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

        # z[TankKalmanDelta2.MEAS_THETA_NAVX] is an angle, so it needs special treatment
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, TankKalmanDelta2.MEAS_THETA_NAVX]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, TankKalmanDelta2.MEAS_THETA_NAVX]), Wm))
        z[TankKalmanDelta2.MEAS_THETA_NAVX] = math.atan2(sum_sin, sum_cos)

        return z
