#!/usr/bin/env python3

import math
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter

from tank_encoders import TankEncoderTracker
from tank_kalman_2 import TankKalman2

pi2 = math.pi/2.0


def cmp_arrays(calc, expect, tol, msg):
    print('Testing {}'.format(msg))
    diff = np.abs(np.diff(calc - expect))
    # print('diff =', diff)
    d_max = np.ndarray.max(diff)
    if d_max > tol:
        print('{} does not agree. dPmax = {}'.format(msg, d_max))
        print('calculated:', calc)
        print('expected:', expect)
        assert False
    return


def test_f():
    '''Test the state transition function'''

    f = TankKalman2.f_func
    dt = 0.05               # 50 ms, robot control period

    # straight lines
    x = np.array([0, 0, 15.0, 0, 0, 0])
    for i in range(int(5/dt)):
        x = f(x, dt)
    cmp_arrays(x, np.array([75.0, 0, 15.0, 0, 0, 0]), 1e-6, 'Straight Line 1')

    x = np.array([0, 0, 7.5, 0, -pi2, 0])
    for i in range(int(5/dt)):
        x = f(x, dt)
    cmp_arrays(x, np.array([0, -5*7.5, 7.5, 0, -pi2, 0]), 1e-6, 'Straight Line 2')

    x = np.array([0, 5.0, 10.0, 0, pi2 + pi2/2.0, 0])
    for i in range(int(5/dt)):
        x = f(x, dt)
    dx = 50.0/math.sqrt(2.0)
    expected = np.array([-dx, 5.0 + dx, 10.0, 0, pi2 + pi2/2.0, 0])
    cmp_arrays(x, expected, 1e-6, 'Straight Line 3')

    x = np.array([0, 5.0, -10.0, 0, pi2 + pi2/2.0, 0])
    for i in range(int(5/dt)):
        x = f(x, dt)
    dx = 50.0/math.sqrt(2.0)
    expected = np.array([dx, 5.0 - dx, -10.0, 0, pi2 + pi2/2.0, 0])
    cmp_arrays(x, expected, 1e-6, 'Backwards Line')

    # circles
    da_tot = pi2
    omega = da_tot / 5.0
    v = 15.0
    s = v * 5.0
    r = abs(2.0 * s / math.pi)
    x = np.array([0, 0, 15.0, 0, 0, omega])
    for i in range(int(5.0/dt)):
        x = f(x, dt)
    cmp_arrays(x, np.array([r, r, 15.0, 0, da_tot, omega]), 1e-6, 'Curve 1')

    da_tot = -2.0 * math.pi
    omega = da_tot / 10.0
    v = 10.0
    x = np.array([0, 0, v, 0, 0, omega])
    for i in range(int(10.0/dt)):
        x = f(x, dt)
    cmp_arrays(x, np.array([0, 0, v, 0, da_tot, omega]), 1e-6, 'Curve 2')

    da_tot = -pi2
    omega = da_tot / 10.0
    v = -10.0
    r = abs(v * 10.0 / da_tot)
    x = np.array([0, 0, v, 0, 0, omega])
    for i in range(int(10.0/dt)):
        x = f(x, dt)
    # print('back curve', r, x)
    cmp_arrays(x, np.array([-r, r, v, 0, da_tot, omega]), 1e-6, 'Backward Curve')

    x = np.array([0, 5.0, 10.0, 1.0, pi2 + pi2/2.0, 0])
    for i in range(int(5/dt)):
        x = f(x, dt)
    dx = 50.0/math.sqrt(2.0)
    dp = dx/10.
    expected = np.array([-dx-dp, 5.0 + dx - dp, 10.0, 1.0, pi2 + pi2/2.0, 0])
    cmp_arrays(x, expected, 1e-6, 'Strafe 1')

    da_tot = pi2
    omega = da_tot / 10.0
    v = 10.0
    r = v * 10.0 / da_tot
    vp = r / 10.0
    x = np.array([0, 0, v, vp, 0, omega])
    for i in range(int(10.0/dt)):
        x = f(x, dt)
    # print('s2', x)
    # this is the sum of two circles: forward is as expected. Perpendicular is at 90deg, curving up and back.
    r2 = vp * 10.0 / da_tot
    expected = np.array([r - r2, r + r2, v, vp, da_tot, omega])
    # print('exp', expected)

    # not quite as accurate because of the step size
    cmp_arrays(x, expected, 1e-3, 'Strafe 2')

    print("test_f() passed")
    return


def test_kalman_motion(testname, vel_s, vel_p, turn_rad, dangle_tot, variance_diag,
                       start_angle=0, manual_p=None, manual_x=None, t_total=5.0):
    dt = 0.05               # 50 ms, robot control period
    kf = TankKalman2(dt, 2.0, 0.0, 0.0)
    kf.Q = np.zeros((6, 6))

    omega = dangle_tot / t_total
    kf.x = np.array([0, 0, vel_s, vel_p, start_angle, omega])
    kf.P = np.diag(variance_diag)
    for i in range(int(t_total / dt)):
        UnscentedKalmanFilter.predict(kf)

    if manual_x is not None:
        x_expect = manual_x
    elif dangle_tot < 0.0001:
        x = vel_s * t_total
        y = vel_p * t_total
        if abs(start_angle) > 0.0001:
            x2 = x * math.cos(start_angle) - y * math.sin(start_angle)
            y2 = x * math.sin(start_angle) + y * math.cos(start_angle)
            x = x2
            y = y2
        x_expect = np.array([x, y, vel_s, vel_p, start_angle, omega])
    else:
        raise Exception('not implemented')

    # print('kf.x =', kf.x)
    # print('kf.P =', kf.P)

    cmp_arrays(kf.x, x_expect, 0.015, testname + ' x')

    if manual_p is not None:
        cmp_arrays(kf.P, manual_p, 1e-3, testname + ' manual_P')
    else:
        encoders = TankEncoderTracker(kf.wheel_base, 0, 0)

        p_enc0 = np.diag([variance_diag[0], variance_diag[1], variance_diag[TankKalman2.STATE_THETA]])
        if dangle_tot < 0.0001:
            p_sp = encoders.propagate_line(vel_s * t_total, p_enc0)
        else:
            dl = TankEncoderTracker.dl_func(turn_rad, kf.x[TankKalman2.STATE_THETA], kf.wheel_base)
            dr = TankEncoderTracker.dr_func(turn_rad, kf.x[TankKalman2.STATE_THETA], kf.wheel_base)
            p_sp = encoders.propagate_arc(dl, dr, p_enc0)

        t_rot = TankEncoderTracker.t_matrix(kf.x[TankKalman2.STATE_THETA])
        p_xy = t_rot @ p_sp @ t_rot.T
        t_expand = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]])
        p_expect = t_expand @ p_xy @ t_expand.T

        cmp_arrays(kf.P, p_expect, 1e-3, testname + ' encoder_P')

    return kf


def test_predict():
    # standard values
    sig_a = math.radians(1.0)
    t_total = 5.0
    v_s = 5.0

    # straight motion in x, slight angle error
    test_kalman_motion('Straight', v_s, 0, 0, 0, [1e-10, 1e-10, 1e-10, 1e-10, sig_a**2, 1e-10])

    # straight motion at 45deg, slight angle error
    test_kalman_motion('Straight 45deg', v_s, 0, 0, 0, [1e-10, 1e-10, 1e-10, 1e-10, sig_a**2, 1e-10], start_angle=math.pi/4.0)

    # ----------------------------------------

    # straight motion in x, slight angle and parallel velocity error
    sig_vs = 1.0
    P0diag = [1e-10, 1e-10, sig_vs**2, 1e-10, sig_a**2, 1e-10]
    p_exp = np.diag(P0diag)
    p_exp[0, 0] = (t_total * sig_vs)**2
    p_exp[0, 2] = p_exp[2, 0] = t_total * sig_vs**2
    p_exp[1, 1] = (t_total * v_s * math.sin(sig_a))**2
    p_exp[1, 4] = p_exp[4, 1] = t_total * v_s * math.sin(sig_a)**2

    test_kalman_motion('Straight v_s', v_s, 0, 0, 0, P0diag, manual_p=p_exp, t_total=t_total)

    # ----------------------------------------

    # straight motion in x, slight angle and strafe velocity error
    sig_vp = 0.5
    P0diag = [1e-10, 1e-10, 1e-10, sig_vp**2, sig_a**2, 1e-10]
    p_exp = np.diag(P0diag)
    p_exp[1, 1] = (t_total * v_s * math.sin(sig_a))**2 + (5.0 * sig_vp)**2
    p_exp[1, 3] = p_exp[3, 1] = t_total * sig_vp**2
    p_exp[1, 4] = p_exp[4, 1] = t_total * v_s * math.sin(sig_a)**2

    test_kalman_motion('Straight v_p', v_s, 0, 0, 0, P0diag, manual_p=p_exp, t_total=t_total)

    # ----------------------------------------

    # curve
    da_tot = math.pi / 2.0
    r = abs(v_s * t_total / da_tot)
    x_exp = np.array([r, r, v_s, 0, da_tot, da_tot/t_total])
    test_kalman_motion('Curve', v_s, 0, r, da_tot, [1e-10, 1e-10, 1e-10, 1e-10, sig_a**2, 1e-10],
                       t_total=t_total, manual_x=x_exp)

    # ----------------------------------------

    # Rotate
    da_tot = math.pi
    t_total = 5.0
    # note: normalizing angles sends 180 -> -180.
    x_exp = np.array([0, 0, 0, 0, -da_tot, da_tot/t_total])
    test_kalman_motion('Rotation', 0, 0, 0, da_tot, [1e-10, 1e-10, 1e-10, 1e-10, sig_a**2, 1e-10],
                       t_total=t_total, manual_x=x_exp)

    print("test_predict() passed")
    return


def test_mean_res_funcs():
    x1 = np.array([1, 2, 3, 4, math.radians(5), 3])
    x2 = np.array([11, 12, -13, 6, math.radians(355), -3])
    res = TankKalman2.x_residual_fn(x2, x1)
    cmp_arrays(res, [10, 10, -16, 2, math.radians(-10.0), -6], 0.001, 'X Residual 1')

    mean = TankKalman2.x_mean_fn(np.array([x1, x2]), np.array([0.5, 0.5]))
    cmp_arrays(mean, [6, 7, -5, 5, 0, 0], 0.001, 'X Mean 1')

    x1 = np.array([1, 2, 3, 4, math.radians(170), 3])
    x2 = np.array([11, 12, -13, 6, math.radians(200), -3])
    res = TankKalman2.x_residual_fn(x2, x1)
    cmp_arrays(res, [10, 10, -16, 2, math.radians(30.0), -6], 0.001, 'X Residual 2')

    mean = TankKalman2.x_mean_fn(np.array([x1, x2]), np.array([0.5, 0.5]))
    cmp_arrays(mean, [6, 7, -5, 5, math.radians(-175), 0], 0.001, 'X Mean 2')

    print('mean/residual passed')
    return


test_f()
test_predict()
test_mean_res_funcs()
