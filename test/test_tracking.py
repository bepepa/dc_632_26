import pytest
import numpy as np

from dc_632_26.tracking import (
    First_Order_Filter,
    Second_Order_Filter,
    Integrator,
    PLL,
    ALL,
)


def test_first_order_filter():
    H1 = First_Order_Filter(0.1)  # instantiate filter
    assert 0.2 == H1(2)  # pass d_n=2 to filter, output sould be 0.2
    assert 0.1 == H1.alpha  # retrieve the value of alpha
    H1.alpha = 0.2  # change the gain
    assert 0.2 == H1.alpha  # and check it


def test_second_order_filter():
    H2 = Second_Order_Filter(0.1, 0.025)  # instantiate filter
    assert 0.2 == H2(2)  # pass d_n=2 to filter, output sould be 0.2
    assert 0.05 == H2.state  # check that state was updated
    assert 0.25 == H2(2)  # pass 2 again, this time state is not zero
    assert 0.1 == H2.alpha1  # retrieve the value of alpha1
    H2.alpha1 = 0.2  # change the gain
    assert 0.2 == H2.alpha1  # and check it
    assert 0.025 == H2.alpha2  # retrieve the value of alpha2
    H2.alpha2 = 0.01  # change the gain
    assert 0.01 == H2.alpha2  # and check it
    H2.state = 0  # set the state
    assert 0 == H2.state  # and check it
    assert H2.damping_factor == 1.0  # check the default damping factor


def test_integrator():
    S = Integrator()
    x = np.ones(10)
    y = np.zeros_like(x)
    for n in range(10):
        y[n] = S(x[n])

    assert np.allclose(y, np.arange(10))


def test_pll():
    "check the PLL loop convergence"
    N = 1000  # number of samples
    pll = PLL()  # instantiate PLL
    Z_n = np.exp(
        1j * (2 * np.pi * 0.001 * np.arange(N) + np.pi / 4)
    )  # "received" signal
    s_ref = np.ones_like(Z_n)  # reference signal magnitude
    output = np.zeros_like(Z_n)
    for n in range(N):
        output[n] = pll.correct(Z_n[n])
        pll.update(output[n], s_ref[n])

    assert np.allclose(
        output[-1], s_ref[-1]
    )  # check that loop has converged at the end


def test_all():
    "check the ALL loop convergence"
    N = 1000
    all = ALL()  # instantiate the ALL class
    Z_n = 2 * np.ones(N)  # input signal
    s_ref = np.ones_like(Z_n)  # reference signal magnitude
    output = np.zeros_like(Z_n)
    for n in range(N):
        output[n] = all.correct(Z_n[n])
        all.update(output[n], s_ref[n])

    assert np.allclose(
        output[-1], s_ref[-1]
    )  # check that loop has converged at the end


def test_both():
    "check the convergence of both PLL and ALL loops together"
    N = 1000
    pll = PLL()
    all = ALL()
    Z_n = 2 * np.exp(1j * (2 * np.pi * 0.001 * np.arange(N) + np.pi / 4))
    s_ref = np.ones_like(Z_n)
    output = np.zeros_like(Z_n)

    for n in range(N):
        tmp = all.correct(Z_n[n])  # correct amplitude
        output[n] = pll.correct(tmp)  # and phase
        pll.update(output[n], s_ref[n])
        all.update(output[n], s_ref[n])

    assert np.allclose(output[-1], s_ref[-1])
