"""
Script to test function outputs from LebwohlLasher_mpi4py
Numerical function outputs are given a 10% tolerance as some methods sacrifice precision for speed

"""
from pytest import approx
import numpy as np

from ..code_files.LebwohlLasher_mpi4py import initdat, one_energy, all_energy, get_order, MC_step_parallel

def test_initdat():
    # testing using an arbitrary side length of 10
    nmax = 50
    arr = initdat(nmax)
    
    # testing that the array is the expected shape
    assert arr.shape == (nmax, nmax)
    
    # testing that the array elements are in the expected range i.e. [0, 2pi]
    assert (arr >= 0).all()
    assert (arr <= 2*np.pi).all()

def test_one_energy():
    # initialising lattice, using arbitrary side length and picking a random single cell
    nmax = 50
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    ix, iy = 1, 1
    
    # testing function output against calculated true value
    en = one_energy(arr, ix, iy, nmax)
    assert en == approx(0.50459804367517, rel=0.1)

def test_all_energy():
    # initialising lattice
    nmax = 50
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    
    # testing function output against calculated true value
    # this is not the same as the original code - slight reduction in precision using numpy vectorisation
    enall = all_energy(arr, nmax)
    assert enall == approx(-2566.273312501283, rel=0.1)

def test_get_order():
    # initialising lattice
    nmax = 50
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi

    # testing function output against calculated true value
    order_parameter = get_order(arr, nmax)
    assert order_parameter == approx(0.27109543777243506, rel=0.1)

def test_MC_step():
    # initialising lattice and picking an arbitrary value for reduced temp
    nmax = 50
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    Ts = 0.5

    # testing function output against calculated true value
    MC = MC_step_parallel(arr, Ts, nmax)
    assert MC > 0.0 and MC < 1.0