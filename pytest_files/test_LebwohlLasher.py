import pytest
import numpy as np
import tempfile
import os
import glob
from ..code_files.LebwohlLasher import initdat, savedat, one_energy, all_energy, get_order, MC_step

def test_initdat():
    # testing using an arbitrary side length of 10
    nmax = 10
    arr = initdat(nmax)
    
    # testing that the array is the expected shape
    assert arr.shape == (nmax, nmax)
    
    # testing that the array elements are in the expected range i.e. [0, 2pi]
    assert (arr >= 0).all()
    assert (arr <= 2*np.pi).all()

def test_savedat():
    # initialising
    nmax = 10
    nsteps = 100
    Ts = 0.5

    # setting test parameters
    ratio = [0.5] * (nsteps + 1)
    energy = [0.0] * (nsteps + 1)
    order = [0.3] * (nsteps + 1)
    
    # creating a temporary directory to store test output, to avoid having multiple files
    with tempfile.TemporaryDirectory() as temp_dir:
        # changing to the temporary directory so that the file saves in it
        os.chdir(temp_dir)

        savedat(arr=None, nsteps=nsteps, Ts=Ts, runtime=1.0, ratio=ratio, energy=energy, order=order, nmax=nmax)

        # testing that a file has been created in the expected format
        generated_files = glob.glob("LL-Output-*.txt")
        assert generated_files, f"No file with the pattern 'LL-Output-*.txt' found in {temp_dir}"

        # reading test file and ensuring contents are as expected
        with open(generated_files[0], "r") as file:
            content = file.read()
        
        assert f"Size of lattice:     {nmax}x{nmax}" in content
        assert f"Number of MC steps:  {nsteps}" in content
        assert f"Reduced temperature: {Ts:.3f}" in content
        assert "# MC step:  Ratio:     Energy:   Order:" in content

def test_one_energy():
    # initialising lattice, using arbitrary side length and picking a random single cell
    nmax = 10
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    ix, iy = 1, 1
    
    # testing function output against calculated true value
    en = one_energy(arr, ix, iy, nmax)
    assert en == -1.815714435326352

def test_all_energy():
    # initialising lattice
    nmax = 10
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    
    # testing function output against calculated true value
    enall = all_energy(arr, nmax)
    assert enall == -117.80421697004066

def test_get_order():
    # initialising lattice
    nmax = 10
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi

    # testing function output against calculated true value
    order_parameter = get_order(arr, nmax)
    assert order_parameter == 0.2950862155984001

def test_MC_step():
    # initialising lattice and picking an arbitrary value for reduced temp
    nmax = 10
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    Ts = 0.5

    # testing function output against calculated true value
    MC = MC_step(arr, Ts, nmax)
    assert MC == 0.64