"""
Cythonised version of basic Python Lebwohl-Lasher code.

"""
import sys
import time
import datetime
import numpy as np

# cython imports
cimport cython
cimport numpy as np
from libc.math cimport sin, cos, pi, exp

# MPI imports and initialisation
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

#=======================================================================
# cythonised function
cdef np.ndarray initdat(int nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    cdef np.ndarray[double, ndim=2] arr = np.random.random_sample((nmax,nmax))*2.0*pi
    return arr

#=======================================================================
# cythonised function
cdef savedat(double[:, ::1] arr, int nsteps, double Ts, double[:] ratio, double[:] energy, double[:] order, int nmax):
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "/user/home/zj20898/Year4/LebwohlLasher_MiniProject/LL-Output-{:s}.txt".format(current_datetime).encode('utf-8')
    
    cdef float runtime = 0.0
    cdef int i

    with open(filename, "wb") as FileOut:
        FileOut.write("#=====================================================\n".encode('utf-8'))
        FileOut.write("# File created:        {:s}\n".format(current_datetime).encode('utf-8'))
        FileOut.write("# Size of lattice:     {:d}x{:d}\n".format(nmax, nmax).encode('utf-8'))
        FileOut.write("# Number of MC steps:  {:d}\n".format(nsteps).encode('utf-8'))
        FileOut.write("# Reduced temperature: {:5.3f}\n".format(Ts).encode('utf-8'))
        FileOut.write("# Run time (s):        {:8.6f}\n".format(runtime).encode('utf-8'))
        FileOut.write("#=====================================================\n".encode('utf-8'))
        FileOut.write("# MC step:  Ratio:     Energy:   Order:\n".encode('utf-8'))
        FileOut.write("#=====================================================\n".encode('utf-8'))
        
        for i in range(nsteps + 1):
            FileOut.write("\n   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i, ratio[i], energy[i], order[i]).encode('utf-8'))

#=======================================================================
# cythonised function
cdef float one_energy(np.ndarray[double, ndim=2] arr, int ix, int iy, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    cdef float en = 0.0
    cdef int ixp = (ix+1)%nmax 
    cdef int ixm = (ix-1)%nmax 
    cdef int iyp = (iy+1)%nmax 
    cdef int iym = (iy-1)%nmax 

    cdef float ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    return en
#=======================================================================
# cythonised function
cdef float all_energy(np.ndarray[double, ndim=2] arr, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    cdef float enall = 0.0
    cdef int i,j

    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
# cythonised function
cdef float get_order(np.ndarray[double, ndim=2] arr,int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    cdef int a, b, i, j

    cdef double[:,:] Qab = np.zeros((3, 3), dtype=np.float64)
    cdef double[:,:] delta = np.eye(3, dtype=np.float64)
    cdef double[:,:,:] lab = np.empty((3, nmax, nmax), dtype=np.float64)
    cdef double[:] eigenvalues
    cdef float scalar = (2.0*nmax*nmax)

    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3.0*lab[a,i,j]*lab[b,i,j] - delta[a,b]
            Qab[a,b] /= scalar

    eigenvalues = np.linalg.eigvals(Qab)

    return np.max(eigenvalues)
#=======================================================================
# cythonised and parallelised function
cdef float MC_step_parallel(np.ndarray[double, ndim=2] arr, double Ts,int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    cdef float scale = 0.1+Ts
    cdef float accept = 0.0

    cdef int[:, :] xran = np.random.randint(0,high=nmax, size=(nmax,nmax), dtype=np.int32)
    cdef int[:, :] yran = np.random.randint(0,high=nmax, size=(nmax,nmax), dtype=np.int32)
    cdef double[:, :]aran = np.random.normal(scale=scale, size=(nmax,nmax))

    cdef int i, j, ix, iy
    cdef float ang, en0, en1, boltz, rand_num

    # dividing grid into chunks and determining start and end points for loop
    cdef int chunk_size = nmax // size
    cdef int start = rank * chunk_size
    cdef int end = 0

    if rank < size-1:
        end += (rank+1)*chunk_size
    else:
        end += nmax

    for i in range(start, end):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
                boltz = exp( -(en1 - en0) / Ts )
                rand_num = np.random.uniform(0.0, 1.0)

                if boltz >= rand_num:
                    accept += 1
                else:
                    arr[ix,iy] -= ang

    # reducing output of all processes
    cdef float combined_accept = MPI.COMM_WORLD.allreduce(accept, op=MPI.SUM)

    return combined_accept/(nmax*nmax)
#=======================================================================
def main(program, int nsteps, int nmax, double temp):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    lattice = initdat(nmax)

    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)

    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice,nmax)

    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step_parallel(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial

    if rank == 0:
      print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
      savedat(lattice,nsteps,temp,ratio,energy,order,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 4:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE>".format(sys.argv[0]))
#=======================================================================
