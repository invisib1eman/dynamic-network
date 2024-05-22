#!/bin/bash

#SBATCH --job-name="install_lammps"     ## When you run squeue -u NETID this is how you can identify the job
#SBATCH -A b1030 	            ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH -p buyin  	            ### PARTITION (buyin, short, normal, etc)
#SBATCH -N 1 		            ## how many computers do you need
#SBATCH -t 12:00:00             ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --ntasks-per-node=28     ## how many cpus or processors do you need on each computer
#SBATCH -e errlog.%j 
#SBATCH -o outlog.%j  ## standard out and standard error goes to this file

module purge all
module load cmake/3.15.4
module load fftw/3.3.8-openmpi-4.0.5-gcc-10.2.0
module load cuda/11.2.1-gcc-10.2.0
module load hdf5/1.8.12
module load ffmpeg/4.2
module load eigen/3.3.4


cd $SLURM_SUBMIT_DIR
export dir="/projects/b1021/Jianshe/codes/lammps/lammps-stable/lammps-29Sep2021/cmake"
export indir="/projects/b1021/Jianshe/codes/lammps/lammps-stable/install"

cmake -C ${dir}/presets/most.cmake -C ${dir}/presets/nolib.cmake -D CMAKE_INSTALL_PREFIX=${indir}  ${dir}
cmake -D LAMMPS_MACHINE=exchange .
cmake -D BUILD_TYPE=Release .
cmake -D DOWNLOAD_KIM=on .
cmake -D DOWNLOAD_VORO=on .
cmake -D PKG_GPU=on GPU_API=cuda .
cmake -D PKG_H5MD=on .
cmake -D BUILD_SHARED_LIBS=on .
cmake -D PKG_COMPRESS=on .
cmake -D PKG_MESSAGE=on .
cmake -D PKG_MPIIO=on .
cmake -D PKG_VORONOI=on .
cmake -D PKG_COLVARS=on .
#cmake -D PKG_INTEL=on .
cmake -D PKG_KIM=on .
cmake -D PKG_MOLFILE=on .
cmake -D PKG_PHONON=on .
cmake -D PKG_PTM=on .
cmake -D PKG_QTB=on .
cmake -D PKG_SMTBQ=on .
cmake -D PKG_TALLY=on .
cmake -D FFT_SINGLE=off .
make -j 28

make install

## tutorial, page28 with cmake, 52

 







