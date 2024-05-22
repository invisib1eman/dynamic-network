#!/bin/bash

lammps_dir=/projects/b1021/Jianshe/codes/lammps/lammps-stable/install3
kim_dir=/projects/b1021/Jianshe/codes/lammps/lammps-stable/build3/kim_build-prefix/
module purge all
module load fftw/3.3.8-openmpi-4.0.5-gcc-10.2.0
module load cuda/11.2.1-gcc-10.2.0
module load hdf5/1.8.12
module load ffmpeg/4.2
module load eigen/3.3.4

export PATH="$PATH:${lammps_dir}/bin"
export MANPATH="$MANPATH:${lammps_dir}/share/man"
export LIBRARY_PATH="$LIBRARY_PATH:${lammps_dir}/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${lammps_dir}/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${kim_dir}/lib"
export LIBRARY_PATH="$LIBRARY_PATH:${lammps_dir}/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${lammps_dir}/lib64"
export C_INCLUDE_PATH="$C_INCLUDE_PATH:${lammps_dir}/include"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:${lammps_dir}/include"
export INCLUDE="$INCLUDE:${lammps_dir}/include"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:${lammps_dir}/lib64/pkgconfig"
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:${lammps_dir}/"
export LAMMPS_POTENTIALS="$LAMMPS_POTENTIALS:${lammps_dir}/share/lammps/potentials"
