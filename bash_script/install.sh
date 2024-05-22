#!/bin/bash

module purge all
module load cmake/3.15.4
module load fftw/3.3.8-openmpi-4.0.5-gcc-10.2.0
module load cuda/11.2.1-gcc-10.2.0
module load hdf5/1.8.12
module load ffmpeg/4.2
module load eigen/3.3.4

export dir="/projects/b1021/Jianshe/codes/lammps/lammps-stable/lammps-29Sep2021/cmake"
export indir="/projects/b1021/Jianshe/codes/lammps/lammps-stable/install3"

cmake -C ${dir}/presets/most.cmake -C ${dir}/presets/nolib.cmake -D CMAKE_INSTALL_PREFIX=${indir}  ${dir}
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
make -j 4

make install

## tutorial, page28 with cmake, 52

 







