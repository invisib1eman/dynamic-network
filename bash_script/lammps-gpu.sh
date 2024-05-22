#!/bin/bash

module purge
module load python-anaconda3
source activate /projects/b1164/software/clingo-py38
module use /projects/b1164/src/spack/share/spack/modules/linux-rhel7-x86_64/
module load lammps/20210310-openmpi-intel

