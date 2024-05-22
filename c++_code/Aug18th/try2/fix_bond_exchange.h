/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(bond/exchange,FixBondExchange);
// clang-format on
#else

#ifndef LMP_FIX_BOND_EXCHANGE_H
#define LMP_FIX_BOND_EXCHANGE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondExchange : public Fix {
 public:
  FixBondExchange(class LAMMPS *, int, char **);
  ~FixBondExchange();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void post_integrate();
  int modify_param(int, char **);
  double compute_vector(int);
  double memory_usage();

 private:
  double fraction, cutsq, energy_barrier;
  int nmax, tflag, iatomtype, jatomtype, btype;
  int exchange1, exchange2;
  int naccept1, foursome, naccept2;
  int angleflag;

  tagint *partner, *finalpartner;
  double *distsq;

  char *id_temp;
  int *type;
  double **x;

  class NeighList *list;
  class Compute *temperature;
  class RanMars *random;

  double dist_rsq(int, int);
  double pair_eng(int, int);
  double bond_eng(int, int, int);
  double angle_eng(int, int, int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid bond type in fix bond/exchange command

Self-explanatory.

E: Cannot use fix bond/exchange with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Cannot yet use fix bond/exchange with this improper style

This is a current restriction in LAMMPS.

E: Fix bond/exchange needs ghost atoms from further away

This is because the fix needs to walk bonds to a certain distance to
acquire needed info, The comm_modify cutoff command can be used to
extend the communication range.

*/
