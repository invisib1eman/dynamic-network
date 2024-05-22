// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_bond_exchange.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBondExchange::FixBondExchange(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  tflag(0), alist(nullptr), id_temp(nullptr), type(nullptr), x(nullptr), list(nullptr),
  temperature(nullptr), random(nullptr)
{
  
  if (narg < 8) error->all(FLERR,"Illegal fix bond/exchange command");

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/exchange command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 4;
  global_freq = 1;
  extvector = 0;

  iatomtype = utils::numeric(FLERR,arg[4],false,lmp);
  jatomtype = utils::numeric(FLERR,arg[5],false,lmp);
  double cutoff = utils::numeric(FLERR,arg[6],false,lmp);
  btype = utils::inumeric(FLERR,arg[7],false,lmp);

  if (iatomtype < 1 || iatomtype > atom->ntypes ||
      jatomtype < 1 || jatomtype > atom->ntypes || iatomtype==jatomtype)
    error->all(FLERR,"Invalid atom type in fix bond/exchange command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/exchange command");
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/exchange command");

  cutsq = cutoff*cutoff;

  // optional keywords

  energy_barrier = 0;
  exchange2 = 1;
  exchange1 = 1;
  fraction = 1.0;
  int seed = 12345;

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"barrier") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/exchange command");
      energy_barrier = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (energy_barrier < 0) error->all(FLERR,"Illegal fix bond/exchange command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"frac") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/exchange command");
      fraction = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      seed = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (fraction < 0.0 || fraction > 1.0)
        error->all(FLERR,"Illegal fix bond/exchange command");
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/exchange command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"exchange") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/exchange command");
      exchange1 = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      exchange2 = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (exchange1 < 0 || exchange2 < 0)
        error->all(FLERR,"Illegal fix bond/exchange command");
      iarg += 3;
    } else error->all(FLERR,"Illegal fix bond/exchange command");
  }

  // error check

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix bond/exchange with non-molecular systems");


  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + comm->me);

  // create a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp",id_temp));
  tflag = 1;

  // initialize atom list

  nmax = 0;
  alist = nullptr;
  
  naccept1 = foursome = 0;
  naccept2 = 0;

}

/* ---------------------------------------------------------------------- */

FixBondExchange::~FixBondExchange()
{
  delete random;

  // delete temperature if fix created it

  if (tflag) modify->delete_compute(id_temp);
  delete [] id_temp;

  memory->destroy(alist);
}

/* ---------------------------------------------------------------------- */

int FixBondExchange::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::init()
{
  // require an atom style with molecule IDs

  if (atom->molecule == nullptr)
    error->all(FLERR,
               "Must use atom style with molecule IDs with fix bond/exchange");

  int icompute = modify->find_compute(id_temp);
  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix bond/exchange does not exist");
  temperature = modify->compute[icompute];

  // pair and bonds must be defined
  // no dihedral or improper potentials allowed
  // special bonds must be 0 1 1

  if (force->pair == nullptr || force->bond == nullptr)
    error->all(FLERR,"Fix bond/exchange requires pair and bond styles");

  if (force->pair->single_enable == 0)
    error->all(FLERR,"Pair style does not support fix bond/exchange");

  if (force->angle == nullptr && atom->nangles > 0 && comm->me == 0)
    error->warning(FLERR,"Fix bond/exchange will not preserve correct angle "
                   "topology because no angle_style is defined");

  if (force->dihedral || force->improper)
    error->all(FLERR,"Fix bond/exchange cannot use dihedral or improper styles");

  if (force->special_lj[1] != 0.0 || force->special_lj[2] != 1.0 ||
      force->special_lj[3] != 1.0)
    error->all(FLERR,"Fix bond/exchange requires special_bonds = 0,1,1");

  // need a half neighbor list, built every Nevery steps

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  // zero out stats

  naccept1 = foursome = 0;
  naccept2 = 0;

  angleflag = 0;
  if (force->angle) angleflag = 1;
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   look for and perform exchanges
   NOTE: used to do this every pre_neighbor(), but think that is a bug
         b/c was doing it after exchange() and before neighbor->build()
         which is when neigh lists are actually out-of-date or even bogus,
         now do it based on user-specified Nevery, and trigger reneigh
         if any exchanges performed, like fix bond/create
------------------------------------------------------------------------- */

void FixBondExchange::post_integrate()
{
  int i,j,k,m,n,ii,jj,n3,inum,jnum,itype,jtype,possible;
  int inext,iprev,ilast,jnext,jprev,jlast,ibond,iangle,jbond,jangle;
  int ibondtype,jbondtype,iangletype,inextangletype,jangletype,jnextangletype;
  tagint itag,inexttag,iprevtag,ilasttag,jtag,jnexttag,jprevtag,jlasttag;
  tagint i1,i2,i3,j1,j2,j3;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double delta,factor;
  int accept_any, accept_any1, accept_any2;

  if (update->ntimestep % nevery) return;

  // compute current temp for Boltzmann factor test

  double t_current = temperature->compute_scalar();

  // local ptrs to atom arrays

  tagint *tag = atom->tag;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int **angle_type = atom->angle_type;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int newton_bond = force->newton_bond;
  int nlocal = atom->nlocal;

  type = atom->type;
  x = atom->x;

  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // randomize list of my owned atoms that are in fix group
  // grow atom list if necessary

  if (atom->nmax > nmax) {
    memory->destroy(alist);
    nmax = atom->nmax;
    memory->create(alist,nmax,"bondexchange:alist");
  }
  
  int neligible = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit)
      alist[neligible++] = i;
  }

  int tmp;
  for (i = 0; i < neligible; i++) {
    j = static_cast<int> (random->uniform() * neligible);
    tmp = alist[i];
    alist[i] = alist[j];
    alist[j] = tmp;
  }

  int ntest = static_cast<int> (fraction * neligible);
  
  int accept = 0;

  for (int itest = 0; itest < ntest; itest++) {
    i = alist[itest];
    if (!(mask[i] & groupbit)) continue;
    if (i >= nlocal) continue;
    itype = type[i];

    possible = 0;
    for (m = 0; m < nspecial[i][0]; m++){
      inext = atom->map(special[i][m]);
      if (!(mask[inext] & groupbit)) continue;
      if (inext >= nlocal || inext < 0) continue;
      if (type[inext] != itype) {
        possible = 1;
        break;
      } 
    }
    if (!possible) continue;

    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++){
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      if (!(mask[j] & groupbit)) continue;
      if (itype != jtype) continue;
      if (j >= nlocal) continue;

      possible = 0;
      for (m = 0; m < nspecial[j][0]; m++){
        jnext = atom->map(special[j][m]);
        if (type[jnext] != jtype && (mask[jnext] & groupbit)){
          possible = 1;
          break;
        }
      }

      if ((possible == 0) && (exchange1 == 1)){
        if (dist_rsq(j,inext) > cutsq || dist_rsq(j,inext) < 0.7569) continue;
        accept = 0;
        foursome++;
        delta = energy_barrier;
        delta += pair_eng(i,inext) - pair_eng(inext,j);
        delta += bond_eng(btype,j,inext) - bond_eng(btype,i,inext);
        if (delta < 0.0) accept = 1;
        else {
          factor = exp(-delta/force->boltz/t_current);
          if (random->uniform() < factor) accept = 1;
        }
        goto done0;
      
      }
        
      else if ((possible == 1) && (exchange2 == 1)){

        possible = 0;
        for (m = 0; m < nspecial[j][0]; m++){
          jnext = atom->map(special[j][m]);
          if (jnext >= nlocal || jnext < 0) continue;
          if (type[jnext] != jtype && (mask[jnext] & groupbit)){
            possible = 1;
            break;
          }
        }
        if (!possible) continue;
       
        if (inext == jnext || inext == j) continue;
        if (dist_rsq(i,jnext) > cutsq || dist_rsq(i,jnext) < 0.7569) continue;
        if (dist_rsq(j,inext) > cutsq || dist_rsq(j,inext) < 0.7569) continue;

        accept = 0;
        foursome++;
        delta = energy_barrier;
        delta += pair_eng(i,inext) + pair_eng(j,jnext) -
          pair_eng(i,jnext) - pair_eng(inext,j);
        delta += bond_eng(btype,i,jnext) + bond_eng(btype,j,inext) -
          bond_eng(btype,i,inext) - bond_eng(btype,j,jnext);
        if (delta < 0.0) accept = 1;
        else {
          factor = exp(-delta/force->boltz/t_current);
          if (random->uniform() < factor) accept = 1;
        }
        goto done1;
        
      }
    }
  } 
  
  done0:

  // trigger immediate reneighboring if exchanges occurred on one or more procs

  MPI_Allreduce(&accept,&accept_any,1,MPI_INT,MPI_SUM,world);
  if (accept_any) next_reneighbor = update->ntimestep;

  if (!accept) return;
  naccept1++;

  // change bond partners of affected atoms
  // on atom i: bond i-inext changes to i-jnext
  // on atom j: bond j-jnext changes to j-inext
  // on atom inext: bond inext-i changes to inext-j
  // on atom jnext: bond jnext-j changes to jnext-i

  itag = tag[i];
  inexttag = tag[inext];
  jtag = tag[j];
  
  possible = 0;
  for (ibond = 0; ibond < num_bond[i]; ibond++){
    if (bond_atom[i][ibond] == inexttag) {
      for (k = ibond; k< num_bond[i]-1; k++){
        bond_atom[i][k] = bond_atom[i][k+1];
        bond_type[i][k] = bond_type[i][k+1];
      } 
      num_bond[i]--;
      possible = 1;
      break;
    }
  }

  if (possible){
    bond_type[j][num_bond[j]] = btype;
    bond_atom[j][num_bond[j]] = inexttag;
    num_bond[j]++;
  }
 
  for (ibond = 0; ibond < num_bond[inext]; ibond++)
    if (bond_atom[inext][ibond] == itag) bond_atom[inext][ibond] = jtag;
 
  // set global tags of 4 atoms in bonds

  // change 1st special neighbors of affected atoms: i,j,inext,jnext
  // don't need to change 2nd/3rd special neighbors for any atom
  //   since special bonds = 0 1 1 means they are never used

  for (m = 0; m < nspecial[i][0]; m++)
    if (special[i][m] == inexttag) break;
  n3 = nspecial[i][2];
  for (; m < n3-1; m++) special[i][m] = special[i][m+1];
  nspecial[i][0]--;
  nspecial[i][1]--;
  nspecial[i][2]--;

  for (m = nspecial[j][2]; m > nspecial[j][0]; m--) {
    special[j][m] = special[j][m-1];
  }
  special[j][nspecial[j][0]] = inexttag;
  nspecial[j][0]++;
  nspecial[j][1]++;
  nspecial[j][2]++;

  for (m = 0; m < nspecial[inext][0]; m++)
    if (special[inext][m] == itag) special[inext][m] = jtag;
  
 return;

 done1:

  // trigger immediate reneighboring if exchanges occurred on one or more procs

  MPI_Allreduce(&accept,&accept_any,1,MPI_INT,MPI_SUM,world);
  if (accept_any) next_reneighbor = update->ntimestep;

  if (!accept) return;
  naccept2++;

  // change bond partners of affected atoms
  // on atom i: bond i-inext changes to i-jnext
  // on atom j: bond j-jnext changes to j-inext
  // on atom inext: bond inext-i changes to inext-j
  // on atom jnext: bond jnext-j changes to jnext-i

  for (ibond = 0; ibond < num_bond[i]; ibond++)
    if (bond_atom[i][ibond] == tag[inext]) bond_atom[i][ibond] = tag[jnext];
  for (jbond = 0; jbond < num_bond[j]; jbond++)
    if (bond_atom[j][jbond] == tag[jnext]) bond_atom[j][jbond] = tag[inext];
  for (ibond = 0; ibond < num_bond[inext]; ibond++)
    if (bond_atom[inext][ibond] == tag[i]) bond_atom[inext][ibond] = tag[j];
  for (jbond = 0; jbond < num_bond[jnext]; jbond++)
    if (bond_atom[jnext][jbond] == tag[j]) bond_atom[jnext][jbond] = tag[i];

  // set global tags of 4 atoms in bonds

  itag = tag[i];
  inexttag = tag[inext];

  jtag = tag[j];
  jnexttag = tag[jnext];

  // change 1st special neighbors of affected atoms: i,j,inext,jnext
  // don't need to change 2nd/3rd special neighbors for any atom
  //   since special bonds = 0 1 1 means they are never used

  for (m = 0; m < nspecial[i][0]; m++)
    if (special[i][m] == inexttag) special[i][m] = jnexttag;
  for (m = 0; m < nspecial[j][0]; m++)
    if (special[j][m] == jnexttag) special[j][m] = inexttag;
  for (m = 0; m < nspecial[inext][0]; m++)
    if (special[inext][m] == itag) special[inext][m] = jtag;
  for (m = 0; m < nspecial[jnext][0]; m++)
    if (special[jnext][m] == jtag) special[jnext][m] = itag;

  // done if no angles

  if (!angleflag) return;

  // set global tags of 4 additional atoms in angles, 0 if no angle

  if (iprev >= 0) iprevtag = tag[iprev];
  else iprevtag = 0;
  if (ilast >= 0) ilasttag = tag[ilast];
  else ilasttag = 0;

  if (jprev >= 0) jprevtag = tag[jprev];
  else jprevtag = 0;
  if (jlast >= 0) jlasttag = tag[jlast];
  else jlasttag = 0;

  for (iangle = 0; iangle < num_angle[i]; iangle++) {
    i1 = angle_atom1[i][iangle];
    i2 = angle_atom2[i][iangle];
    i3 = angle_atom3[i][iangle];

    if (i1 == iprevtag && i2 == itag && i3 == inexttag)
      angle_atom3[i][iangle] = jnexttag;
    else if (i1 == inexttag && i2 == itag && i3 == iprevtag)
      angle_atom1[i][iangle] = jnexttag;
    else if (i1 == itag && i2 == inexttag && i3 == ilasttag) {
      angle_atom2[i][iangle] = jnexttag;
      angle_atom3[i][iangle] = jlasttag;
    } else if (i1 == ilasttag && i2 == inexttag && i3 == itag) {
      angle_atom1[i][iangle] = jlasttag;
      angle_atom2[i][iangle] = jnexttag;
    }
  }

  for (jangle = 0; jangle < num_angle[j]; jangle++) {
    j1 = angle_atom1[j][jangle];
    j2 = angle_atom2[j][jangle];
    j3 = angle_atom3[j][jangle];

    if (j1 == jprevtag && j2 == jtag && j3 == jnexttag)
      angle_atom3[j][jangle] = inexttag;
    else if (j1 == jnexttag && j2 == jtag && j3 == jprevtag)
      angle_atom1[j][jangle] = inexttag;
    else if (j1 == jtag && j2 == jnexttag && j3 == jlasttag) {
      angle_atom2[j][jangle] = inexttag;
      angle_atom3[j][jangle] = ilasttag;
    } else if (j1 == jlasttag && j2 == jnexttag && j3 == jtag) {
      angle_atom1[j][jangle] = ilasttag;
      angle_atom2[j][jangle] = inexttag;
    }
  }

  for (iangle = 0; iangle < num_angle[inext]; iangle++) {
    i1 = angle_atom1[inext][iangle];
    i2 = angle_atom2[inext][iangle];
    i3 = angle_atom3[inext][iangle];

    if (i1 == iprevtag && i2 == itag && i3 == inexttag) {
      angle_atom1[inext][iangle] = jprevtag;
      angle_atom2[inext][iangle] = jtag;
    } else if (i1 == inexttag && i2 == itag && i3 == iprevtag) {
      angle_atom2[inext][iangle] = jtag;
      angle_atom3[inext][iangle] = jprevtag;
    } else if (i1 == itag && i2 == inexttag && i3 == ilasttag)
      angle_atom1[inext][iangle] = jtag;
    else if (i1 == ilasttag && i2 == inexttag && i3 == itag)
      angle_atom3[inext][iangle] = jtag;
  }

  for (jangle = 0; jangle < num_angle[jnext]; jangle++) {
    j1 = angle_atom1[jnext][jangle];
    j2 = angle_atom2[jnext][jangle];
    j3 = angle_atom3[jnext][jangle];

    if (j1 == jprevtag && j2 == jtag && j3 == jnexttag) {
      angle_atom1[jnext][jangle] = iprevtag;
      angle_atom2[jnext][jangle] = itag;
    } else if (j1 == jnexttag && j2 == jtag && j3 == jprevtag) {
      angle_atom2[jnext][jangle] = itag;
      angle_atom3[jnext][jangle] = iprevtag;
    } else if (j1 == jtag && j2 == jnexttag && j3 == jlasttag)
      angle_atom1[jnext][jangle] = itag;
    else if (j1 == jlasttag && j2 == jnexttag && j3 == jtag)
      angle_atom3[jnext][jangle] = itag;
  }

  // done if newton bond set

  if (newton_bond) return;

  for (iangle = 0; iangle < num_angle[iprev]; iangle++) {
    i1 = angle_atom1[iprev][iangle];
    i2 = angle_atom2[iprev][iangle];
    i3 = angle_atom3[iprev][iangle];

    if (i1 == iprevtag && i2 == itag && i3 == inexttag)
      angle_atom3[iprev][iangle] = jnexttag;
    else if (i1 == inexttag && i2 == itag && i3 == iprevtag)
      angle_atom1[iprev][iangle] = jnexttag;
  }

  for (jangle = 0; jangle < num_angle[jprev]; jangle++) {
    j1 = angle_atom1[jprev][jangle];
    j2 = angle_atom2[jprev][jangle];
    j3 = angle_atom3[jprev][jangle];

    if (j1 == jprevtag && j2 == jtag && j3 == jnexttag)
      angle_atom3[jprev][jangle] = inexttag;
    else if (j1 == jnexttag && j2 == jtag && j3 == jprevtag)
      angle_atom1[jprev][jangle] = inexttag;
  }

  for (iangle = 0; iangle < num_angle[ilast]; iangle++) {
    i1 = angle_atom1[ilast][iangle];
    i2 = angle_atom2[ilast][iangle];
    i3 = angle_atom3[ilast][iangle];

    if (i1 == itag && i2 == inexttag && i3 == ilasttag)
      angle_atom1[ilast][iangle] = jtag;
    else if (i1 == ilasttag && i2 == inexttag && i3 == itag)
      angle_atom3[ilast][iangle] = jtag;
  }

  for (jangle = 0; jangle < num_angle[jlast]; jangle++) {
    j1 = angle_atom1[jlast][jangle];
    j2 = angle_atom2[jlast][jangle];
    j3 = angle_atom3[jlast][jangle];

    if (j1 == jtag && j2 == jnexttag && j3 == jlasttag)
      angle_atom1[jlast][jangle] = itag;
    else if (j1 == jlasttag && j2 == jnexttag && j3 == jtag)
      angle_atom3[jlast][jangle] = itag;
  }
  
}

/* ---------------------------------------------------------------------- */

int FixBondExchange::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (tflag) {
      modify->delete_compute(id_temp);
      tflag = 0;
    }
    delete [] id_temp;
    id_temp = utils::strdup(arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,"Fix_modify temperature ID does not "
                 "compute temperature");
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR,"Group for fix_modify temp != fix group");
    return 2;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   compute squared distance between atoms I,J
   must use minimum_image since J was found thru atom->map()
------------------------------------------------------------------------- */

double FixBondExchange::dist_rsq(int i, int j)
{
  double delx = x[i][0] - x[j][0];
  double dely = x[i][1] - x[j][1];
  double delz = x[i][2] - x[j][2];
  domain->minimum_image(delx,dely,delz);
  return (delx*delx + dely*dely + delz*delz);
}

/* ----------------------------------------------------------------------
   return pairwise interaction energy between atoms I,J
   will always be full non-bond interaction, so factors = 1 in single() call
------------------------------------------------------------------------- */

double FixBondExchange::pair_eng(int i, int j)
{
  double tmp;
  double rsq = dist_rsq(i,j);
  return force->pair->single(i,j,type[i],type[j],rsq,1.0,1.0,tmp);
}

/* ---------------------------------------------------------------------- */

double FixBondExchange::bond_eng(int btype, int i, int j)
{
  double tmp;
  double rsq = dist_rsq(i,j);
  return force->bond->single(btype,rsq,i,j,tmp);
}

/* ---------------------------------------------------------------------- */

double FixBondExchange::angle_eng(int atype, int i, int j, int k)
{
  // test for non-existent angle at end of chain

  if (i == -1 || k == -1) return 0.0;
  return force->angle->single(atype,i,j,k);
}

/* ----------------------------------------------------------------------
   return bond exchangeping stats
   n = 1 is # of exchange A-B/B
   n = 2 is # of exchange A-B/A-B
   n = 3 is # of exchange A-B/B + A-B/A-B
   n = 4 is # of attempted exchanges
------------------------------------------------------------------------- */

double FixBondExchange::compute_vector(int n)
{
  double one,all;
  if (n == 0) one = naccept1;
  else if (n == 1) one = naccept2;
  else if (n == 2) one = naccept1 + naccept2;
  else if ( n == 3) one = foursome;
  MPI_Allreduce(&one,&all,1,MPI_DOUBLE,MPI_SUM,world);
  return all;
}

/* ----------------------------------------------------------------------
   memory usage of alist
------------------------------------------------------------------------- */

double FixBondExchange::memory_usage()
{
  double bytes = (double)nmax * sizeof(int);
  return bytes;
}
