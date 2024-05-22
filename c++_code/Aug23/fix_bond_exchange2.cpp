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
#include "math_const.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

#define BIG 1.0e20
#define DELTA 16
/* ---------------------------------------------------------------------- */

FixBondExchange::FixBondExchange(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  tflag(0), id_temp(nullptr), type(nullptr), x(nullptr), list(nullptr),
  temperature(nullptr), random(nullptr), partner(nullptr), finalpartner(nullptr), 
  distsq(nullptr), exchanged(nullptr), copy(nullptr), nextpartner(nullptr), finalnextpartner(nullptr)
{

  if (narg < 8) error->all(FLERR,"Illegal fix bond/exchange command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/exchange command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 4;
  global_freq = 1;
  extvector = 0;

  iatomtype = utils::inumeric(FLERR,arg[4],false,lmp);
  jatomtype = utils::inumeric(FLERR,arg[5],false,lmp);
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

  energy_barrier = 0.0;
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

  random = new RanMars(lmp,seed + me);

  comm_forward = MAX(3,3+atom->maxspecial);
  comm_reverse = 3;

  // create a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp",id_temp));
  tflag = 1;

  // allocate arrays local to this fix

  nmax = 0;
  maxexchange = 0;
  partner = finalpartner = nullptr;
  nextpartner = finalnextpartner = nullptr;
  distsq = nullptr;

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

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

  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(nextpartner);
  memory->destroy(finalnextpartner);
  memory->destroy(distsq);
  memory->destroy(exchanged);
  delete [] copy;

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

  lastcheck = -1;

}

/* ---------------------------------------------------------------------- */

void FixBondExchange::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible, possible1, possible2, possible3;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;

  int inext,iprev,ilast,jnext,jprev,jlast,ibond,iangle,jbond,jangle;
  int ibondtype,jbondtype,iangletype,inextangletype,jangletype,jnextangletype;
  tagint itag,inexttag,iprevtag,ilasttag,jtag,jnexttag,jprevtag,jlasttag;
  tagint i1,i2,i3,j1,j2,j3;
  int accept_any;
  double delta,factor;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  if (lastcheck < neighbor->lastcall) check_ghosts();

  comm->forward_comm();
  // compute current temp for Boltzmann factor test

  double t_current = temperature->compute_scalar();


  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(nextpartner);
    memory->destroy(finalnextpartner);
    memory->destroy(distsq);
    nmax = atom->nmax;
    memory->create(partner,nmax,"bond/exchange:partner");
    memory->create(finalpartner,nmax,"bond/exchange:finalpartner");
    memory->create(nextpartner,nmax,"bond/exchange:nextpartner");
    memory->create(finalnextpartner,nmax,"bond/exchange:finalnextpartner");
    memory->create(distsq,nmax,"bond/exchange:distsq");

  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    nextpartner[i] = 0;
    finalnextpartner[i] = 0;
    distsq[i] = BIG;
  }

  // loop over neighbors of my atoms
  // each atom sets one closest eligible partner atom ID to bond with

  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *mask = atom->mask;

  x = atom->x;
  type = atom->type;

  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      if (!(mask[j] & groupbit)) continue;
      if (jtype == itype) continue;
      
      for (k = 0; k < nspecial[i][0]; k++)
        if (special[i][k] == tag[j]) possible = 0;
      if (!possible) continue;

      rsq = dist_rsq(i,j);
      if (rsq >= cutsq || rsq < 0.7569) continue;

      if (rsq < distsq[i] && rsq > 0.7569) {
        partner[i] = tag[j];
        distsq[i] = rsq;
      }
      if (rsq < distsq[j] && rsq > 0.7569) {
        partner[j] = tag[i];
        distsq[j] = rsq;
      }
    }
  }
  // reverse comm of partner info

  if (force->newton_bond) comm->reverse_comm_fix(this);

  commflag = 1;
  comm->forward_comm_fix(this,2);

  int accept = 0;

  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    possible = 0;
    for (m = 0; m < nspecial[i][0]; m++){
      inext = atom->map(special[i][m]);
      if (!(mask[inext] & groupbit)) continue;
      //if (inext >= nlocal || inext < 0) continue;
      if (type[inext] != type[i]) {
        possible = 1;
        break;
      } 
    }
    if (!possible) continue;

    possible1 = 0;
    for (m = 0; m < nspecial[j][0]; m++){
      jnext = atom->map(special[j][m]);
      //if (jnext >= nlocal || jnext < 0) continue;
      if (jnext == i) continue;
      if (type[jnext] != type[j] && (mask[jnext] & groupbit)){
        possible1 = 1;
        break;
      }
    }
      
    // exchange bond
    /*************************************************************/
    if ((possible == 1) && (possible1 == 1) && (exchange2 == 1) && (tag[i] < tag[j])) {
      if (dist_rsq(inext,jnext) >= cutsq || dist_rsq(inext,jnext) < 0.7569) continue;

      delta = energy_barrier;
      delta += pair_eng(i,inext) + pair_eng(j,jnext) -
        pair_eng(i,j) - pair_eng(inext,jnext);
      delta += bond_eng(btype,i,j) + bond_eng(btype,jnext,inext) -
        bond_eng(btype,i,inext) - bond_eng(btype,j,jnext);
      if ((delta < 0.0)) accept = 1;
      else {
        factor = exp(-delta/force->boltz/t_current);
        if ((random->uniform() < factor)) accept = 1;
      }
      if (!accept) continue;
        naccept2++;

      for (ibond = 0; ibond < num_bond[i]; ibond++)
        if (bond_atom[i][ibond] == tag[inext]) bond_atom[i][ibond] = tag[j];
      for (jbond = 0; jbond < num_bond[j]; jbond++)
        if (bond_atom[j][jbond] == tag[jnext]) bond_atom[j][jbond] = tag[i];
      for (ibond = 0; ibond < num_bond[inext]; ibond++)
        if (bond_atom[inext][ibond] == tag[i]) bond_atom[inext][ibond] = tag[jnext];
      for (jbond = 0; jbond < num_bond[jnext]; jbond++)
        if (bond_atom[jnext][jbond] == tag[j]) bond_atom[jnext][jbond] = tag[inext];

      itag = tag[i];
      inexttag = tag[inext];
      jtag = tag[j];
      jnexttag = tag[jnext];

      for (m = 0; m < nspecial[i][0]; m++)
        if (special[i][m] == inexttag) special[i][m] = jtag;
      for (m = 0; m < nspecial[inext][0]; m++)
        if (special[inext][m] == itag) special[inext][m] = jnexttag;
      for (m = 0; m < nspecial[j][0]; m++)
        if (special[j][m] == jnexttag) special[j][m] = itag;
      for (m = 0; m < nspecial[jnext][0]; m++)
        if (special[jnext][m] == jtag) special[jnext][m] = inexttag;
      
      finalpartner[i] = tag[j];
      finalpartner[j] = tag[i];
      finalnextpartner[i] = tag[inext];
      finalnextpartner[j] = tag[jnext];
      
    }

    else if ((possible == 1) && (possible1 == 0) && (exchange1 == 1)) {
      delta = energy_barrier;
      delta += pair_eng(i,inext) - pair_eng(i,j);
      delta += bond_eng(btype,i,j) - bond_eng(btype,i,inext);
      if (delta < 0.0) accept = 1;
      else {
        factor = exp(-delta/force->boltz/t_current);
        if ((random->uniform() < factor)) accept = 1;
      }
      if (!accept) continue;
      naccept1++;

      itag = tag[i];
      inexttag = tag[inext];
      jtag = tag[j];

      possible2=0;
      for (ibond = 0; ibond < num_bond[inext]; ibond++){
        if (bond_atom[inext][ibond] == itag) {
          for (k = ibond; k< num_bond[inext]-1; k++){
            bond_atom[inext][k] = bond_atom[inext][k+1];
            bond_type[inext][k] = bond_type[inext][k+1];
          } 
          num_bond[inext]--;
          possible2=1;
        break;
        }
      }

      if (possible2) {
        bond_type[j][num_bond[j]] = btype;
        bond_atom[j][num_bond[j]] = itag;
        num_bond[j]++;
      }

      for (ibond = 0; ibond < num_bond[i]; ibond++){
        if (bond_atom[i][ibond] == inexttag) {
        bond_atom[i][ibond] = jtag;
        } 
      }

      for (m = 0; m < nspecial[inext][0]; m++)
        if (special[inext][m] == itag) break;
      n3 = nspecial[inext][2];
      for (; m < n3-1; m++) special[inext][m] = special[inext][m+1];
        nspecial[inext][0]--;
        nspecial[inext][1]--;
        nspecial[inext][2]--;

      for (m = nspecial[j][2]; m > nspecial[j][0]; m--) {
        special[j][m] = special[j][m-1];
      }
      special[j][nspecial[j][0]] = itag;
      nspecial[j][0]++;
      nspecial[j][1]++;
      nspecial[j][2]++;

      for (m = 0; m < nspecial[i][0]; m++)
        if (special[i][m] == inexttag) special[i][m] = jtag;
      
      finalpartner[i] = tag[j];
      finalpartner[j] = tag[i];
      finalnextpartner[i] = tag[inext];
      finalnextpartner[j] = 0;
      
    }
    /*************************************************************/

  }

  int naccept;
  naccept = naccept1 + naccept2;
  MPI_Allreduce(&naccept,&accept_any,1,MPI_INT,MPI_SUM,world);
  if (accept_any) next_reneighbor = update->ntimestep;
  if (!accept_any) return;

  commflag = 2;
  comm->forward_comm_fix(this);
  /*
  nexchange = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (nexchange == maxexchange) {
        maxexchange += DELTA;
        memory->grow(exchanged,maxexchange,2,"bond/exchange:exchanged");
      }
      exchanged[nexchange][0] = tag[i];
      exchanged[nexchange][1] = finalpartner[i];
      nexchange++;
    }
  }

  update_topology();*/

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
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondExchange::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 4*nmax * sizeof(tagint);
  bytes += (double)nmax * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

int FixBondExchange::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;

  if (commflag == 1) {
    m = 0;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner[j]).d;
      buf[m++] = ubuf(nextpartner[j]).d;
      buf[m++] = distsq[j];
      //buf[m++] = probability[j];
    }
    return m;
  }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
    buf[m++] = ubuf(finalnextpartner[j]).d;
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  if (commflag == 1) {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      partner[i] = (tagint) ubuf(buf[m++]).i;
      nextpartner[i] = (tagint) ubuf(buf[m++]).i;
      distsq[i] = buf[m++];
      //probability[i] = buf[m++];
    }

  } else {

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      finalpartner[i] = (tagint) ubuf(buf[m++]).i;
      finalnextpartner[i] = (tagint) ubuf(buf[m++]).i;
      ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (j = 0; j < ns; j++)
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBondExchange::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = ubuf(nextpartner[i]).d;
    buf[m++] = distsq[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    if (buf[m+1] < distsq[j]) {
      partner[j] = (tagint) ubuf(buf[m++]).i;
      nextpartner[j] = (tagint) ubuf(buf[m++]).i;
      distsq[j] = buf[m++];
    } else m += 3;
  }
}

void FixBondExchange::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall)
    error->all(FLERR,"Fix bond/break needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and broken bonds
   influenced = 1 if atom's topology is affected by any broken bond
     yes if is one of 2 atoms in bond
     yes if both atom IDs appear in atom's special list
     else no
   if influenced:
     check for angles/dihedrals/impropers to break due to specific broken bonds
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
------------------------------------------------------------------------- */

void FixBondExchange::update_topology()
{
  int i,j,k,n,influence,influenced,found;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  //nangles = 0;
  //ndihedrals = 0;
  //nimpropers = 0;

  //printf("NBREAK %d: ",nbreak);
  //for (i = 0; i < nbreak; i++)
  //  printf(" %d %d,",broken[i][0],broken[i][1]);
  //printf("\n");

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < nexchange; j++) {
      id1 = exchanged[j][0];
      id2 = exchanged[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2) influence = 1;
      else {
        n = nspecial[i][1];
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) {
            influence = 1;
            break;
          }
      }
      if (!influence) continue;
      influenced = 1;

      //if (angleflag) break_angles(i,id1,id2);
      //if (dihedralflag) break_dihedrals(i,id1,id2);
      //if (improperflag) break_impropers(i,id1,id2);
    }

    if (influenced) rebuild_special_one(i);
  }

  int newton_bond = force->newton_bond;
  /****************
  int all;
  if (angleflag) {
    MPI_Allreduce(&nangles,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 3;
    atom->nangles -= all;
  }
  if (dihedralflag) {
    MPI_Allreduce(&ndihedrals,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->ndihedrals -= all;
  }
  if (improperflag) {
    MPI_Allreduce(&nimpropers,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->nimpropers -= all;
  }
  *************/
}

void FixBondExchange::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  /***
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);*/

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  /*for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);*/

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates
------------------------------------------------------------------------- */

int FixBondExchange::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}