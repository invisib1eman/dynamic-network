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

#include "v3_fix_bond_exchange.h"

#include <cstring>
#include <cmath>
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
#include "respa.h"
#include "update.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

#define BIG 1.0e20
#define DELTA 16
/* ---------------------------------------------------------------------- */

FixBondExchange::FixBondExchange(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  partner(nullptr), finalpartner(nullptr), distsq(nullptr), probability(nullptr),
  created(nullptr), copy(nullptr), random(nullptr), list(nullptr), tflag(0), 
  id_temp(nullptr), temperature(nullptr), type(nullptr), x(nullptr)
{
  if (narg < 6) error->all(FLERR,"Illegal fix bond/exchange command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/exchange command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = utils::inumeric(FLERR,arg[4],false,lmp);
  cutoff = utils::numeric(FLERR,arg[5],false,lmp);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/exchange command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/exchange command");

  cutsq = cutoff*cutoff;

  // optional keywords

  energy_barrier = 0;
  fraction = 1.0;
  int seed = 12345;

  int iarg = 6;
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
    } else error->all(FLERR,"Illegal fix bond/exchange command");
  }

  // error check

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix bond/exchange with non-molecular systems");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);

  // set comm sizes needed by this fix
  // forward is big due to comm of exchaged bonds and 1-2 neighbors

  comm_forward = MAX(2,2+atom->maxspecial);
  comm_reverse = 2;

// exchange a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp",id_temp));
  tflag = 1;

  // allocate arrays local to this fix

  nmax = 0;

  maxexchange = 0;
  maxcreate = 0;
  created = nullptr;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special_one() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // zero out stats

  exchangecount = 0;
  exchangecounttotal = 0;

  createcount = 0;
  createcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondExchange::~FixBondExchange()
{
  delete random;

  // delete locally stored arrays
  if (tflag) modify->delete_compute(id_temp);
  delete [] id_temp;

  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(distsq);
  memory->destroy(created);
  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixBondExchange::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::init()
{
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  int icompute = modify->find_compute(id_temp);
  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix bond/exchange does not exist");
  temperature = modify->compute[icompute];

  // need a half neighbor list, built every Nevery steps

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  // enable angle/dihedral/improper exchangeing if any defined

  if (atom->nangles) angleflag = 1;
  else angleflag = 0;
  if (atom->ndihedrals) dihedralflag = 1;
  else dihedralflag = 0;
  if (atom->nimpropers) improperflag = 1;
  else improperflag = 0;

  if (force->improper) {
    if (force->improper_match("class2") || force->improper_match("ring"))
      error->all(FLERR,"Cannot yet use fix bond/exchange with this "
                 "improper style");
  }

  lastcheck = -1;

  // DEBUG
  //print_bb();
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
  
  double delta,factor;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // compute current temp for Boltzmann factor test

  double t_current = temperature->compute_scalar();

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check

  //if (lastcheck < neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(distsq);
    nmax = atom->nmax;
    memory->exchange(partner,nmax,"bond/exchange:partner");
    memory->exchange(finalpartner,nmax,"bond/exchange:finalpartner");
    memory->exchange(distsq,nmax,"bond/exchange:distsq");
    probability = distsq;
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
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

      // do not allow a duplicate bond to be created
      // check 1-2 neighbors of atom I

      for (k = 0; k < nspecial[i][0]; k++)
        if (special[i][k] == tag[j]) possible = 0;
      if (!possible) continue;

      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      //rsq = dist_rsq(i,j);
      if (rsq >= cutsq) continue;

      if (rsq < distsq[i]) {
        partner[i] = tag[j];
        distsq[i] = rsq;
      }
      if (rsq < distsq[j]) {
        partner[j] = tag[i];
        distsq[j] = rsq;
      }
    }
  }

  // reverse comm of partner info

  if (force->newton_bond) comm->reverse_comm_fix(this);

  // each atom now knows its winning partner
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it

  if (fraction < 1.0) {
    for (i = 0; i < nlocal; i++)
      if (partner[i]) probability[i] = random->uniform();
  }

  commflag = 1;
  comm->forward_comm_fix(this,3);

  // exchange bonds
  // if both atoms list each other as winning bond partner
  // and probability constraint is satisfied

  int newton_bond = force->newton_bond;

  nexchange = 0;
  ncreate = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    possible = 0;
    for (m = 0; m < nspecial[i][0]; m++){
      inext = atom->map(special[i][m]);
      if (!(mask[inext] & groupbit)) continue;
      if ((inext < nlocal) && (inext > 0) && type[inext] != type[i]) {
        possible = 1;
        break;
      } 
    }
    
    possible1 = 0;
    for (m = 0; m < nspecial[j][0]; m++){
      jnext = atom->map(special[j][m]);
      if (jnext >= nlocal || jnext < 0) continue;
      if (type[jnext] != type[j] && (mask[jnext] & groupbit)){
        possible1 = 1;
        break;
      }
    }

    if ((possible == 0) && (possible1 == 0)) {
      if (!newton_bond || tag[i] < tag[j]) {
        if (num_bond[i] == atom->bond_per_atom)
          error->one(FLERR,"New bond exceeded bonds per atom in fix bond/exchange");
        bond_type[i][num_bond[i]] = btype;
        bond_atom[i][num_bond[i]] = tag[j];
        num_bond[i]++;
      }

      slist = special[i];
      n1 = nspecial[i][0];
      n2 = nspecial[i][1];
      n3 = nspecial[i][2];
      for (m = n1; m < n3; m++)
        if (slist[m] == tag[j]) break;
      if (m < n3) {
        for (n = m; n < n3-1; n++) slist[n] = slist[n+1];
        n3--;
      if (m < n2) n2--;
      }
      if (n3 == atom->maxspecial)
        error->one(FLERR,
                 "New bond exceeded special list size in fix bond/exchange");
      for (m = n3; m > n1; m--) slist[m] = slist[m-1];
      slist[n1] = tag[j];
      nspecial[i][0] = n1+1;
      nspecial[i][1] = n2+1;
      nspecial[i][2] = n3+1;

      finalpartner[i] = tag[j];
      finalpartner[j] = tag[i];

      if (tag[i] < tag[j]) ncreate++;

    }

    // exchange bond
    /*************************************************************/
    if ((possible == 1) && (possible1 == 1)) {
      if (inext == jnext || inext == j) continue;
      if (dist_rsq(inext,jnext) >= cutsq) continue;

      delta = energy_barrier;
      delta += pair_eng(i,inext) + pair_eng(j,jnext) -
        pair_eng(i,j) - pair_eng(inext,jnext);
      delta += bond_eng(btype,i,j) + bond_eng(btype,jnext,inext) -
        bond_eng(btype,i,inext) - bond_eng(btype,j,jnext);
      if ((delta < 0.0) && (tag[i] < tag[j])) nexchange++;
      else {
        factor = exp(-delta/force->boltz/t_current);
        if ((random->uniform() < factor) && (tag[i] < tag[j])) nexchange++;
      }

      for (ibond = 0; ibond < num_bond[i]; ibond++)
        if (bond_atom[i][ibond] == tag[inext]) bond_atom[i][ibond] = tag[j];
      for (ibond = 0; ibond < num_bond[inext]; ibond++)
        if (bond_atom[inext][ibond] == tag[i]) bond_atom[inext][ibond] = tag[jnext];

      itag = tag[i];
      inexttag = tag[inext];
      jtag = tag[j];
      jnexttag = tag[jnext];

      for (m = 0; m < nspecial[i][0]; m++)
        if (special[i][m] == inexttag) special[i][m] = jtag;
      for (m = 0; m < nspecial[inext][0]; m++)
        if (special[inext][m] == itag) special[inext][m] = jnexttag;
      
    }

    if (((possible == 1) && (possible1 == 0)) && (tag[i] < tag[j])) {
      delta = energy_barrier;
      delta += pair_eng(i,inext) - pair_eng(i,j);
      delta += bond_eng(btype,i,j) - bond_eng(btype,i,inext);
      if (delta < 0.0) nexchange++;
      else {
        factor = exp(-delta/force->boltz/t_current);
        if ((random->uniform() < factor) && (tag[i] < tag[j])) nexchange++;
      }

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
      
    }

    if ((possible == 0) && (possible1 == 1) && (tag[i] < tag[j])) {
      delta = energy_barrier;
      delta += pair_eng(j,jnext) - pair_eng(i,j);
      delta += bond_eng(btype,i,j) - bond_eng(btype,j,jnext);
      if (delta < 0.0) nexchange++;
      else {
        factor = exp(-delta/force->boltz/t_current);
        if (random->uniform() < factor) nexchange++;
      }

      itag = tag[i];
      inexttag = tag[jnext];
      jtag = tag[j];

      possible3=0;
      for (ibond = 0; ibond < num_bond[jnext]; ibond++){
        if (bond_atom[jnext][ibond] == jtag) {
          for (k = ibond; k< num_bond[jnext]-1; k++){
            bond_atom[jnext][k] = bond_atom[jnext][k+1];
            bond_type[jnext][k] = bond_type[jnext][k+1];
          } 
          num_bond[jnext]--;
          possible3=1;
          break;
        }
      }

      if (possible3) {
        bond_type[i][num_bond[i]] = btype;
        bond_atom[i][num_bond[i]] = jtag;
        num_bond[i]++;
      }

      for (ibond = 0; ibond < num_bond[j]; ibond++){
        if (bond_atom[j][ibond] == jnexttag) {
        bond_atom[j][ibond] = itag;
        } 
      }

      for (m = 0; m < nspecial[jnext][0]; m++)
        if (special[jnext][m] == jtag) break;
      n3 = nspecial[jnext][2];
      for (; m < n3-1; m++) special[jnext][m] = special[jnext][m+1];
        nspecial[jnext][0]--;
        nspecial[jnext][1]--;
        nspecial[jnext][2]--;

      for (m = nspecial[i][2]; m > nspecial[i][0]; m--) {
        special[i][m] = special[i][m-1];
      }
      special[i][nspecial[i][0]] = jtag;
      nspecial[i][0]++;
      nspecial[i][1]++;
      nspecial[i][2]++;

      for (m = 0; m < nspecial[j][0]; m++)
        if (special[j][m] == jnexttag) special[j][m] = itag;
      
    }
    
    /*************************************************************/

  }

  // tally stats

  MPI_Allreduce(&nexchange,&exchangecount,1,MPI_INT,MPI_SUM,world);
  exchangecounttotal += exchangecount;

  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;
  atom->nbonds += createcount;

  // trigger reneighboring if any bonds were exchaged
  // this insures neigh lists will immediately reflect the topology changes
  // done if no bonds exchaged

  if (createcount) next_reneighbor = update->ntimestep;
  if (!createcount) return;

  if (exchangecount) next_reneighbor = update->ntimestep;
  if (!exchangecount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect exchaged bonds

  commflag = 2;
  comm->forward_comm_fix(this);

  // exchange list of exchaged bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner is now set for owned and ghost atoms so loop over nall
  // OK if duplicates in exchaged list due to ghosts duplicating owned atoms
  // check J < 0 to insure a exchaged bond to unknown atom is included
  //   i.e. bond partner outside of cutoff length

  ncreate = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (ncreate == maxcreate) {
        maxcreate += DELTA;
        memory->grow(created,maxcreate,2,"bond/create:created");
      }
      created[ncreate][0] = tag[i];
      created[ncreate][1] = finalpartner[i];
      ncreate++;
    }
  }

  // update special neigh lists of all atoms affected by any exchaged bond
  // also remove angles/dihedrals/impropers exchaged by exchaged bonds

  //update_topology();

  // DEBUG
  // print_bb();
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly deleted
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is deleted
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

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
    error->all(FLERR,"Fix bond/exchange needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and created bonds
   influenced = 1 if atom's topology is affected by any exchaged bond
     yes if is one of 2 atoms in bond
     yes if both atom IDs appear in atom's special list
     else no
   if influenced:
     check for angles/dihedrals/impropers to exchange due to specific exchaged bonds
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
------------------------------------------------------------------------- */

void FixBondExchange::update_topology()
{
  int i,j,k,n,influence,influenced;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  nangles = 0;
  ndihedrals = 0;
  nimpropers = 0;
  overflow = 0;

  //printf("NCREATE %d: ",ncreate);
  //for (i = 0; i < ncreate; i++)
  //  printf(" %d %d,",created[i][0],created[i][1]);
  //printf("\n");

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < ncreate; j++) {
      id1 = created[j][0];
      id2 = created[j][1];

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
    }

    // rebuild_special_one() first, since used by create_angles, etc

    if (influenced) {
      rebuild_special_one(i);
      if (angleflag) exchange_angles(i);
      if (dihedralflag) exchange_dihedrals(i);
      if (improperflag) exchange_impropers(i);
    }
  }

  int overflowall;
  MPI_Allreduce(&overflow,&overflowall,1,MPI_INT,MPI_SUM,world);
  if (overflowall) error->all(FLERR,"Fix bond/exchange induced too many "
                              "angles/dihedrals/impropers per atom");

  int newton_bond = force->newton_bond;

  int all;
  if (angleflag) {
    MPI_Allreduce(&nangles,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 3;
    atom->nangles += all;
  }
  if (dihedralflag) {
    MPI_Allreduce(&ndihedrals,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->ndihedrals += all;
  }
  if (improperflag) {
    MPI_Allreduce(&nimpropers,&all,1,MPI_INT,MPI_SUM,world);
    if (!newton_bond) all /= 4;
    atom->nimpropers += all;
  }
}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs
------------------------------------------------------------------------- */

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
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/exchange needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/exchange");

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/exchange needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/exchange");

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   exchange any angles owned by atom M induced by newly created bonds
   walk special list to find all possible angles to exchange
   only add an angle if a new bond is one of its 2 bonds (I-J,J-K)
   for newton_bond on, atom M is central atom
   for newton_bond off, atom M is any of 3 atoms in angle
------------------------------------------------------------------------- */

void FixBondExchange::exchange_angles(int m)
{
  
}

/* ----------------------------------------------------------------------
   exchange any dihedrals owned by atom M induced by newly created bonds
   walk special list to find all possible dihedrals to exchange
   only add a dihedral if a new bond is one of its 3 bonds (I-J,J-K,K-L)
   for newton_bond on, atom M is central atom
   for newton_bond off, atom M is any of 4 atoms in dihedral
------------------------------------------------------------------------- */

void FixBondExchange::exchange_dihedrals(int m)
{
  
}

/* ----------------------------------------------------------------------
   exchange any impropers owned by atom M induced by newly created bonds
   walk special list to find all possible impropers to exchange
   only add an improper if a new bond is one of its 3 bonds (I-J,I-K,I-L)
   for newton_bond on, atom M is central atom
   for newton_bond off, atom M is any of 4 atoms in improper
------------------------------------------------------------------------- */

void FixBondExchange::exchange_impropers(int m)
{
  
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

/* ---------------------------------------------------------------------- */

void FixBondExchange::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
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
      buf[m++] = probability[j];
    }
    return m;
  }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
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
      probability[i] = buf[m++];
    }

  } else {

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      finalpartner[i] = (tagint) ubuf(buf[m++]).i;
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
    if (buf[m+1] > distsq[j]) {
      partner[j] = (tagint) ubuf(buf[m++]).i;
      distsq[j] = buf[m++];
    } else m += 2;
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

/* ---------------------------------------------------------------------- */

double FixBondExchange::compute_vector(int n)
{
  if (n == 0) return (double) createcount;
  return (double) exchangecount;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondExchange::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(tagint);
  bytes += (double)nmax * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::print_bb()
{
  for (int i = 0; i < atom->nlocal; i++) {
    printf("TAG " TAGINT_FORMAT ": %d nbonds: ",atom->tag[i],atom->num_bond[i]);
    for (int j = 0; j < atom->num_bond[i]; j++) {
      printf(" " TAGINT_FORMAT,atom->bond_atom[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d nangles: ",atom->tag[i],atom->num_angle[i]);
    for (int j = 0; j < atom->num_angle[i]; j++) {
      printf(" " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT ",",
             atom->angle_atom1[i][j], atom->angle_atom2[i][j],
             atom->angle_atom3[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d ndihedrals: ",atom->tag[i],atom->num_dihedral[i]);
    for (int j = 0; j < atom->num_dihedral[i]; j++) {
      printf(" " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " "
             TAGINT_FORMAT ",", atom->dihedral_atom1[i][j],
             atom->dihedral_atom2[i][j],atom->dihedral_atom3[i][j],
             atom->dihedral_atom4[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d nimpropers: ",atom->tag[i],atom->num_improper[i]);
    for (int j = 0; j < atom->num_improper[i]; j++) {
      printf(" " TAGINT_FORMAT " " TAGINT_FORMAT " " TAGINT_FORMAT " "
             TAGINT_FORMAT ",",atom->improper_atom1[i][j],
             atom->improper_atom2[i][j],atom->improper_atom3[i][j],
             atom->improper_atom4[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d %d %d nspecial: ",atom->tag[i],
           atom->nspecial[i][0],atom->nspecial[i][1],atom->nspecial[i][2]);
    for (int j = 0; j < atom->nspecial[i][2]; j++) {
      printf(" " TAGINT_FORMAT,atom->special[i][j]);
    }
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixBondExchange::print_copy(const char *str, tagint m,
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}
