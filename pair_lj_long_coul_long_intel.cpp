/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Rodrigo Canales (RWTH Aachen University)
------------------------------------------------------------------------- */

#include "pair_lj_long_coul_long_intel.h"
#include "suffix.h"
using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

PairLJLongCoulLongIntel::PairLJLongCoulLongIntel(LAMMPS *lmp) :
  PairLJLongCoulLong(lmp)
{
  suffix_flag |= Suffix::INTEL;
  respa_enable = 0;
  cut_respa = NULL;
}

/* ---------------------------------------------------------------------- */

PairLJLongCoulLongIntel::~PairLJLongCoulLongIntel()
{
}
