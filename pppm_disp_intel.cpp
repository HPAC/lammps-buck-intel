/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: William McDoniel (RWTH Aachen University)
------------------------------------------------------------------------- */


#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include "pppm_disp_intel.h"
#include "atom.h"
#include "fft3d_wrap.h"
#include "error.h"
#include "gridcomm.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "suffix.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define MAXORDER   7
#define OFFSET 16384
#define SMALL 0.00001
#define LARGE 10000.0
#define EPS_HOC 1.0e-7

enum{GEOMETRIC,ARITHMETIC,SIXTHPOWER};
enum{REVERSE_RHO, REVERSE_RHO_G, REVERSE_RHO_A, REVERSE_RHO_NONE};
enum{FORWARD_IK, FORWARD_AD, FORWARD_IK_PERATOM, FORWARD_AD_PERATOM,
     FORWARD_IK_G, FORWARD_AD_G, FORWARD_IK_PERATOM_G, FORWARD_AD_PERATOM_G,
     FORWARD_IK_A, FORWARD_AD_A, FORWARD_IK_PERATOM_A, FORWARD_AD_PERATOM_A,
     FORWARD_IK_NONE, FORWARD_AD_NONE, FORWARD_IK_PERATOM_NONE, FORWARD_AD_PERATOM_NONE};


#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

/* ---------------------------------------------------------------------- */

PPPMDispIntel::PPPMDispIntel(LAMMPS *lmp, int narg, char **arg) : PPPMDisp(lmp, narg, arg)
{
  suffix_flag |= Suffix::INTEL;
}

PPPMDispIntel::~PPPMDispIntel()
{
}


/* ----------------------------------------------------------------------
   called once before run
------------------------------------------------------------------------- */

void PPPMDispIntel::init()
{
  PPPMDisp::init();

  int ifix = modify->find_fix("package_intel");
  if (ifix < 0)
    error->all(FLERR,
               "The 'package intel' command is required for /intel styles");
  fix = static_cast<FixIntel *>(modify->fix[ifix]);

  #ifdef _LMP_INTEL_OFFLOAD
  _use_base = 0;
  if (fix->offload_balance() != 0.0) {
    _use_base = 1;
    return;
  }
  #endif

  fix->kspace_init_check();

  if (order > INTEL_P3M_MAXORDER)
    error->all(FLERR,"PPPM order greater than supported by USER-INTEL\n");

  /*
  if (fix->precision() == FixIntel::PREC_MODE_MIXED)
    pack_force_const(force_const_single, fix->get_mixed_buffers());
  else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE)
    pack_force_const(force_const_double, fix->get_double_buffers());
  else
    pack_force_const(force_const_single, fix->get_single_buffers());
  */
}

/* ----------------------------------------------------------------------
   compute the PPPM long-range force, energy, virial
------------------------------------------------------------------------- */

void PPPMDispIntel::compute(int eflag, int vflag)
{

  #ifdef _LMP_INTEL_OFFLOAD
  if (_use_base) {
    PPPM::compute(eflag, vflag);
    return;
  }
  #endif

  double p3mtime, p3mtime_compute, p3mtime_particlemap, p3mtime_makerho, p3mtime_poisson, p3mtime_fieldforce, p3mtime_brick2fft, p3mtime_total;
  struct timespec tv;
  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

  static double p3mtime_wholetimestep = p3mtime;
  printf("Timestep duration: %g\n\n", p3mtime - p3mtime_wholetimestep);
  p3mtime_wholetimestep = p3mtime;
  p3mtime_total = p3mtime;



  int i;
  // convert atoms from box to lamda coords

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = evflag_atom = eflag_global = vflag_global =
	 eflag_atom = vflag_atom = 0;

  if (evflag_atom && !peratom_allocate_flag) {
    allocate_peratom();
    if (function[0]) {
      cg_peratom->ghost_notify();
      cg_peratom->setup();
    }
    if (function[1] + function[2] + function[3]) {
      cg_peratom_6->ghost_notify();
      cg_peratom_6->setup();
    }
    peratom_allocate_flag = 1;
  }

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  // extend size of per-atom arrays if necessary

  if (atom->nlocal > nmax) {

    if (function[0]) memory->destroy(part2grid);
    if (function[1] + function[2] + function[3]) memory->destroy(part2grid_6);
    nmax = atom->nmax;
    if (function[0]) memory->create(part2grid,nmax,3,"pppm/disp:part2grid");
    if (function[1] + function[2] + function[3])
      memory->create(part2grid_6,nmax,3,"pppm/disp:part2grid_6");
  }


  energy = 0.0;
  energy_1 = 0.0;
  energy_6 = 0.0;
  if (vflag) for (i = 0; i < 6; i++) virial_6[i] = virial_1[i] = 0.0;

  // find grid points for all my particles
  // distribute partcles' charges/dispersion coefficients on the grid
  // communication between processors and remapping two fft
  // Solution of poissons equation in k-space and backtransformation
  // communication between processors
  // calculation of forces

  if (function[0]) {
    //perfrom calculations for coulomb interactions only

    particle_map_c(delxinv, delyinv, delzinv, shift, part2grid, nupper, nlower,
                 nxlo_out, nylo_out, nzlo_out, nxhi_out, nyhi_out, nzhi_out);

    make_rho_c();

    cg->reverse_comm(this,REVERSE_RHO);

    brick2fft(nxlo_in, nylo_in, nzlo_in, nxhi_in, nyhi_in, nzhi_in,
	      density_brick, density_fft, work1,remap);

    if (differentiation_flag == 1) {

      poisson_ad(work1, work2, density_fft, fft1, fft2,
                 nx_pppm, ny_pppm, nz_pppm, nfft,
                 nxlo_fft, nylo_fft, nzlo_fft, nxhi_fft, nyhi_fft, nzhi_fft,
                 nxlo_in, nylo_in, nzlo_in, nxhi_in, nyhi_in, nzhi_in,
                 energy_1, greensfn,
                 virial_1, vg,vg2,
                 u_brick, v0_brick, v1_brick, v2_brick, v3_brick, v4_brick, v5_brick);

      cg->forward_comm(this,FORWARD_AD);

      fieldforce_c_ad();

      if (vflag_atom) cg_peratom->forward_comm(this, FORWARD_AD_PERATOM);

    } else {
      poisson_ik(work1, work2, density_fft, fft1, fft2,
                 nx_pppm, ny_pppm, nz_pppm, nfft,
                 nxlo_fft, nylo_fft, nzlo_fft, nxhi_fft, nyhi_fft, nzhi_fft,
                 nxlo_in, nylo_in, nzlo_in, nxhi_in, nyhi_in, nzhi_in,
                 energy_1, greensfn,
	         fkx, fky, fkz,fkx2, fky2, fkz2,
                 vdx_brick, vdy_brick, vdz_brick, virial_1, vg,vg2,
                 u_brick, v0_brick, v1_brick, v2_brick, v3_brick, v4_brick, v5_brick);

      cg->forward_comm(this, FORWARD_IK);

      fieldforce_c_ik();

      if (evflag_atom) cg_peratom->forward_comm(this, FORWARD_IK_PERATOM);
    }
    if (evflag_atom) fieldforce_c_peratom();
  }

  if (function[1]) {
    //perfrom calculations for geometric mixing

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_particlemap = 0;
  else p3mtime_particlemap = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

    particle_map(delxinv_6, delyinv_6, delzinv_6, shift_6, part2grid_6, nupper_6, nlower_6,
                 nxlo_out_6, nylo_out_6, nzlo_out_6, nxhi_out_6, nyhi_out_6, nzhi_out_6);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
  printf("particle map time: %g\n", p3mtime - p3mtime_particlemap);
  p3mtime_makerho = p3mtime;


    make_rho_g();

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
  printf("make rho time: %g\n", p3mtime - p3mtime_makerho);

    cg_6->reverse_comm(this, REVERSE_RHO_G);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_brick2fft = 0;
  else p3mtime_brick2fft = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

    brick2fft(nxlo_in_6, nylo_in_6, nzlo_in_6, nxhi_in_6, nyhi_in_6, nzhi_in_6,
	      density_brick_g, density_fft_g, work1_6,remap_6);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
  printf("brick2fft time: %g\n", p3mtime - p3mtime_brick2fft);

    if (differentiation_flag == 1) {

      poisson_ad(work1_6, work2_6, density_fft_g, fft1_6, fft2_6,
                 nx_pppm_6, ny_pppm_6, nz_pppm_6, nfft_6,
                 nxlo_fft_6, nylo_fft_6, nzlo_fft_6, nxhi_fft_6, nyhi_fft_6, nzhi_fft_6,
                 nxlo_in_6, nylo_in_6, nzlo_in_6, nxhi_in_6, nyhi_in_6, nzhi_in_6,
                 energy_6, greensfn_6,
                 virial_6, vg_6, vg2_6,
                 u_brick_g, v0_brick_g, v1_brick_g, v2_brick_g, v3_brick_g, v4_brick_g, v5_brick_g);

      cg_6->forward_comm(this,FORWARD_AD_G);

      fieldforce_g_ad();

      if (vflag_atom) cg_peratom_6->forward_comm(this,FORWARD_AD_PERATOM_G);

    } else {

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_poisson = 0;
  else p3mtime_poisson = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

      poisson_ik(work1_6, work2_6, density_fft_g, fft1_6, fft2_6,
                 nx_pppm_6, ny_pppm_6, nz_pppm_6, nfft_6,
                 nxlo_fft_6, nylo_fft_6, nzlo_fft_6, nxhi_fft_6, nyhi_fft_6, nzhi_fft_6,
                 nxlo_in_6, nylo_in_6, nzlo_in_6, nxhi_in_6, nyhi_in_6, nzhi_in_6,
                 energy_6, greensfn_6,
	         fkx_6, fky_6, fkz_6,fkx2_6, fky2_6, fkz2_6,
                 vdx_brick_g, vdy_brick_g, vdz_brick_g, virial_6, vg_6, vg2_6,
                 u_brick_g, v0_brick_g, v1_brick_g, v2_brick_g, v3_brick_g, v4_brick_g, v5_brick_g);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
  printf("poisson time: %g\n", p3mtime - p3mtime_poisson);

      cg_6->forward_comm(this,FORWARD_IK_G);


  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_fieldforce = 0;
  else p3mtime_fieldforce = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

      fieldforce_g_ik();

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
  printf("fieldforce time: %g\n", p3mtime - p3mtime_fieldforce);


      if (evflag_atom) cg_peratom_6->forward_comm(this, FORWARD_IK_PERATOM_G);
    }
    if (evflag_atom) fieldforce_g_peratom();
  }

  if (function[2]) {
    //perform calculations for arithmetic mixing


    particle_map(delxinv_6, delyinv_6, delzinv_6, shift_6, part2grid_6, nupper_6, nlower_6,
                 nxlo_out_6, nylo_out_6, nzlo_out_6, nxhi_out_6, nyhi_out_6, nzhi_out_6);
    make_rho_a();

    cg_6->reverse_comm(this, REVERSE_RHO_A);

    brick2fft_a();

    if ( differentiation_flag == 1) {

      poisson_ad(work1_6, work2_6, density_fft_a3, fft1_6, fft2_6,
                 nx_pppm_6, ny_pppm_6, nz_pppm_6, nfft_6,
                 nxlo_fft_6, nylo_fft_6, nzlo_fft_6, nxhi_fft_6, nyhi_fft_6, nzhi_fft_6,
                 nxlo_in_6, nylo_in_6, nzlo_in_6, nxhi_in_6, nyhi_in_6, nzhi_in_6,
                 energy_6, greensfn_6,
                 virial_6, vg_6, vg2_6,
                 u_brick_a3, v0_brick_a3, v1_brick_a3, v2_brick_a3, v3_brick_a3, v4_brick_a3, v5_brick_a3);
      poisson_2s_ad(density_fft_a0, density_fft_a6,
                    u_brick_a0, v0_brick_a0, v1_brick_a0, v2_brick_a0, v3_brick_a0, v4_brick_a0, v5_brick_a0,
                    u_brick_a6, v0_brick_a6, v1_brick_a6, v2_brick_a6, v3_brick_a6, v4_brick_a6, v5_brick_a6);
      poisson_2s_ad(density_fft_a1, density_fft_a5,
                    u_brick_a1, v0_brick_a1, v1_brick_a1, v2_brick_a1, v3_brick_a1, v4_brick_a1, v5_brick_a1,
                    u_brick_a5, v0_brick_a5, v1_brick_a5, v2_brick_a5, v3_brick_a5, v4_brick_a5, v5_brick_a5);
      poisson_2s_ad(density_fft_a2, density_fft_a4,
                    u_brick_a2, v0_brick_a2, v1_brick_a2, v2_brick_a2, v3_brick_a2, v4_brick_a2, v5_brick_a2,
                    u_brick_a4, v0_brick_a4, v1_brick_a4, v2_brick_a4, v3_brick_a4, v4_brick_a4, v5_brick_a4);

      cg_6->forward_comm(this, FORWARD_AD_A);

      fieldforce_a_ad();

      if (evflag_atom) cg_peratom_6->forward_comm(this, FORWARD_AD_PERATOM_A);

    }  else {

      poisson_ik(work1_6, work2_6, density_fft_a3, fft1_6, fft2_6,
                 nx_pppm_6, ny_pppm_6, nz_pppm_6, nfft_6,
                 nxlo_fft_6, nylo_fft_6, nzlo_fft_6, nxhi_fft_6, nyhi_fft_6, nzhi_fft_6,
                 nxlo_in_6, nylo_in_6, nzlo_in_6, nxhi_in_6, nyhi_in_6, nzhi_in_6,
                 energy_6, greensfn_6,
	         fkx_6, fky_6, fkz_6,fkx2_6, fky2_6, fkz2_6,
                 vdx_brick_a3, vdy_brick_a3, vdz_brick_a3, virial_6, vg_6, vg2_6,
                 u_brick_a3, v0_brick_a3, v1_brick_a3, v2_brick_a3, v3_brick_a3, v4_brick_a3, v5_brick_a3);
      poisson_2s_ik(density_fft_a0, density_fft_a6,
                    vdx_brick_a0, vdy_brick_a0, vdz_brick_a0,
                    vdx_brick_a6, vdy_brick_a6, vdz_brick_a6,
                    u_brick_a0, v0_brick_a0, v1_brick_a0, v2_brick_a0, v3_brick_a0, v4_brick_a0, v5_brick_a0,
                    u_brick_a6, v0_brick_a6, v1_brick_a6, v2_brick_a6, v3_brick_a6, v4_brick_a6, v5_brick_a6);
      poisson_2s_ik(density_fft_a1, density_fft_a5,
                    vdx_brick_a1, vdy_brick_a1, vdz_brick_a1,
                    vdx_brick_a5, vdy_brick_a5, vdz_brick_a5,
                    u_brick_a1, v0_brick_a1, v1_brick_a1, v2_brick_a1, v3_brick_a1, v4_brick_a1, v5_brick_a1,
                    u_brick_a5, v0_brick_a5, v1_brick_a5, v2_brick_a5, v3_brick_a5, v4_brick_a5, v5_brick_a5);
      poisson_2s_ik(density_fft_a2, density_fft_a4,
                    vdx_brick_a2, vdy_brick_a2, vdz_brick_a2,
                    vdx_brick_a4, vdy_brick_a4, vdz_brick_a4,
                    u_brick_a2, v0_brick_a2, v1_brick_a2, v2_brick_a2, v3_brick_a2, v4_brick_a2, v5_brick_a2,
                    u_brick_a4, v0_brick_a4, v1_brick_a4, v2_brick_a4, v3_brick_a4, v4_brick_a4, v5_brick_a4);

      cg_6->forward_comm(this, FORWARD_IK_A);

      fieldforce_a_ik();

      if (evflag_atom) cg_peratom_6->forward_comm(this, FORWARD_IK_PERATOM_A);
    }
    if (evflag_atom) fieldforce_a_peratom();
  }

  if (function[3]) {
    //perfrom calculations if no mixing rule applies
    particle_map(delxinv_6, delyinv_6, delzinv_6, shift_6, part2grid_6, nupper_6, nlower_6,
                 nxlo_out_6, nylo_out_6, nzlo_out_6, nxhi_out_6, nyhi_out_6, nzhi_out_6);

    make_rho_none();

    cg_6->reverse_comm(this, REVERSE_RHO_NONE);

    brick2fft_none();

    if (differentiation_flag == 1) {

      int n = 0;
      for (int k = 0; k<nsplit_alloc/2; k++) {
        poisson_none_ad(n,n+1,density_fft_none[n],density_fft_none[n+1],
                        u_brick_none[n],u_brick_none[n+1],
                        v0_brick_none, v1_brick_none, v2_brick_none,
                        v3_brick_none, v4_brick_none, v5_brick_none);
        n += 2;
      }

      cg_6->forward_comm(this,FORWARD_AD_NONE);

      fieldforce_none_ad();

      if (vflag_atom) cg_peratom_6->forward_comm(this,FORWARD_AD_PERATOM_NONE);

    } else {
      int n = 0;
      for (int k = 0; k<nsplit_alloc/2; k++) {

        poisson_none_ik(n,n+1,density_fft_none[n], density_fft_none[n+1],
                        vdx_brick_none[n], vdy_brick_none[n], vdz_brick_none[n],
                        vdx_brick_none[n+1], vdy_brick_none[n+1], vdz_brick_none[n+1],
                        u_brick_none, v0_brick_none, v1_brick_none, v2_brick_none,
                        v3_brick_none, v4_brick_none, v5_brick_none);
        n += 2;
      }

      cg_6->forward_comm(this,FORWARD_IK_NONE);

      fieldforce_none_ik();

      if (evflag_atom)
        cg_peratom_6->forward_comm(this, FORWARD_IK_PERATOM_NONE);
    }
    if (evflag_atom) fieldforce_none_peratom();
  }

  // update qsum and qsqsum, if atom count has changed and energy needed

  if ((eflag_global || eflag_atom) && atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // sum energy across procs and add in volume-dependent term

  const double qscale = force->qqrd2e * scale;
  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy_1,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    energy_1 = energy_all;
    MPI_Allreduce(&energy_6,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    energy_6 = energy_all;

    energy_1 *= 0.5*volume;
    energy_6 *= 0.5*volume;

    energy_1 -= g_ewald*qsqsum/MY_PIS +
      MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy_6 += - MY_PI*MY_PIS/(6*volume)*pow(g_ewald_6,3)*csumij +
      1.0/12.0*pow(g_ewald_6,6)*csum;
    energy_1 *= qscale;
  }

  // sum virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial_1,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < 6; i++) virial[i] = 0.5*qscale*volume*virial_all[i];
    MPI_Allreduce(virial_6,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < 6; i++) virial[i] += 0.5*volume*virial_all[i];
    if (function[1]+function[2]+function[3]){
      double a =  MY_PI*MY_PIS/(6*volume)*pow(g_ewald_6,3)*csumij;
      virial[0] -= a;
      virial[1] -= a;
      virial[2] -= a;
    }
  }

  if (eflag_atom) {
    if (function[0]) {
      double *q = atom->q;
      for (i = 0; i < atom->nlocal; i++) {
        eatom[i] -= qscale*g_ewald*q[i]*q[i]/MY_PIS + qscale*MY_PI2*q[i]*qsum / (g_ewald*g_ewald*volume); //coulomb self energy correction
      }
    }
    if (function[1] + function[2] + function[3]) {
      int tmp;
      for (i = 0; i < atom->nlocal; i++) {
        tmp = atom->type[i];
        eatom[i] += - MY_PI*MY_PIS/(6*volume)*pow(g_ewald_6,3)*csumi[tmp] +
                      1.0/12.0*pow(g_ewald_6,6)*cii[tmp];
      }
    }
  }

  if (vflag_atom) {
    if (function[1] + function[2] + function[3]) {
      int tmp;
      for (i = 0; i < atom->nlocal; i++) {
        tmp = atom->type[i];
        for (int n = 0; n < 3; n++) vatom[i][n] -= MY_PI*MY_PIS/(6*volume)*pow(g_ewald_6,3)*csumi[tmp]; //dispersion self virial correction
      }
    }
  }


  // 2d slab correction

  if (slabflag) slabcorr(eflag);
  if (function[0]) energy += energy_1;
  if (function[1] + function[2] + function[3]) energy += energy_6;

  // convert atoms back from lamda to box coords

  if (triclinic) domain->lamda2x(atom->nlocal);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
  printf("total p3mtime: %g\n", p3mtime - p3mtime_total);

}


/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik differentiation
------------------------------------------------------------------------- */

void PPPMDispIntel::poisson_ik(FFT_SCALAR* wk1, FFT_SCALAR* wk2,
                           FFT_SCALAR* dfft, LAMMPS_NS::FFT3d* ft1,LAMMPS_NS::FFT3d* ft2,
                           int nx_p, int ny_p, int nz_p, int nft,
                           int nxlo_ft, int nylo_ft, int nzlo_ft,
                           int nxhi_ft, int nyhi_ft, int nzhi_ft,
                           int nxlo_i, int nylo_i, int nzlo_i,
                           int nxhi_i, int nyhi_i, int nzhi_i,
                           double& egy, double* gfn,
                           double* kx, double* ky, double* kz,
                           double* kx2, double* ky2, double* kz2,
                           FFT_SCALAR*** vx_brick, FFT_SCALAR*** vy_brick, FFT_SCALAR*** vz_brick,
                           double* vir, double** vcoeff, double** vcoeff2,
                           FFT_SCALAR*** u_pa, FFT_SCALAR*** v0_pa, FFT_SCALAR*** v1_pa, FFT_SCALAR*** v2_pa,
                           FFT_SCALAR*** v3_pa, FFT_SCALAR*** v4_pa, FFT_SCALAR*** v5_pa)


{

  double p3mtime, p3mtime_fft = 0.;
  struct timespec tv;

  int i,j,k,n;
  double eng;

  // transform charge/dispersion density (r -> k)
  n = 0;
  for (i = 0; i < nft; i++) {
    wk1[n++] = dfft[i];
    wk1[n++] = ZEROF;
  }


  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

  ft1->compute(wk1,wk1,1);

 if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_fft += 0;
  else p3mtime_fft += ((tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.)) - p3mtime;


  // if requested, compute energy and virial contribution

  double scaleinv = 1.0/(nx_p*ny_p*nz_p);
  double s2 = scaleinv*scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nft; i++) {
	eng = s2 * gfn[i] * (wk1[n]*wk1[n] + wk1[n+1]*wk1[n+1]);
	for (j = 0; j < 6; j++) vir[j] += eng*vcoeff[i][j];
	if (eflag_global) egy += eng;
	n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nft; i++) {
	egy +=
	  s2 * gfn[i] * (wk1[n]*wk1[n] + wk1[n+1]*wk1[n+1]);
	n += 2;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nft; i++) {
    wk1[n++] *= scaleinv * gfn[i];
    wk1[n++] *= scaleinv * gfn[i];
  }

  // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x & y direction gradient

  n = 0;
  for (k = nzlo_ft; k <= nzhi_ft; k++)
    for (j = nylo_ft; j <= nyhi_ft; j++)
      for (i = nxlo_ft; i <= nxhi_ft; i++) {
	wk2[n] = 0.5*(kx[i]-kx2[i])*wk1[n+1] + 0.5*(ky[j]-ky2[j])*wk1[n];
	wk2[n+1] = -0.5*(kx[i]-kx2[i])*wk1[n] + 0.5*(ky[j]-ky2[j])*wk1[n+1];
	n += 2;
      }

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

  ft2->compute(wk2,wk2,-1);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_fft += 0;
  else p3mtime_fft += ((tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.)) - p3mtime;

  n = 0;
  for (k = nzlo_i; k <= nzhi_i; k++)
    for (j = nylo_i; j <= nyhi_i; j++)
      for (i = nxlo_i; i <= nxhi_i; i++) {
	vx_brick[k][j][i] = wk2[n++];
	vy_brick[k][j][i] = wk2[n++];
      }

  if (!eflag_atom) {
    // z direction gradient only

    n = 0;
    for (k = nzlo_ft; k <= nzhi_ft; k++)
      for (j = nylo_ft; j <= nyhi_ft; j++)
        for (i = nxlo_ft; i <= nxhi_ft; i++) {
	  wk2[n] = kz[k]*wk1[n+1];
	  wk2[n+1] = -kz[k]*wk1[n];
	  n += 2;
        }

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

    ft2->compute(wk2,wk2,-1);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_fft += 0;
  else p3mtime_fft += ((tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.)) - p3mtime;

    n = 0;
    for (k = nzlo_i; k <= nzhi_i; k++)
      for (j = nylo_i; j <= nyhi_i; j++)
        for (i = nxlo_i; i <= nxhi_i; i++) {
	  vz_brick[k][j][i] = wk2[n];
	  n += 2;
        }

  }

  else {
    // z direction gradient & per-atom energy

    n = 0;
    for (k = nzlo_ft; k <= nzhi_ft; k++)
      for (j = nylo_ft; j <= nyhi_ft; j++)
        for (i = nxlo_ft; i <= nxhi_ft; i++) {
	  wk2[n] = 0.5*(kz[k]-kz2[k])*wk1[n+1] - wk1[n+1];
	  wk2[n+1] = -0.5*(kz[k]-kz2[k])*wk1[n] + wk1[n];
	  n += 2;
        }


  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

    ft2->compute(wk2,wk2,-1);

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_fft += 0;
  else p3mtime_fft += ((tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.)) - p3mtime;



    n = 0;
    for (k = nzlo_i; k <= nzhi_i; k++)
      for (j = nylo_i; j <= nyhi_i; j++)
        for (i = nxlo_i; i <= nxhi_i; i++) {
	  vz_brick[k][j][i] = wk2[n++];
	  u_pa[k][j][i] = wk2[n++];;
        }
  }

  if (vflag_atom) poisson_peratom(wk1, wk2, ft2, vcoeff, vcoeff2, nft,
                                  nxlo_i, nylo_i, nzlo_i, nxhi_i, nyhi_i, nzhi_i,
                                  v0_pa, v1_pa, v2_pa, v3_pa, v4_pa, v5_pa);


  printf("fft time %g\n", p3mtime_fft);
}
