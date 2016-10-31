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
                        Rodrigo Canales (RWTH Aachen University)
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

  #ifdef HPAC_TIMING
  double p3mtime, p3mtime_compute, p3mtime_particlemap, p3mtime_makerho, p3mtime_poisson, p3mtime_fieldforce, p3mtime_brick2fft, p3mtime_total;
  struct timespec tv;
  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);

  static double p3mtime_wholetimestep = p3mtime;
  printf("Timestep duration: %g\n\n", p3mtime - p3mtime_wholetimestep);
  p3mtime_wholetimestep = p3mtime;
  p3mtime_total = p3mtime;
  #endif


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

    // particle_map_c(delxinv, delyinv, delzinv, shift, part2grid, nupper, nlower,
    //              nxlo_out, nylo_out, nzlo_out, nxhi_out, nyhi_out, nzhi_out);
    particle_map<'c', double, double>(fix->get_double_buffers());
    //make_rho_c();

    make_rho<'c', double, double>(fix->get_double_buffers());

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

      //fieldforce_c_ik();
      fieldforce_ik<'c',double, double>(fix->get_double_buffers());

      if (evflag_atom) cg_peratom->forward_comm(this, FORWARD_IK_PERATOM);
    }
    if (evflag_atom) fieldforce_c_peratom();
  }

  if (function[1]) {
    //perfrom calculations for geometric mixing

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_particlemap = 0;
    else p3mtime_particlemap = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    #endif

    // particle_map(delxinv_6, delyinv_6, delzinv_6, shift_6, part2grid_6,
    // 		 nupper_6, nlower_6, nxlo_out_6, nylo_out_6, nzlo_out_6,
    // 		 nxhi_out_6, nyhi_out_6, nzhi_out_6);

    particle_map<'g', double, double>(fix->get_double_buffers());

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
    else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    printf("particle map time: %g\n", p3mtime - p3mtime_particlemap);
    p3mtime_makerho = p3mtime;
    #endif

    //make_rho_g();
    make_rho<'g', double, double>(fix->get_double_buffers());

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
    else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    printf("make rho time: %g\n", p3mtime - p3mtime_makerho);
    #endif

    cg_6->reverse_comm(this, REVERSE_RHO_G);

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_brick2fft = 0;
    else p3mtime_brick2fft = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    #endif

    brick2fft(nxlo_in_6, nylo_in_6, nzlo_in_6, nxhi_in_6, nyhi_in_6, nzhi_in_6,
	      density_brick_g, density_fft_g, work1_6,remap_6);

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
    else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    printf("brick2fft time: %g\n", p3mtime - p3mtime_brick2fft);
    #endif

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

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_poisson = 0;
    else p3mtime_poisson = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    #endif


    poisson_ik(work1_6, work2_6, density_fft_g, fft1_6, fft2_6,
               nx_pppm_6, ny_pppm_6, nz_pppm_6, nfft_6,
               nxlo_fft_6, nylo_fft_6, nzlo_fft_6, nxhi_fft_6, nyhi_fft_6, nzhi_fft_6,
               nxlo_in_6, nylo_in_6, nzlo_in_6, nxhi_in_6, nyhi_in_6, nzhi_in_6,
               energy_6, greensfn_6,
               fkx_6, fky_6, fkz_6,fkx2_6, fky2_6, fkz2_6,
               vdx_brick_g, vdy_brick_g, vdz_brick_g, virial_6, vg_6, vg2_6,
               u_brick_g, v0_brick_g, v1_brick_g, v2_brick_g, v3_brick_g, v4_brick_g, v5_brick_g);

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
    else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    printf("poisson time: %g\n", p3mtime - p3mtime_poisson);
    #endif

    cg_6->forward_comm(this,FORWARD_IK_G);

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime_fieldforce = 0;
    else p3mtime_fieldforce = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    #endif

    //fieldforce_g_ik();
    fieldforce_ik<'g',double, double>(fix->get_double_buffers());

    #ifdef HPAC_TIMING
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
    else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
    printf("fieldforce time: %g\n", p3mtime - p3mtime_fieldforce);
    #endif


    if (evflag_atom) cg_peratom_6->forward_comm(this, FORWARD_IK_PERATOM_G);
    }
      if (evflag_atom) fieldforce_g_peratom();
    }

    if (function[2]) {
     //perform calculations for arithmetic mixing


    // particle_map(delxinv_6, delyinv_6, delzinv_6, shift_6, part2grid_6,
    //              nupper_6, nlower_6, nxlo_out_6, nylo_out_6, nzlo_out_6,
    //              nxhi_out_6, nyhi_out_6, nzhi_out_6);

    particle_map<'g', double, double>(fix->get_double_buffers());
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
    // perform calculations if no mixing rule applies
    // particle_map(delxinv_6, delyinv_6, delzinv_6, shift_6, part2grid_6,
    //              nupper_6, nlower_6, nxlo_out_6, nylo_out_6, nzlo_out_6,
    //              nxhi_out_6, nyhi_out_6, nzhi_out_6);

    particle_map<'g', double, double>(fix->get_double_buffers());
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

  #ifdef HPAC_TIMING
  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) p3mtime = 0;
  else p3mtime = (tv.tv_sec-1.46358e9) + ((double)tv.tv_nsec/1000000000.);
  printf("total p3mtime: %g\n", p3mtime - p3mtime_total);
  #endif
}

template<const char VARIANT, class flt_t, class acc_t>
void PPPMDispIntel::particle_map(IntelBuffers<flt_t,acc_t> *buffers)
{


  if (!ISFINITE(boxlo[0]) || !ISFINITE(boxlo[1]) || !ISFINITE(boxlo[2]))
    error->one(FLERR,"Non-numeric box dimensions - simulation unstable");

    // particle_map_c (delxinv, ... )
    // particle_map (delxinv_6, ...)
  #define SWITCHVAR(_c, _g ) VARIANT == 'c' ? _c : _g



  const flt_t xi = SWITCHVAR(delxinv, delxinv_6);
  const flt_t yi = SWITCHVAR(delyinv, delyinv_6);
  const flt_t zi = SWITCHVAR(delzinv, delzinv_6);
  const flt_t fshift = SWITCHVAR(shift, shift_6);
  const int nxlo = SWITCHVAR(nxlo_out, nxlo_out_6);
  const int nylo = SWITCHVAR(nylo_out, nylo_out_6);
  const int nzlo = SWITCHVAR(nzlo_out, nzlo_out_6);
  const int nxhi = SWITCHVAR(nxhi_out, nxhi_out_6);
  const int nyhi = SWITCHVAR(nyhi_out, nyhi_out_6);
  const int nzhi = SWITCHVAR(nzhi_out, nzhi_out_6);
  const int nup = SWITCHVAR(nupper, nupper_6);
  const int nlow = SWITCHVAR(nlower, nlower_6);
  int ** const p2g = SWITCHVAR(part2grid, part2grid_6);

  #undef SWITCHVAR

  const flt_t lo0 = boxlo[0];
  const flt_t lo1 = boxlo[1];
  const flt_t lo2 = boxlo[2];

  ATOM_T * _noalias const x = buffers->get_x(0);
  int nlocal = atom->nlocal;
  int nthr = comm->nthreads;
  int flag = 0;

#if defined(_OPENMP)
#pragma omp parallel default(none) \
  shared(nlocal, nthr) \
  reduction(+:flag)
#endif
  {
    int iifrom=0, iito=nlocal, tid=0;
    IP_PRE_omp_range_id(iifrom, iito, tid, nlocal, nthr);
  #if defined(LMP_SIMD_COMPILER)
  #pragma vector aligned
  #pragma simd reduction(+:flag)
  #endif
  for (int i = iifrom; i < iito; i++) {

    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    int nx = static_cast<int> ((x[i].x-lo0)*xi+fshift) - OFFSET;
    int ny = static_cast<int> ((x[i].y-lo1)*yi+fshift) - OFFSET;
    int nz = static_cast<int> ((x[i].z-lo2)*zi+fshift) - OFFSET;

    p2g[i][0] = nx;
    p2g[i][1] = ny;
    p2g[i][2] = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick
    if (nx+nlow < nxlo || nx+nup > nxhi ||
        ny+nlow < nylo || ny+nup > nyhi ||
        nz+nlow < nzlo || nz+nup > nzhi)
      flag = 1;
  }
  }
  if (flag) error->one(FLERR,"Out of range atoms - cannot compute PPPM");

}


template<const char VARIANT, class flt_t, class acc_t>
void PPPMDispIntel::make_rho(IntelBuffers<flt_t,acc_t> *buffers)
{
  // A macro to assign the values to the different variants
  #define SWITCHVAR(_c, _g ) VARIANT == 'c' ? _c : _g

  const int fngrid = SWITCHVAR(ngrid, ngrid_6);
  const flt_t xi = SWITCHVAR(delxinv, delxinv_6);
  const flt_t yi = SWITCHVAR(delyinv, delyinv_6);
  const flt_t zi = SWITCHVAR(delzinv, delzinv_6);
  const flt_t fshiftone = SWITCHVAR(shiftone, shiftone_6);
  const flt_t fdelvolinv = SWITCHVAR(delvolinv, delvolinv_6);
  const int nxlo = SWITCHVAR(nxlo_out, nxlo_out_6);
  const int nylo = SWITCHVAR(nylo_out, nylo_out_6);
  const int nzlo = SWITCHVAR(nzlo_out, nzlo_out_6);
  const int nxhi = SWITCHVAR(nxhi_out, nxhi_out_6);
  const int nyhi = SWITCHVAR(nyhi_out, nyhi_out_6);
  const int nzhi = SWITCHVAR(nzhi_out, nzhi_out_6);
  const int nup = SWITCHVAR(nupper, nupper_6);
  const int nlow = SWITCHVAR(nlower, nlower_6);
  FFT_SCALAR *** fdbrick = SWITCHVAR(density_brick, density_brick_g);
  FFT_SCALAR ** const frho_coeff = SWITCHVAR(rho_coeff, rho_coeff_6);
  int ** const p2g = SWITCHVAR(part2grid, part2grid_6);
  const int forder = SWITCHVAR(order, order_6);

  #undef SWITCHVAR

  const flt_t lo0 = boxlo[0];
  const flt_t lo1 = boxlo[1];
  const flt_t lo2 = boxlo[2];

  const int nthr = comm->nthreads;


  FFT_SCALAR * _noalias const densityThr =
    &(fdbrick[nzlo][nylo][nxlo]);

  // clear 3d density array
  memset(densityThr, 0.0, fngrid*sizeof(FFT_SCALAR));


  //icc 16.0 does not support OpenMP 4.5 and so doesn't support
  //array reduction.  This sets up private arrays in order to
  //do the reduction manually.

  FFT_SCALAR localDensity[nthr * fngrid * 2];
  memset(localDensity, 0.0, nthr * fngrid * 2 * sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  ATOM_T * _noalias const x = buffers->get_x(0);
  flt_t * _noalias const q = buffers->get_q(0);
  int nlocal = atom->nlocal;


  const int nix = nxhi - nxlo + 1;
  const int niy = nyhi - nylo + 1;
  const int nsize = nup - nlow;
  const int tripcount = nup = nlow +1;



  //Parallelize over the atoms
  #if defined(_OPENMP)
  #pragma omp parallel default(none) \
    shared(nthr, nlocal, localDensity)
  #endif
  {
    int jfrom, jto, tid;
    IP_PRE_omp_range_id(jfrom, jto, tid, nlocal, nthr);

    _declspec(align(64)) flt_t rho[3][8];
    rho[0][7] = 0.0;

    for (int i = jfrom; i < jto; ++i) {

      int nx = p2g[i][0];
      int ny = p2g[i][1];
      int nz = p2g[i][2];

      int nysum = nlow + ny - nylo;
      int nxsum = nlow + nx - nxlo + fngrid*tid;
      int nzsum = (nlow + nz - nzlo) * nix * niy + nysum*nix + nxsum;

      FFT_SCALAR dx = nx+fshiftone - (x[i].x-lo0)*xi;
      FFT_SCALAR dy = ny+fshiftone - (x[i].y-lo1)*yi;
      FFT_SCALAR dz = nz+fshiftone - (x[i].z-lo2)*zi;


      #if defined(LMP_SIMD_COMPILER)
      #pragma simd
      #endif
      for (int k = nlow; k <= nup; k++) {
        FFT_SCALAR r1,r2,r3;
        r1 = r2 = r3 = ZEROF;

        for (int l = forder-1; l >= 0; --l) {
          r1 = frho_coeff[l][k] + r1*dx;
          r2 = frho_coeff[l][k] + r2*dy;
          r3 = frho_coeff[l][k] + r3*dz;
        }
        rho[0][k-nlow] = r1;
        rho[1][k-nlow] = r2;
        rho[2][k-nlow] = r3;
      }

      FFT_SCALAR z0 = fdelvolinv * q[i];

      #pragma loop_count = 7
      for (int n = 0; n < tripcount; ++n) {
        int mz = n*nix*niy +nzsum;
        FFT_SCALAR y0 = z0*rho[2][n];

        #pragma loop_count = 7
        for (int m = 0; m < tripcount; ++m) {
          int mzy = mz + m*nix;
          FFT_SCALAR x0 = y0*rho[1][m];

          #pragma simd
          for (int l = 0; l < 8; ++l) {
            int mzyx = mzy + l;
            localDensity[mzyx] += x0*rho[0][l];
          }
        }
      }
    }
  }

  //do the reduction
  #if defined(_OPENMP)
  #pragma omp parallel default(none) \
    shared(nthr, nlocal, localDensity)
  #endif
  {
    int jfrom, jto, tid;
    IP_PRE_omp_range_id(jfrom, jto, tid, fngrid, nthr);

    #if defined(LMP_SIMD_COMPILER)
    //#pragma vector aligned nontemporal
      #pragma simd
      #endif
  for (int i = jfrom; i < jto; ++i) {
    for(int j = 0; j < nthr; ++j) {
      densityThr[i] += localDensity[i + j*fngrid];
    }
  }
  }

}

template<const char VARIANT, class flt_t, class acc_t>
void fieldforce_ik(IntelBuffers<flt_t,acc_t> *buffers) {

  ATOM_T * _noalias const x = buffers->get_x(0);
  flt_t * _noalias const q = buffers->get_q(0);
  FORCE_T * _noalias const f = buffers->get_f();


  int nlocal = atom->nlocal;
  int nthr = comm->nthreads;

  const flt_t lo0 = boxlo[0];
  const flt_t lo1 = boxlo[1];
  const flt_t lo2 = boxlo[2];

  #define SWITCHVAR(_c, _g ) VARIANT == 'c' ? _c : _g

  const flt_t xi = SWITCHVAR(delxinv, delxinv_6);
  const flt_t yi = SWITCHVAR(delyinv, delyxinv_6);
  const flt_t zi = SWITCHVAR(delzinv, delzinv_6);
  const flt_t fshift = SWITCHVAR(shift, shift_6);
  const flt_t fshiftone = SWITCHVAR(shiftone, shiftone_6);
  const int nxlo = SWITCHVAR(nxlo_out, nxlo_out_6);
  const int nylo = SWITCHVAR(nylo_out, nylo_out_6);
  const int nzlo = SWITCHVAR(nzlo_out, nzlo_out_6);
  const int nxhi = SWITCHVAR(nxhi_out, nxhi_out_6);
  const int nyhi = SWITCHVAR(nyhi_out, nyhi_out_6);
  const int nzhi = SWITCHVAR(nzhi_out, nzhi_out_6);
  const int nup = SWITCHVAR(nupper, nupper_6);
  const int nlow = SWITCHVAR(nlower, nlower_6);
  FFT_SCALAR ** const frho_coeff = SWITCHVAR(rho_coeff, rho_coeff_6);
  FFT_SCALAR *** const fvdx_brick = SWITCHVAR(vdx_brick, vdx_brick_g);
  FFT_SCALAR *** const fvdy_brick = SWITCHVAR(vdy_brick, vdy_brick_g);
  FFT_SCALAR *** const fvdz_brick = SWITCHVAR(vdz_brick, vdz_brick_g);

  #undef SWITCHVAR
  const flt_t fqqrd2es = qqrd2e * scale;
  const int tripcount = nup - nlow + 1;

  int niz = nzhi - nzlo + 1;
  int niy = nyhi - nylo + 1;
  int nix = nxhi - nxlo + 1;

  FFT_SCALAR vdxy_brick[2*niz*niy*nix+16]; //have to allocate for worst case where the particle is in a corner
  FFT_SCALAR vdz0_brick[2*niz*niy*nix+16];
  #if defined(LMP_SIMD_COMPILER)
    #pragma vector
    #pragma simd
  #endif
  for (int iz = 0; iz < niz; ++iz) {
    for (int iy = 0; iy < niy; ++iy) {
      for ( int ix = 0; ix < nix; ++ix) {
      int iter = 2*(iz*niy*nix + iy*nix + ix);
        vdxy_brick[iter] = fvdx_brick[nzlo + iz][nylo + iy][nxlo + ix];
        vdxy_brick[iter+1] = fvdy_brick[nzlo + iz][nylo + iy][nxlo + ix];
        vdz0_brick[iter] = fvdz_brick[nzlo + iz][nylo + iy][nxlo + ix];
        vdz0_brick[iter+1] = 0.;
      }
    }
  }

  #if defined(_OPENMP)
    #pragma omp parallel default(none) \
    shared(nlocal, nthr, nix, niy, niz, vdxy_brick, vdz0_brick)
  #endif
  { // Begin Parallel region
    int iifrom=0, iito=nlocal, tid=0;
    FFT_SCALAR rho0[16] = {0.0};
    FFT_SCALAR rho1[8];
    FFT_SCALAR rho2[8];

    IP_PRE_omp_range_id(iifrom, iito, tid, nlocal, nthr);

    for (int i = iifrom; i < iito; i++) {
      int nx = p2g[i][0];
      int ny = p2g[i][1];
      int nz = p2g[i][2];

      int nxsum = nx - nxlo + nlow;
      int nysum = ny - nylo + nlow;
      int nzsum = 2*((nz - nzlo + nlow )*nix*niy + nysum*nix + nxsum);
      FFT_SCALAR dx = nx + fshiftone - (x[i].x-lo0)*xi;
      FFT_SCALAR dy = ny + fshiftone - (x[i].y-lo1)*yi;
      FFT_SCALAR dz = nz + fshiftone - (x[i].z-lo2)*zi;

      #pragma simd
      for (int k = nlow; k <= nup; k++) {
        FFT_SCALAR r1, r2, r3;
        r1 = r2 = r3 = ZEROF;
        for (int l = order-1; l >= 0; l--) {
          r1 = frho_coeff[l][k] + r1*dx;
          r2 = frho_coeff[l][k] + r2*dy;
          r3 = frho_coeff[l][k] + r3*dz;
        }
      int rho_iter = 2*(k - nlower);
      rho0[rho_iter] = r1;
      rho0[rho_iter+1] = r1;
      rho1[k-nlower] = r2;
      rho2[k-nlower] = r3;
    }


    FFT_SCALAR ekxy[16]={ZEROF};
    FFT_SCALAR ekz0[16]={ZEROF};
    FFT_SCALAR ekxsum, ekysum, ekzsum;
    ekxsum = ekysum = ekzsum = ZEROF;
    for (int n = 0; n < tripcount; n++) {
      int mz = 2*n*nix*niy+nzsum;
      FFT_SCALAR z0 = rho2[n];

      for (int m = 0; m < tripcount; m++) {
        int mzy = mz + 2*m*nix;
        FFT_SCALAR y0 = z0*rho1[m];
        #pragma simd
        for (int l = 0; l < 16; l++) {
          FFT_SCALAR x0 = y0*rho0[l];
          ekxy[l] -= x0*vdxy_brick[mzy+l];
          ekz0[l] -= x0*vdz0_brick[mzy+l];
        }
      }
    }


    for (int l = 0; l < 16; l=l+2){
    	ekxsum += ekxy[l];
    	ekysum += ekxy[l+1];
    	ekzsum += ekz0[l];
    }

    // convert E-field to force



    flt_t ffactor;
    if (VARIANT == 'c'){
      ffactor =  fqqrd2es * q[i];
    }
    if (VARIANT == 'g'){
      const int type = atom->type[i];
      ffactor = B[type];
    }
    f[i].x += ffactor*ekxsum;
    f[i].y += ffactor*ekysum;
    if (slabflag != 2) f[i].z += qfactor*ekzsum;
  } // End loop over atoms

} // End Parallel region
}

template<const char VARIANT, class flt_t, class acc_t>
void fieldforce_ad(IntelBuffers<flt_t,acc_t> *buffers){

}
