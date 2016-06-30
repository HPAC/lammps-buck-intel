/* -*- c++ -*- ----------------------------------------------------------
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
   Contributing author: Rodrigo Canales (RWTH Aachen University)
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(pppm/disp/intel,PPPMDispIntel)

#else

#ifndef LMP_PPPMDISPINTEL_H
#define LMP_PPPMDISPINTEL_H

#include "pppm_disp.h"
#include "fix_intel.h"
#include "remap_wrap.h"

namespace LAMMPS_NS {

class PPPMDispIntel : public PPPMDisp {
 public:
  PPPMDispIntel(class LAMMPS *, int, char **);
  virtual ~PPPMDispIntel();
  virtual void init();
  virtual void compute(int, int);
  //  virtual void brick2fft();

 protected:
  FixIntel *fix;

  #ifdef _LMP_INTEL_OFFLOAD
  int _use_base;
  #endif

  void poisson_ik(FFT_SCALAR* wk1, FFT_SCALAR* wk2,
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
                  FFT_SCALAR*** v3_pa, FFT_SCALAR*** v4_pa, FFT_SCALAR*** v5_pa);
  // template<class flt_t, class acc_t>
  // void particle_map(IntelBuffers<flt_t,acc_t> *buffers);
  // template<class flt_t, class acc_t>
  // void make_rho(IntelBuffers<flt_t,acc_t> *buffers);
  // template<class flt_t, class acc_t>
  // void fieldforce_ik(IntelBuffers<flt_t,acc_t> *buffers);
  // template<class flt_t, class acc_t>
  // void fieldforce_ad(IntelBuffers<flt_t,acc_t> *buffers);
  // template<class flt_t, class acc_t>
  // void poisson_ik(IntelBuffers<flt_t,acc_t> *buffers);
  // template<class flt_t, class acc_t>
  // void poisson_ad(IntelBuffers<flt_t,acc_t> *buffers);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: PPPM order greater than supported by USER-INTEL

There is a compile time limit on the maximum order for PPPM
in the USER-INTEL package that might be different from LAMMPS

*/
