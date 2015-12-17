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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_buck_coul_cut_intel.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "suffix.h"
#include "force.h"
#include "modify.h"


#ifdef MIC_INTRIN
  #ifdef _LMP_INTEL_OFFLOAD
  #pragma offload_attribute(push,target(mic))
  #endif

  #include "mmic.h"

  #ifdef _LMP_INTEL_OFFLOAD
  #pragma offload_attribute(pop)
  #endif
#endif


using namespace LAMMPS_NS;
using namespace MathConst;

#define C_FORCE_T typename ForceConst<flt_t>::c_force_t
#define C_ENERGY_T typename ForceConst<flt_t>::c_energy_t
#define C_CUT_T typename ForceConst<flt_t>::c_cut_t


PairBuckCoulCutIntel::PairBuckCoulCutIntel(LAMMPS *lmp) :
  PairBuckCoulCut(lmp)
{
  suffix_flag |= Suffix::INTEL;

}


PairBuckCoulCutIntel::~PairBuckCoulCutIntel()
{
}

void PairBuckCoulCutIntel::compute(int eflag, int vflag)
{
  if (fix->precision()==FixIntel::PREC_MODE_MIXED)
    compute<float,double>(eflag, vflag, fix->get_mixed_buffers(), 
                          force_const_single);
  else if (fix->precision()==FixIntel::PREC_MODE_DOUBLE)
    compute<double,double>(eflag, vflag, fix->get_double_buffers(),
                           force_const_double);
  else
    compute<float,float>(eflag, vflag, fix->get_single_buffers(),
                         force_const_single);

  fix->balance_stamp();
  vflag_fdotr = 0;
}

template <class flt_t, class acc_t>
void PairBuckCoulCutIntel::compute(int eflag, int vflag,
				     IntelBuffers<flt_t,acc_t> *buffers,
				     const ForceConst<flt_t> &fc)
{
  if (eflag || vflag) {
    ev_setup(eflag,vflag);
  } else evflag = vflag_fdotr = 0;

  const int inum = list->inum;
  const int nthreads = comm->nthreads;
  const int host_start = fix->host_start_pair();
  const int offload_end = fix->offload_end_pair();
  const int ago = neighbor->ago;

  if (ago != 0 && fix->separate_buffers() == 0) {
    fix->start_watch(TIME_PACK);
    #if defined(_OPENMP)
    #pragma omp parallel default(none) shared(eflag,vflag,buffers,fc)
    #endif
    {
      int ifrom, ito, tid;
      IP_PRE_omp_range_id_align(ifrom, ito, tid, atom->nlocal + atom->nghost, 
				nthreads, sizeof(ATOM_T));
      buffers->thr_pack(ifrom,ito,ago);
    }
    fix->stop_watch(TIME_PACK);
  }
  
  if (evflag || vflag_fdotr) {
    int ovflag = 0;
    if (vflag_fdotr) ovflag = 2;
    else if (vflag) ovflag = 1;
    if (eflag) {
      if (force->newton_pair) {
	eval<1,1,1>(1, ovflag, buffers, fc, 0, offload_end);
	eval<1,1,1>(0, ovflag, buffers, fc, host_start, inum);
      } else {
	eval<1,1,0>(1, ovflag, buffers, fc, 0, offload_end);
	eval<1,1,0>(0, ovflag, buffers, fc, host_start, inum);
      }
    } else {
      if (force->newton_pair) {
	eval<1,0,1>(1, ovflag, buffers, fc, 0, offload_end);
	eval<1,0,1>(0, ovflag, buffers, fc, host_start, inum);
      } else {
	eval<1,0,0>(1, ovflag, buffers, fc, 0, offload_end);
	eval<1,0,0>(0, ovflag, buffers, fc, host_start, inum);
      }
    }
  } else {
    if (force->newton_pair) {
      eval<0,0,1>(1, 0, buffers, fc, 0, offload_end);
      eval<0,0,1>(0, 0, buffers, fc, host_start, inum);
    } else {
      eval<0,0,0>(1, 0, buffers, fc, 0, offload_end);
      eval<0,0,0>(0, 0, buffers, fc, host_start, inum);
    }
  }
}

/* ---------------------------------------------------------------------- */

template <int EVFLAG, int EFLAG, int NEWTON_PAIR, class flt_t, class acc_t>
void PairBuckCoulCutIntel::eval(const int offload, const int vflag,
				     IntelBuffers<flt_t,acc_t> *buffers,
				     const ForceConst<flt_t> &fc,
				     const int astart, const int aend)
{
  const int inum = aend - astart;
  if (inum == 0) return;
  int nlocal, nall, minlocal;
  fix->get_buffern(offload, nlocal, nall, minlocal);

  const int ago = neighbor->ago;
  IP_PRE_pack_separate_buffers(fix, buffers, ago, offload, nlocal, nall);

  ATOM_T * _noalias const x = buffers->get_x(offload);
  flt_t * _noalias const q = buffers->get_q(offload);

  const int * _noalias const numneigh = list->numneigh;
  const int * _noalias const cnumneigh = buffers->cnumneigh(list);
  const int * _noalias const firstneigh = buffers->firstneigh(list);

  const flt_t * _noalias const special_coul = fc.special_coul;
  const flt_t * _noalias const special_lj = fc.special_lj;
  const flt_t qqrd2e = force->qqrd2e;

  const C_FORCE_T * _noalias const c_force = fc.c_force[0];
  const C_ENERGY_T * _noalias const c_energy = fc.c_energy[0];
  const C_CUT_T * _noalias const c_cut = fc.c_cut[0];

  const int ntypes = atom->ntypes + 1;
  const int eatom = this->eflag_atom;

  // Determine how much data to transfer
  int x_size, q_size, f_stride, ev_size, separate_flag;
  IP_PRE_get_transfern(ago, NEWTON_PAIR, EVFLAG, EFLAG, vflag,
		       buffers, offload, fix, separate_flag,
		       x_size, q_size, ev_size, f_stride);

  int tc;
  FORCE_T * _noalias f_start;
  acc_t * _noalias ev_global;
  IP_PRE_get_buffers(offload, buffers, fix, tc, f_start, ev_global);

  const int nthreads = tc;
  #ifdef _LMP_INTEL_OFFLOAD
  int *overflow = fix->get_off_overflow_flag();
  double *timer_compute = fix->off_watch_pair();
  // Redeclare as local variables for offload
  //const int ncoultablebits = this->ncoultablebits;
  const int ncoulmask = this->ncoulmask;
  const int ncoulshiftbits = this->ncoulshiftbits;

  if (offload) fix->start_watch(TIME_OFFLOAD_LATENCY);
  #pragma offload target(mic:_cop) if(offload)                 \
    in(special_lj,special_coul:length(0) alloc_if(0) free_if(0)) \
    in(c_force, c_energy, c_cut:length(0) alloc_if(0) free_if(0))      \
    in(firstneigh:length(0) alloc_if(0) free_if(0)) \
    in(cnumneigh:length(0) alloc_if(0) free_if(0)) \
    in(numneigh:length(0) alloc_if(0) free_if(0)) \
    in(x:length(x_size) alloc_if(0) free_if(0)) \
    in(q:length(q_size) alloc_if(0) free_if(0)) \
    in(overflow:length(0) alloc_if(0) free_if(0)) \
    in(astart,nthreads,qqrd2e,inum,nall,ntypes,vflag,eatom) \
    in(f_stride,nlocal,minlocal,separate_flag,offload) \
    out(f_start:length(f_stride) alloc_if(0) free_if(0)) \
    out(ev_global:length(ev_size) alloc_if(0) free_if(0)) \
    out(timer_compute:length(1) alloc_if(0) free_if(0)) \
    signal(f_start)
  #endif
  {
    #ifdef __MIC__
    #ifdef _LMP_INTEL_OFFLOAD
    *timer_compute = MIC_Wtime();
    #endif
    #endif

    IP_PRE_repack_for_offload(NEWTON_PAIR, separate_flag, nlocal, nall, 
			      f_stride, x, q);

    acc_t oevdwl, oecoul, ov0, ov1, ov2, ov3, ov4, ov5;
    if (EVFLAG) {
      oevdwl = oecoul = (acc_t)0;
      if (vflag) ov0 = ov1 = ov2 = ov3 = ov4 = ov5 = (acc_t)0;
    }

    // loop over neighbors of my atoms
    #if defined(_OPENMP)
    #pragma omp parallel default(none)        \
      shared(f_start,f_stride,nlocal,nall,minlocal)	\
      reduction(+:oevdwl,oecoul,ov0,ov1,ov2,ov3,ov4,ov5)
    #endif
    {
      int iifrom, iito, tid;
      IP_PRE_omp_range_id(iifrom, iito, tid, inum, nthreads);
      iifrom += astart;
      iito += astart;

      FORCE_T * _noalias const f = f_start - minlocal + (tid * f_stride);
      memset(f + minlocal, 0, f_stride * sizeof(FORCE_T));

      for (int i = iifrom; i < iito; ++i) {
        const int itype = x[i].w;

        const int ptr_off = itype * ntypes;
        const C_FORCE_T * _noalias const c_forcei = c_force + ptr_off;
        const C_ENERGY_T * _noalias const c_energyi = c_energy + ptr_off;
        const C_CUT_T * _noalias const c_cuti = c_cut + ptr_off;
        const int   * _noalias const jlist = firstneigh + cnumneigh[i];
        const int jnum = numneigh[i];

        acc_t fxtmp,fytmp,fztmp,fwtmp;
        acc_t sevdwl, secoul, sv0, sv1, sv2, sv3, sv4, sv5;
  
        const flt_t xtmp = x[i].x;
        const flt_t ytmp = x[i].y;
        const flt_t ztmp = x[i].z;
        const flt_t qtmp = q[i];
        fxtmp = fytmp = fztmp = (acc_t)0;
        if (EVFLAG) {
          if (EFLAG) fwtmp = sevdwl = secoul = (acc_t)0;
          if (vflag==1) sv0 = sv1 = sv2 = sv3 = sv4 = sv5 = (acc_t)0;
        }

        //Begin Intrinsics code
#if defined(MIC_INTRIN) && defined(__MIC__) && defined(__INTEL_COMPILER)
        typedef typename intrin::vector_op<acc_t> ac; // vector functions for acc_t precision
        typedef typename intrin::vector_op<flt_t> fl; // vector functions for flt_t precision

        __declspec(align(64)) const int jjd[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        typename ac::fvec4 faccmic;  //force accumulator (zero by default)
        typename fl::fvec4 xmici; //positions for xi
        typename fl::fvec buck1mic, buck2mic, rhoinvmic,
                          cutsqmic, cut_coulsqmic, cut_ljsqmic,
                          amic, cmic, offsetmic;


        typename ac::fvec sv0mic, sv1mic, sv2mic, sv3mic, sv4mic, sv5mic, sevdwlmic, secoulmic;
        sv0mic = ac::setzero();
        sv1mic = ac::setzero();
        sv2mic = ac::setzero();
        sv3mic = ac::setzero();
        sv4mic = ac::setzero();
        sv5mic = ac::setzero();
        sevdwlmic = ac::setzero();
        secoulmic = ac::setzero();

        typename fl::fvec qqr_i = fl::mul(fl::set1(qqrd2e),fl::set1(qtmp));
        typename fl::fmask localmaski;
        
        if(NEWTON_PAIR || i < nlocal)
          localmaski = fl::BASEMASK;
        else
          localmaski = 0;
        
        typename fl::ivec nlocalmic = fl::set1(nlocal);

        
        // Setting the value of the 1st atom type. Changing it later if necessary
        cutsqmic = fl::set1(c_cuti[1].cutsq);
        cut_coulsqmic = fl::set1(c_cuti[1].cut_coulsq);
        cut_ljsqmic = fl::set1(c_cuti[1].cut_ljsq);
        buck1mic = fl::set1(c_forcei[1].buck1);
        buck2mic = fl::set1(c_forcei[1].buck2);
        rhoinvmic = fl::set1(c_forcei[1].rhoinv);
        amic = fl::set1(c_energyi[1].a);
        cmic = fl::set1(c_energyi[1].c);
        offsetmic = fl::set1(c_energyi[1].offset);

        // Loading the position of the i atom.
        xmici.x = fl::set1(xtmp);
        xmici.y = fl::set1(ytmp);
        xmici.z = fl::set1(ztmp);

        //Vectorized LOOP trhough j atoms
        for (int jj = 0; jj < jnum; jj += fl::VL){
          //typename fl::fvec4 ftmpmic;  // temp force (zero by default)
          __declspec(align(64)) int jidx[16];
          typename fl::fmask mymask;
          int elem = jnum-jj;
          
          //Checking the # of elements we have to load in the current iteraton
          if(elem >= fl::VL)
            mymask = fl::BASEMASK;
          else
            mymask = (0x1 << elem) - 1;

          typename fl::ivec jjmic, sbmic, jmic, nmaskmic;

          jjmic = fl::add( fl::load(jjd), fl::set1(jj));
          jmic = fl::mask_gather(fl::set1(0), mymask, jjmic, jlist);
          sbmic = fl::rshift( jmic, fl::set1(SBBITS));
          sbmic = fl::bitwiseand( sbmic, fl::set1(3));
          jmic = fl::bitwiseand( jmic, fl::set1(NEIGHMASK));


          fl::mask_packstore(jidx, fl::BASEMASK, jmic);
          
          typename fl::fvec4 xmicj = fl::swizzleload(&x[0].x, jidx, mymask);

          /*
          if (1 && tid == 0 && i==(iifrom) && jj<(8)){
            __declspec(align(64)) flt_t dxmic[8], dymic[8], bfcoul[8], bfbuck[8], dximic[8];
            fl::mask_store(dxmic, fl::BASEMASK, xmicj.x );
            fl::mask_store(dymic, fl::BASEMASK, xmicj.y );
            fl::mask_store(dximic, fl::BASEMASK, xmici.x );
            for (int gg=0; gg<8; gg++)
              printf("[vec] i:%d \tj:%d \tx:%.4g \ty:%.4g \tx_i:%.4g\n",
                     i, jidx[gg], dxmic[gg], dymic[gg], dximic[gg]);
            
            }
          */
          xmicj.x = fl::sub(xmici.x, xmicj.x);
          xmicj.y = fl::sub(xmici.y, xmicj.y);
          xmicj.z = fl::sub(xmici.z, xmicj.z);

          typename fl::fvec rsqmic;
          rsqmic = fl::rsq(xmicj); //x^2+y^2+z^2;

          typename fl::ivec typemic;
          if (ntypes > 2){ //load the parameters for this type of atom
            typemic = fl::cast_int(xmicj.w);
            //the structures c_force, c_cut anc c_emnergy
            // contain 4 elements, thus we need offset:
            typemic = fl::mul(typemic, fl::set1(4));
            cutsqmic = fl::logather(typemic, &c_cuti[0].cutsq);
            cut_coulsqmic = fl::logather(typemic, &c_cuti[0].cut_coulsq);
            cut_ljsqmic = fl::logather(typemic, &c_cuti[0].cut_ljsq);
            buck1mic = fl::logather(typemic, &c_forcei[0].buck1);
            buck2mic = fl::logather(typemic, &c_forcei[0].buck2);
            rhoinvmic = fl::logather(typemic, &c_forcei[0].rhoinv);
            amic = fl::logather(typemic, &c_energyi[0].a);
            cmic = fl::logather(typemic, &c_energyi[0].c);
            offsetmic = fl::logather(typemic, &c_energyi[0].offset);
          }


 
          typename fl::fmask incutsq = fl::cmplt(rsqmic, cutsqmic);
          typename fl::fmask sbtrue = fl::cmpgt(sbmic, fl::set1(0));

          if(incutsq){
       
            typename fl::fvec rmic, rinvmic, r2invmic;
            rmic = fl::sqrt(rsqmic);
            rinvmic = fl::recip(rmic);
            r2invmic = fl::recip(rsqmic);


            typename fl::fvec4 ftmpmic;
            typename fl::fmask localmaskj;

            //Coulomb Force
            typename fl::fmask incut_coulsq = fl::cmplt(rsqmic, cut_coulsqmic);
            typename fl::fvec fcoulmic = fl::setzero();
            typename fl::fvec ecoulmic = fl::setzero();
            if(incut_coulsq){
              typename fl::fvec qmicj = fl::logather(jmic, q);
              fcoulmic = fl::mask_mul(fl::setzero(), incut_coulsq, fl::mul(qqr_i,qmicj), rinvmic);
              if(EFLAG)
                ecoulmic = fcoulmic;
              if(sbtrue){
                typename fl::fvec factor_coulmic = fl::mask_logather(fl::set1((flt_t)1.0),
                                                                     sbtrue,  sbmic, special_coul);
                fcoulmic = fl::mul(factor_coulmic, fcoulmic);
                if(EFLAG)
                  ecoulmic = fl::mul(factor_coulmic, ecoulmic);
              }
            }

          
          
            //Buck force (lj cut)
            typename fl::fmask incut_ljsq = fl::cmplt(rsqmic, cut_ljsqmic);
            typename fl::fvec fbuckmic = fl::setzero();
            typename fl::fvec evdwlmic = fl::setzero();
            if(incut_ljsq){
              typename fl::fvec r6invmic = fl::mul(r2invmic, fl::mul(r2invmic, r2invmic));
              typename fl::fvec rexpmic = fl::exp(fl::fnmadd(rmic, rhoinvmic, fl::setzero()));
              fbuckmic = fl::mul(rmic, fl::mul(rexpmic,buck1mic));
              fbuckmic = fl::fnmadd(r6invmic, buck2mic, fbuckmic);
              fbuckmic = fl::mask_mov(fl::setzero(), incut_ljsq, fbuckmic);
              if(EFLAG){
                evdwlmic = fl::mul(rexpmic, amic);
                evdwlmic = fl::fnmadd(r6invmic, cmic, evdwlmic);
                evdwlmic = fl::mask_sub(fl::setzero(), incut_ljsq, evdwlmic, offsetmic);
              }

              if(sbtrue){
                typename fl::fvec factor_buckmic = fl::mask_logather(fl::set1((flt_t)1.0),
                                                                     sbtrue,  sbmic, special_lj);
                fbuckmic = fl::mul(factor_buckmic, fbuckmic);
                if(EFLAG)
                  evdwlmic = fl::mul(factor_buckmic, fbuckmic);
              }
            }

            typename fl::fvec fpairmic = fl::mul(r2invmic, fl::add(fcoulmic, fbuckmic));
            ftmpmic.x = fl::mul(fpairmic, xmicj.x);
            ftmpmic.y = fl::mul(fpairmic, xmicj.y);
            ftmpmic.z = fl::mul(fpairmic, xmicj.z);
            

            // Add to te accu
            faccmic.x = ac::mask_accu(faccmic.x, mymask, ftmpmic.x);
            faccmic.y = ac::mask_accu(faccmic.y, mymask, ftmpmic.y);
            faccmic.z = ac::mask_accu(faccmic.z, mymask, ftmpmic.z);
           

            if (NEWTON_PAIR){
              localmaskj = mymask;
            }
            else{
              localmaskj = fl::cmplt(jmic, nlocalmic);
              localmaskj &= mymask;
            }
            

            if (EVFLAG){
              typename fl::fvec ev_premic;
              if (localmaski)
                ev_premic = fl::set1((flt_t)0.5);
              else
                ev_premic = fl::setzero();
              
              ev_premic = fl::mask_add(ev_premic, localmaskj, ev_premic, fl::set1((flt_t)0.5));


              if (EFLAG){

                sevdwlmic = ac::mask_accu(sevdwlmic, mymask, fl::mul(ev_premic, evdwlmic));
                secoulmic = ac::mask_accu(secoulmic, mymask, fl::mul(ev_premic, ecoulmic));


                if (eatom){
                  ftmpmic.w = fl::mask_mul(ftmpmic.w, localmaski, 
                                           fl::set1((flt_t)0.5), fl::add(evdwlmic, ecoulmic));
                  faccmic.w = ac::mask_accu(faccmic.w, mymask, ftmpmic.w);
                }
              }
              
              if(vflag == 1){
                typename fl::fvec evpmic = fl::mul(ev_premic, fpairmic);
                typename fl::fvec evprexp = fl::mul(xmicj.x, evpmic);
                typename fl::fvec evpreyp = fl::mul(xmicj.y, evpmic);
                typename fl::fvec evprezp = fl::mul(xmicj.z, evpmic);

                sv0mic = ac::mask_accu(sv0mic, mymask, fl::mul(evprexp, xmicj.x));
                sv1mic = ac::mask_accu(sv1mic, mymask, fl::mul(evpreyp, xmicj.y));
                sv2mic = ac::mask_accu(sv2mic, mymask, fl::mul(evprezp, xmicj.z));
                sv3mic = ac::mask_accu(sv3mic, mymask, fl::mul(evprexp, xmicj.y));
                sv4mic = ac::mask_accu(sv4mic, mymask, fl::mul(evprexp, xmicj.z));
                sv5mic = ac::mask_accu(sv5mic, mymask, fl::mul(evpreyp, xmicj.z));


              }
            }

                                 

            // correct to add energy to f[j].w
            if(EFLAG && eatom)
              ftmpmic.w = fl::mul(fl::set1((flt_t)-0.5), fl::add(evdwlmic, ecoulmic));
            else
              ftmpmic.w = fl::setzero();
            // Update j                        
            if (localmaskj){
              ftmpmic.x = fl::mask_mov(fl::setzero(), localmaskj, ftmpmic.x);
              ftmpmic.y = fl::mask_mov(fl::setzero(), localmaskj, ftmpmic.y);
              ftmpmic.z = fl::mask_mov(fl::setzero(), localmaskj, ftmpmic.z);
              
              ac::swizzlesubstore(&f[0].x, ftmpmic, jidx, mymask);            

            }

          } // in cutsq


        } // for jj

        if (EVFLAG){
          if(vflag == 1){
            sv0 += ac::mask_reduce_add(ac::BASEMASK, sv0mic);
            sv1 += ac::mask_reduce_add(ac::BASEMASK, sv1mic);
            sv2 += ac::mask_reduce_add(ac::BASEMASK, sv2mic);
            sv3 += ac::mask_reduce_add(ac::BASEMASK, sv3mic);
            sv4 += ac::mask_reduce_add(ac::BASEMASK, sv4mic);
            sv5 += ac::mask_reduce_add(ac::BASEMASK, sv5mic);
          }
        }

        if (EFLAG){
          sevdwl += ac::mask_reduce_add(ac::BASEMASK, sevdwlmic);
          secoul += ac::mask_reduce_add(ac::BASEMASK, secoulmic);
          fwtmp = ac::mask_reduce_add(ac::BASEMASK, faccmic.w);
        }

        fxtmp = ac::mask_reduce_add(ac::BASEMASK, faccmic.x);
        fytmp = ac::mask_reduce_add(ac::BASEMASK, faccmic.y);
        fztmp = ac::mask_reduce_add(ac::BASEMASK, faccmic.z);

        //End intrinsics code
#else       

        #if defined(__INTEL_COMPILER) && defined(SIMD_PRAGMA)
        	#pragma vector aligned
	        #pragma simd reduction(+:fxtmp, fytmp, fztmp, fwtmp, sevdwl, secoul, \
	                       sv0, sv1, sv2, sv3, sv4, sv5) 
        #endif
        for (int jj = 0; jj < jnum; jj++) {
          flt_t forcecoul, forcebuck, evdwl, ecoul;
          forcecoul = forcebuck = evdwl = ecoul = (flt_t)0.0;

          const int sbindex = jlist[jj] >> SBBITS & 3;
          const int j = jlist[jj] & NEIGHMASK;

          const flt_t delx = xtmp - x[j].x;
          const flt_t dely = ytmp - x[j].y;
          const flt_t delz = ztmp - x[j].z;
          const int jtype = x[j].w;
          const flt_t rsq = delx * delx + dely * dely + delz * delz;
          const flt_t r = sqrt(rsq);

          const flt_t r2inv = (flt_t)1.0 / rsq;

	  
          #ifdef __MIC__ 
          if (rsq < c_cuti[jtype].cut_coulsq) {
          #endif
            forcecoul = qqrd2e * qtmp*q[j]/r;
            if (EFLAG) 
              ecoul = forcecoul;
            if (sbindex){
              const flt_t factor_coul = special_coul[sbindex];
              forcecoul *= factor_coul;
              if(EFLAG)
                ecoul *= factor_coul;
              
            }
          #ifdef __MIC__
          }
          #else
          if (rsq >= c_cuti[jtype].cut_coulsq)
            { forcecoul = (flt_t)0.0; ecoul = (flt_t)0.0; }
          #endif
          
          #ifdef __MIC__
          if (rsq < c_cuti[jtype].cut_ljsq) {
          #endif
            flt_t r6inv = r2inv * r2inv * r2inv;
            flt_t rexp = exp(-r * c_forcei[jtype].rhoinv);
            forcebuck = r * rexp * c_forcei[jtype].buck1 -
              r6inv * c_forcei[jtype].buck2;
            if (EFLAG) 
              evdwl = rexp * c_energyi[jtype].a -
                r6inv * c_energyi[jtype].c -
                c_energyi[jtype].offset;
            if (sbindex) {
              const flt_t factor_lj = special_lj[sbindex];
              forcebuck *= factor_lj;
              if (EFLAG) 
                evdwl *= factor_lj;
            }
          #ifdef __MIC__
          }
          #else
          if (rsq >= c_cuti[jtype].cut_ljsq)
            { forcebuck = (flt_t)0.0; evdwl = (flt_t)0.0; }
          #endif

          #ifdef __MIC__
              if (rsq < c_cuti[jtype].cutsq) {
          #endif
            const flt_t fpair = (forcecoul + forcebuck) * r2inv;
            fxtmp += delx * fpair;
            fytmp += dely * fpair;
            fztmp += delz * fpair;
            if (NEWTON_PAIR || j < nlocal) {
              f[j].x -= delx * fpair;
              f[j].y -= dely * fpair;
              f[j].z -= delz * fpair;
            }
            
            if (EVFLAG) {
              flt_t ev_pre = (flt_t)0;
              if (NEWTON_PAIR || i < nlocal)
                ev_pre += (flt_t)0.5;
              if (NEWTON_PAIR || j < nlocal)
                ev_pre += (flt_t)0.5;
              
              if (EFLAG) {
                sevdwl += ev_pre * evdwl;
                secoul += ev_pre * ecoul;
                if (eatom) {
                  if (NEWTON_PAIR || i < nlocal)
                    fwtmp += (flt_t)0.5 * evdwl + (flt_t)0.5 * ecoul;
                  if (NEWTON_PAIR || j < nlocal) 
                    f[j].w += (flt_t)0.5 * evdwl + (flt_t)0.5 * ecoul;
                }
              }
              IP_PRE_ev_tally_nbor(vflag, ev_pre, fpair, delx, dely, delz);
            }
          #ifdef __MIC__
          }
          #endif
        } // for jj
#endif

        f[i].x += fxtmp;
        f[i].y += fytmp;
        f[i].z += fztmp;

        IP_PRE_ev_tally_atomq(EVFLAG, EFLAG, vflag, f, fwtmp);
      } // for ii
      #if defined(_OPENMP)
      #pragma omp barrier
      #endif
      IP_PRE_fdotr_acc_force(NEWTON_PAIR, EVFLAG,  EFLAG, vflag, eatom, nall,
			     nlocal, minlocal, nthreads, f_start, f_stride, 
			     x);
    } // end of omp parallel region
    if (EVFLAG) {
      if (EFLAG) {
        ev_global[0] = oevdwl;
        ev_global[1] = oecoul;
      }
      if (vflag) {
        ev_global[2] = ov0;
        ev_global[3] = ov1;
        ev_global[4] = ov2;
        ev_global[5] = ov3;
        ev_global[6] = ov4;
        ev_global[7] = ov5;
      }
    }
    #ifdef __MIC__
    #ifdef _LMP_INTEL_OFFLOAD
    *timer_compute = MIC_Wtime() - *timer_compute;
    #endif
    #endif
  } // end of offload region

  if (offload)
    fix->stop_watch(TIME_OFFLOAD_LATENCY);
  else
    fix->stop_watch(TIME_HOST_PAIR);

  if (EVFLAG)
    fix->add_result_array(f_start, ev_global, offload, eatom);
  else
    fix->add_result_array(f_start, 0, offload);
}

/* ---------------------------------------------------------------------- */

void PairBuckCoulCutIntel::init_style()
{
  PairBuckCoulCut::init_style();
  neighbor->requests[neighbor->nrequest-1]->intel = 1;

  int ifix = modify->find_fix("package_intel");
  if (ifix < 0)
    error->all(FLERR,
               "The 'package intel' command is required for /intel styles");
  fix = static_cast<FixIntel *>(modify->fix[ifix]);
  
  fix->pair_init_check();
  #ifdef _LMP_INTEL_OFFLOAD
  _cop = fix->coprocessor_number();
  #endif

  if (fix->precision() == FixIntel::PREC_MODE_MIXED)
    pack_force_const(force_const_single, fix->get_mixed_buffers());
  else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE)
    pack_force_const(force_const_double, fix->get_double_buffers());
  else
    pack_force_const(force_const_single, fix->get_single_buffers());

}

template <class flt_t, class acc_t>
void PairBuckCoulCutIntel::pack_force_const(ForceConst<flt_t> &fc,
                                          IntelBuffers<flt_t,acc_t> *buffers)
{
  int tp1 = atom->ntypes + 1;
  int ntable = 1;
  if (ncoultablebits)
    for (int i = 0; i < ncoultablebits; i++) ntable *= 2;

  fc.set_ntypes(tp1, ntable, memory, _cop);
  buffers->set_ntypes(tp1);
  flt_t **cutneighsq = buffers->get_cutneighsq();

  // Repeat cutsq calculation because done after call to init_style
  double cut, cutneigh;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        cut = init_one(i, j);
        cutneigh = cut + neighbor->skin;
        cutsq[i][j] = cutsq[j][i] = cut*cut;
        cutneighsq[i][j] = cutneighsq[j][i] = cutneigh * cutneigh;
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    fc.special_lj[i] = force->special_lj[i];
    fc.special_coul[i] = force->special_coul[i];
    fc.special_coul[0] = 1.0;
    fc.special_lj[0] = 1.0;
  }

  for (int i = 0; i < tp1; i++) {
    for (int j = 0; j < tp1; j++) {
      fc.c_cut[i][j].cutsq = cutsq[i][j];
      fc.c_cut[i][j].cut_ljsq = cut_ljsq[i][j];
      fc.c_cut[i][j].cut_coulsq = cut_coulsq[i][j];
      fc.c_force[i][j].buck1 = buck1[i][j];
      fc.c_force[i][j].buck2 = buck2[i][j];
      fc.c_force[i][j].rhoinv = rhoinv[i][j];
      fc.c_energy[i][j].a = a[i][j];
      fc.c_energy[i][j].c = c[i][j];
      fc.c_energy[i][j].offset = offset[i][j];
    }
  }

  #ifdef _LMP_INTEL_OFFLOAD
  if (_cop < 0) return;
  flt_t * special_lj = fc.special_lj;
  flt_t * special_coul = fc.special_coul;
  C_FORCE_T * c_force = fc.c_force[0];
  C_ENERGY_T * c_energy = fc.c_energy[0];
  C_CUT_T * c_cut = fc.c_cut[0];
  flt_t * ocutneighsq = cutneighsq[0];
  int tp1sq = tp1 * tp1;
  #pragma offload_transfer target(mic:_cop) \
    in(special_lj, special_coul: length(4) alloc_if(0) free_if(0)) \
    in(c_force, c_energy, c_cut: length(tp1sq) alloc_if(0) free_if(0))   \
    in(ocutneighsq: length(tp1sq) alloc_if(0) free_if(0))
  #endif
}

/* ---------------------------------------------------------------------- */

template <class flt_t>
void PairBuckCoulCutIntel::ForceConst<flt_t>::set_ntypes(const int ntypes,
							   const int ntable,
							   Memory *memory,
							   const int cop) {
  if ( (ntypes != _ntypes || ntable != _ntable) ) {
    if (_ntypes > 0) {
      #ifdef _LMP_INTEL_OFFLOAD
      flt_t * ospecial_lj = special_lj;
      flt_t * ospecial_coul = special_coul;
      c_force_t * oc_force = c_force[0];
      c_energy_t * oc_energy = c_energy[0];
      c_cut_t * oc_cut = c_cut[0];

      if (ospecial_lj != NULL && oc_force != NULL && oc_cut != NULL &&
          oc_energy != NULL && ospecial_coul != NULL && 
          _cop >= 0) {
        #pragma offload_transfer target(mic:cop) \
          nocopy(ospecial_lj, ospecial_coul: alloc_if(0) free_if(1)) \
          nocopy(oc_force, oc_energy: alloc_if(0) free_if(1))        \
          nocopy(oc_cut: alloc_if(0) free_if(1)) 
      }
      #endif

      _memory->destroy(c_force);
      _memory->destroy(c_energy);
      _memory->destroy(c_cut);

    }
    if (ntypes > 0) {
      _cop = cop;
      memory->create(c_force,ntypes,ntypes,"fc.c_force");
      memory->create(c_energy,ntypes,ntypes,"fc.c_energy");
      memory->create(c_cut,ntypes,ntypes,"fc.c_cut");


      #ifdef _LMP_INTEL_OFFLOAD
      flt_t * ospecial_lj = special_lj;
      flt_t * ospecial_coul = special_coul;
      c_force_t * oc_force = c_force[0];
      c_energy_t * oc_energy = c_energy[0];
      c_cut_t * oc_cut = c_cut[0];
      int tp1sq = ntypes*ntypes;
      if (ospecial_lj != NULL && oc_force != NULL && oc_cut != NULL &&
          oc_energy != NULL && ospecial_coul != NULL &&  
          cop >= 0) {
        #pragma offload_transfer target(mic:cop) \
          nocopy(ospecial_lj: length(4) alloc_if(1) free_if(0)) \
          nocopy(ospecial_coul: length(4) alloc_if(1) free_if(0)) \
          nocopy(oc_force: length(tp1sq) alloc_if(1) free_if(0)) \
          nocopy(oc_energy: length(tp1sq) alloc_if(1) free_if(0)) \
          nocopy(oc_cut: length(tp1sq) alloc_if(1) free_if(0))

      }
      #endif
    }
  }
  _ntypes=ntypes;
  _ntable=ntable;
  _memory=memory;
}


