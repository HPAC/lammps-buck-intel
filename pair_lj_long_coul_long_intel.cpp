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


#define D_PARAM double

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

void PairLJLongCoulLongIntel::compute(int eflag, int vflag)
{
  if (fix->precision()==FixIntel::PREC_MODE_MIXED)
    compute<float,double>(eflag, vflag, fix->get_mixed_buffers());
  else if (fix->precision()==FixIntel::PREC_MODE_DOUBLE)
    compute<double,double>(eflag, vflag, fix->get_double_buffers());
  else
    compute<float,float>(eflag, vflag, fix->get_single_buffers());

  vflag_fdotr = 0;
}


// void PairLJLongCoulLongIntel::compute_inner(int eflag, int vflag)
// {
//   if (fix->precision()==FixIntel::PREC_MODE_MIXED)
//     compute_inner<float,double>(eflag, vflag, fix->get_mixed_buffers(),
//                           force_const_single);
//   else if (fix->precision()==FixIntel::PREC_MODE_DOUBLE)
//     compute_inner<double,double>(eflag, vflag, fix->get_double_buffers(),
//                            force_const_double);
//   else
//     compute_inner<float,float>(eflag, vflag, fix->get_single_buffers(),
//                          force_const_single);

//   vflag_fdotr = 0;
// }

// void PairLJLongCoulLongIntel::compute_outer(int eflag, int vflag)
// {
//   if (fix->precision()==FixIntel::PREC_MODE_MIXED)
//     compute_outer<float,double>(eflag, vflag, fix->get_mixed_buffers(),
//                           force_const_single);
//   else if (fix->precision()==FixIntel::PREC_MODE_DOUBLE)
//     compute_outer<double,double>(eflag, vflag, fix->get_double_buffers(),
//                            force_const_double);
//   else
//     compute_outer<float,float>(eflag, vflag, fix->get_single_buffers(),
//                          force_const_single);

//   vflag_fdotr = 0;
// }

/* ---------------------------------------------------------------------- */


template <class flt_t, class acc_t>
void PairLJLongCoulLongIntel::compute(int eflag, int vflag,
				      IntelBuffers<flt_t,acc_t> *buffers)
{

  // PENDING
  // Call base !!


  if (eflag || vflag)
    ev_setup(eflag,vflag);
  else 
    evflag = vflag_fdotr = 0;

  const int order1 = ewald_order&(1<<1);
  const int order6 = ewald_order&(1<<6);

  //Atom data:
  const int inum = list->inum;
  const int nthreads = comm->nthreads;

  const int host_start = fix->host_start_pair();
  const int ago = neighbor->ago;

  //Pack the data into the internal array
  fix->start_watch(TIME_PACK);
#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {
    int ifrom, ito, tid;
    IP_PRE_omp_range_id_align(ifrom, ito, tid, atom->nlocal + atom->nghost,
                              nthreads, sizeof(ATOM_T));
    buffers->thr_pack(ifrom,ito,ago);
  }
  fix->stop_watch(TIME_PACK);
  int ifrom = 0;
  int ito = inum;
  if (order6){
    if (order1){
      if (ndisptablebits){
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,1,1,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,1,1,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,1,1,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,1,1,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,1,1,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,1,1,1>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,1,1,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,1,1,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,1,1,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,1,1,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,1,1,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,1,1,1>(ifrom, ito, buffers);
          }
        }
      }
      else { //ndisptablebits=0
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,0,1,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,0,1,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,0,1,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,0,1,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,0,1,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,0,1,1>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,0,1,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,0,1,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,0,1,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,0,1,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,0,1,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,0,1,1>(ifrom, ito, buffers);
          }
        }
      }
    }
    else{ //order1=0
      if (ndisptablebits){
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,1,0,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,1,0,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,1,0,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,1,0,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,1,0,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,1,0,1>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,1,0,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,1,0,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,1,0,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,1,0,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,1,0,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,1,0,1>(ifrom, ito, buffers);
          }
        }
      }
      else { //ndisptablebits=0
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,0,0,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,0,0,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,0,0,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,0,0,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,0,0,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,0,0,1>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,0,0,1>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,0,0,1>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,0,0,1>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,0,0,1>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,0,0,1>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,0,0,1>(ifrom, ito, buffers);
          }
        }
      }
    }
  }
  else{ //order6=0
    if (order1){
      if (ndisptablebits){
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,1,1,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,1,1,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,1,1,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,1,1,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,1,1,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,1,1,0>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,1,1,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,1,1,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,1,1,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,1,1,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,1,1,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,1,1,0>(ifrom, ito, buffers);
          }
        }
      }
      else { //ndisptablebits=0
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,0,1,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,0,1,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,0,1,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,0,1,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,0,1,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,0,1,0>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,0,1,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,0,1,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,0,1,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,0,1,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,0,1,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,0,1,0>(ifrom, ito, buffers);
          }
        }
      }
    }
    else{ //order1=0
      if (ndisptablebits){
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,1,0,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,1,0,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,1,0,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,1,0,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,1,0,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,1,0,0>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,1,0,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,1,0,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,1,0,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,1,0,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,1,0,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,1,0,0>(ifrom, ito, buffers);
          }
        }
      }
      else { //ndisptablebits=0
        if (ncoultablebits){
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,1,0,0,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,1,0,0,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,1,0,0,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,1,0,0,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,1,0,0,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,1,0,0,0>(ifrom, ito, buffers);
          }
        }
        else { //ncoultablebits=0
          if (evflag) {
            if (eflag) {
              if (force->newton_pair)
                eval<1,1,1,0,0,0,0>(ifrom, ito, buffers);
              else
                eval<1,1,0,0,0,0,0>(ifrom, ito, buffers);
            }
            else{ //eflag=0
              if (force->newton_pair)
                eval<1,0,1,0,0,0,0>(ifrom, ito, buffers);
              else
                eval<1,0,0,0,0,0,0>(ifrom, ito, buffers);
            }
          }
          else { //evflag=0
            if (force->newton_pair)
              eval<0,0,1,0,0,0,0>(ifrom, ito, buffers);
            else
              eval<0,0,0,0,0,0,0>(ifrom, ito, buffers);
          }
        }
      }
    }
  } // end dispatch



} // end ::compute



      //  EVFLAG, EFLAG, NEWTON_PAIR, CTABLE, LJTABLE, ORDER1, ORDER6      

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1,
          const int ORDER6, class flt_t, class acc_t>
void PairLJLongCoulLongIntel::eval
(int ifrom, int ito, IntelBuffers<flt_t, acc_t> *buffers){

  const int offload = 0;
  int nall = atom->nlocal + atom->nghost;

  int nlocal = atom->nlocal;
  int nthr = comm->nthreads;
  

  const int ago = neighbor->ago;
  
  ATOM_T * _noalias const x = buffers->get_x(0);
  flt_t * _noalias const q = buffers->get_q(0);
  FORCE_T * _noalias const f = buffers->get_f();

  const int * _noalias const cnumneigh = buffers->cnumneigh(list);
  const int * _noalias const firstneigh = buffers->firstneigh(list);


  const flt_t qqrd2e = force->qqrd2e;
  const D_PARAM *special_coul = force->special_coul;
  const D_PARAM *special_lj = force->special_lj;
  
  acc_t oevdwl, oecoul, ov0, ov1, ov2, ov3, ov4, ov5;
  if (EVFLAG) {
    oevdwl = oecoul = (acc_t)0;
    ov0 = ov1 = ov2 = ov3 = ov4 = ov5 = (acc_t)0;
  }


  for (int i = ifrom; i < ito; ++i){

    if (ORDER1)
      flt_t qri = q[i] * qqrd2e;
    
    //Declaring data dependent on i for its reuse on the neighbor loop
    const int typei = x[i].w;
    const flt_t xtmp = x[i].x;
    const flt_t ytmp = x[i].y;
    const flt_t ztmp = x[i].z;

    const flt_t qtmp = q[i];    

    acc_t fxtmp,fytmp,fztmp,fwtmp;
    acc_t sevdwl, secoul, sv0, sv1, sv2, sv3, sv4, sv5;


    D_PARAM *lj1i = lj1[typei];
    D_PARAM *lj2i = lj2[typei];
    D_PARAM *lj3i = lj3[typei];
    D_PARAM *lj4i = lj4[typei];

    D_PARAM *offseti = offset[typei];
    D_PARAM *cutsqi = cutsq[typei];
    D_PARAM *cut_ljsqi = cut_ljsq[typei];
    D_PARAM *cut_coulsqi = cut_coulsq[typei];

    const int jnum = numneigh[i];
    const int   * _noalias const jlist = firstneigh + cnumneigh[i];
    
    for(int jj = 0; jj < jnum; ++jj) {
      
      flt_t forcecoul, forcelj, evdwl, ecoul;
      forcecoul = forcelj = evdwl = ecoul = (flt_t)0.0;
      
      const int sbindex = jlist[jj] >> SBBITS & 3;
      const int j = jlist[jj] & NEIGHMASK;
      
      const flt_t delx = x[j].x - xtmp;
      const flt_t dely = x[j].y - ytmp;
      const flt_t delz = x[j].z - ztmp;
      const int typej = x[j].w;



      const flt_t rsq = delx * delx + dely * dely + delz * delz;
      const flt_t r2inv = (flt_t)1.0/rsq;
      
      if ( ORDER1 && rsq < cut_coulsqi[typej]){
	if (!CTABLE || rsq <= tabinnersq){
	  const flt_t A1 =  0.254829592;
	  const flt_t A2 = -0.284496736;
	  const flt_t A3 =  1.421413741;
	  const flt_t A4 = -1.453152027;
	  const flt_t A5 =  1.061405429;
	  const flt_t EWALD_F = 1.12837917;
	  const flt_t INV_EWALD_P = 1.0 / 0.3275911;
	    
	  const flt_t r = sqrt(rsq);
	  const flt_t grij = g_ewald * r;
	  const flt_t expm2 = exp(-grij * grij);
	  const flt_t t = INV_EWALD_P / (INV_EWALD_P + grij);
	  const flt_t erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	  const flt_t prefactor = qqrd2e * qtmp * q[j] / r;
	    
	    
	  forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
	    
	  if (EFLAG) 
	    ecoul = prefactor * erfc;
	    
	  if (sbindex){
	    const flt_t adjust = ((flt_t)1.0 - special_coul[sbindex])*
	      prefactor;
	    forcecoul -= adjust;
	    
	    if (EFLAG) 
	      ecoul -= adjust;
	  }
	}
	else {
	  float rsq_lookup = rsq;
	  const int itable = (__intel_castf32_u32(rsq_lookup) &
			      ncoulmask) >> ncoulshiftbits;

	  const flt_t fraction = (rsq_lookup - rtable[itable]) *
	    drtable[itable];
	    
	  const flt_t tablet = ftable[itable] + fraction * dftable[itable];
	  forcecoul = qtmp * q[j] * tablet;
	  if (EFLAG) 
	    ecoul = qtmp * q[j] * (etable[itable] +
				   fraction * detable[itable]);
	  if (sbindex) {
	    const flt_t table2 = ctable[itable] + fraction * dctable[itable];
	    const flt_t prefactor = qtmp * q[j] * table2;
	    const flt_t adjust = ((flt_t)1.0 - special_coul[sbindex]) *
	      prefactor;
	    forcecoul -= adjust;
	    if (EFLAG) 
	      ecoul -= adjust;
	  }
	}

      }
      else {
	forcecoul = ecoul = 0.0;	  
      }
      if (rsq < cut_ljsqi[typej]) {
	if (ORDER6) {
	  if (!LJTABLE || rsq <= tabinnerdispsq ) {
	    const flt_t r6inv = r2inv * r2inv * r2inv;
	    const flt_t grij2 = g2 * rsq;
	    const flt_t a2 = (flt_t)1.0/grij2;
	    const flt_t x2 = a2 * exp(-grij2) * lj4i[typej];
	    forcelj = r6inv * r6inv * lj1i[typej] - g8 * x2 * rsq *
	      (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + 1.0 );
	    if (EFLAG)
	      evdwl = r6inv * r6inv * lj3i[typej] - g6 * x2 * ((a2 + 1.0) * a2 + 0.5);

	    if(sbindex){
	      const flt_t f = special_lj[sbindex];
	      const flt_t t = r6inv * (1.0 - f);
	      forcelj += t * (lj2i[typej] - r6inv * lj1i[typej]);
	      if (EFLAG)
		evdwl += t * (lj4i[typej] - r6inv * lj3i[typej]);
	    }
	  }
	  else{
	    float rsq_lookup = rsq;
	    const int itable = (__intel_castf32_u32(rsq_lookup) &
				ndispmask) >> ndispshiftbits;
	    const flt_t fdisp = (rsq - rdisptable[itable]) * drdisptable[itable];
	    const flt_t r6inv = r2inv * r2inv * r2inv;
	    forcelj = r6inv * r6inv*lj1i[typej] - 
	      (fdisptable[itable] + fdisp * dfdisptable[itable]) * lj4i[typej];
	    if(EFLAG)
	      evdwl = r6inv * r6inv * lj3i[typej] - 
		(edisptable[itable] + fdisp * dedisptable[disp_k]) * lj4i[typej];
	    if(sbindex){
	      const flt_t f = special_lj[sbindex];
	      const flt_t t = r6inv * (1.0 - f);
	      forcelj += t * (lj2i[typej] - r6inv * lj1i[typej]);
	      if (EFLAG)
		evdwl += t * (lj4i[typej] - r6inv * lj3i[typej]);
	    }
	  }
	}
	else{
	  const flt_t r6inv = r2inv * r2inv * r2inv;
	  forcelj = r6inv * ( r6inv * lj1i[typej] - lj2i[typej]);
	  if (EFLAG)
	    evdwl = r6inv * ( r6inv * lj3i[typej] - lj4i[typej]) - offseti[typej];
	  if(sbindex){
	    const flt_t factor_lj = special_lj[sbindex];
	    forcelj *= factor_lj;
	    if (EFLAG)
	      evdwl *= factor_lj;
	  }
	    
	}
      }
      else {
	forcelj = 0.0;
	evdwl = 0.0;
      }
      const flt_t fpair = (forcecoul + forcelj) * r2inv;
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
    } // for jj
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
    IP_PRE_ev_tally_atomq(EVFLAG, EFLAG, vflag, f, fwtmp);
    
  } //for ii
  if (vflag == 2){
  #if defined(_OPENMP) && 0
    #pragma omp barrier
  #endif
    IP_PRE_fdotr_acc_force(NEWTON_PAIR, EVFLAG,  EFLAG, vflag, eatom, nall,
      nlocal, minlocal, nthreads, f_start, f_stride,
      x, offload);
  }
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
  if (EVFLAG)
    fix->add_result_array(f_start, ev_global, offload, eatom, 0, vflag);
  else
    fix->add_result_array(f_start, 0, offload);

}
      


/*
  acc_t evdwl, ecoul, fpair;

  
  const int inum = list->inum;
  const int nthreads = comm->nthreads;
  const int host_start = fix->host_start_pair();
  const int offload_end = fix->offload_end_pair();
  const int ago = neighbor->ago;

  if (evflag || vflag_fdotr)

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
*/

void PairLJLongCoulLongIntel::init_style()
{
  PairLJLongCoulLong::init_style();
  //neighbor->requests[neighbor->nrequest-1]->intel = 1;

  int ifix = modify->find_fix("package_intel");
  if (ifix < 0)
    error->all(FLERR,
               "The 'package intel' command is required for /intel styles");
  fix = static_cast<FixIntel *>(modify->fix[ifix]);

  fix->pair_init_check();

  // if (fix->precision() == FixIntel::PREC_MODE_MIXED)
  //   pack_force_const(force_const_single, fix->get_mixed_buffers());
  // else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE)
  //   pack_force_const(force_const_double, fix->get_double_buffers());
  // else
  //   pack_force_const(force_const_single, fix->get_single_buffers());
}

// template <class flt_t, class acc_t>
// void PairLJCutCoulLongIntel::pack_force_const
// (ForceConst<flt_t> &fc, IntelBuffers<flt_t, acc_t> *buffers){

//   int tp1 = atom->ntypes + 1;
//   int ntable = 1;
//   if (ncoultablebits)
//     for (int i = 0; i < ncoultablebits; i++) 
//       ntable *= 2;

//   fc.set_ntypes(tp1, ntable, memory, _cop);
//   buffers->set_ntypes(tp1);
//   flt_t **cutneighsq = buffers->get_cutneighsq();

//   // Repeat cutsq calculation because done after call to init_style
//   double cut, cutneigh;
//   for (int i = 1; i <= atom->ntypes; i++) {
//     for (int j = i; j <= atom->ntypes; j++) {
//       if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
//         cut = init_one(i, j);
//         cutneigh = cut + neighbor->skin;
//         cutsq[i][j] = cutsq[j][i] = cut*cut;
//         cutneighsq[i][j] = cutneighsq[j][i] = cutneigh * cutneigh;
//       }
//     }
//   }

//   fc.g_ewald = force->kspace->g_ewald;
//   fc.tabinnersq = tabinnersq;

//   for (int i = 0; i < 4; i++) {
//     fc.special_lj[i] = force->special_lj[i];
//     fc.special_coul[i] = force->special_coul[i];
//     fc.special_coul[0] = 1.0;
//     fc.special_lj[0] = 1.0;
//   }

//   for (int i = 0; i < tp1; i++) {
//     for (int j = 0; j < tp1; j++) {
//       fc.c_force[i][j].cutsq = cutsq[i][j];
//       fc.c_force[i][j].cut_ljsq = cut_ljsq[i][j];
//       fc.c_force[i][j].lj1 = lj1[i][j];
//       fc.c_force[i][j].lj2 = lj2[i][j];
//       fc.c_energy[i][j].lj3 = lj3[i][j];
//       fc.c_energy[i][j].lj4 = lj4[i][j];
//       fc.c_energy[i][j].offset = offset[i][j];
//     }
//   }

//   if (ncoultablebits) {
//     for (int i = 0; i < ntable; i++) {
//       fc.table[i].r = rtable[i];
//       fc.table[i].dr = drtable[i];
//       fc.table[i].f = ftable[i];
//       fc.table[i].df = dftable[i];
//       fc.etable[i] = etable[i];
//       fc.detable[i] = detable[i];
//       fc.ctable[i] = ctable[i];
//       fc.dctable[i] = dctable[i];
//     }
//   }

// }

// template <class flt_t>
// void PairLJLongCoulLongIntel::ForceConst<flt_t>::set_ntypes
// (const int ntypes, const int ntable, Memory *memory) {
//   if ( (ntypes != _ntypes || ntable != _ntable) ) {
//     if (_ntypes > 0) {
//       _memory->destroy(c_force);
//       _memory->destroy(c_energy);
//       _memory->destroy(table);
//       _memory->destroy(etable);
//       _memory->destroy(detable);
//       _memory->destroy(ctable);
//       _memory->destroy(dctable);
//     }
//     if (ntypes > 0) {
//       memory->create(c_force,ntypes,ntypes,"fc.c_force");
//       memory->create(c_energy,ntypes,ntypes,"fc.c_energy");
//       memory->create(table,ntable,"pair:fc.table");
//       memory->create(etable,ntable,"pair:fc.etable");
//       memory->create(detable,ntable,"pair:fc.detable");
//       memory->create(ctable,ntable,"pair:fc.ctable");
//       memory->create(dctable,ntable,"pair:fc.dctable");
//     }
//   }
//   _ntypes=ntypes;
//   _ntable=ntable;
//   _memory=memory;
// }
