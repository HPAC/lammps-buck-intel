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

#ifdef PAIR_CLASS

PairStyle(buck/long/coul/long/intel,PairBuckLongCoulLongIntel)

#else

#ifndef LMP_PAIR_BUCK_LONG_COUL_LONG_INTEL_H
#define LMP_PAIR_BUCK_LONG_COUL_LONG_INTEL_H

#include "pair_buck_long_coul_long.h"
#include "fix_intel.h"

namespace LAMMPS_NS {

class PairBuckLongCoulLongIntel : public PairBuckLongCoulLong {

 public:
  PairBuckLongCoulLongIntel(class LAMMPS *);
  virtual ~PairBuckLongCoulLongIntel();
  

  virtual void compute(int, int);
  void init_style();

  // typedef struct { float x,y,z; intw } sng4_t;

 private:
  FixIntel *fix;
  
  #ifdef _LMP_INTEL_OFFLOAD
  int _use_base;
  #endif


  template <class flt_t>
  class ForceConst;
  
  template <class flt_t, class acc_t>
  void compute(int eflag, int vflag, IntelBuffers<flt_t,acc_t> *buffers,
	       const ForceConst<flt_t> &fc);

  template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
	    const int, const int,
            class flt_t, class acc_t>
  void eval(int vflag, IntelBuffers<flt_t, acc_t> *buffers, 
	    const ForceConst<flt_t> &fc);

  template <class flt_t, class acc_t>
  void pack_force_const(ForceConst<flt_t> &fc, 
			IntelBuffers<flt_t, acc_t> *buffers);

  template <class flt_t>
  class ForceConst {
   public:
    typedef struct { flt_t cutsq, cut_ljsq, buck1, buck2; } c_force_t;
    typedef struct { flt_t a, c, offset, pad; } c_energy_t;
    typedef struct { flt_t r, dr, f, df; } table_t;
    typedef struct { flt_t e, de, edisp, dedisp;} etable_t;
    _alignvar(flt_t special_coul[4],64);
    _alignvar(flt_t special_lj[4],64);
    flt_t g_ewald, tabinnersq, **rho_inv;
    c_force_t **c_force;
    c_energy_t **c_energy;
    table_t *table;
    table_t *disptable;
    etable_t *etable;
    table_t *ctable;


    ForceConst() : _ntypes(0), _ntable(0) {}
    ~ForceConst() { set_ntypes(0,0,NULL); }
        
    void set_ntypes(const int ntypes, const int ntable, Memory *memory);

   private:
    int _ntypes, _ntable;
    Memory *_memory;
  }; //end class ForceConst

  ForceConst<float> force_const_single;
  ForceConst<double> force_const_double;
}; // end class PairBuckLongCoulLongIntel
};// end LAMMPS_NS
#endif
#endif



