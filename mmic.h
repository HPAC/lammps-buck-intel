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

#include "immintrin.h"
#include "stdio.h"


namespace intrin {

  template<class flt_t>
  struct vector_op{};
#if defined(__MIC__)
  template<>
    struct vector_op<float> {
    typedef __m512i ivec;
    typedef __m512  fvec;
    typedef float   fval;
    typedef int     ival;
    typedef __mmask16 fmask;
    typedef __mmask16 imask;
    static const int VL = 16;
    static const int BASEMASK = 0xFFFF;  
    /*
      Integer functions (common)
    */

    static ivec load(const ival * buff){
      return _mm512_load_epi32(buff);
    }

    static ivec set1(const int val){
      return _mm512_set1_epi32(val);
    }

    static ivec mask_gather( const ivec v0, const imask mask, const ivec idx,  const ival *buff){
      return _mm512_mask_i32gather_epi32(v0, mask, idx, buff, sizeof(ival));
    }


    static void mask_store(int *buff, const imask mask, const ivec v){
      _mm512_mask_store_epi32(buff, mask, v);
    }

    static void mask_packstore(int *buff, const imask mask, const ivec v){
      _mm512_mask_packstorelo_epi32(buff, mask, v);
    }
       /* Arithmetic */

    static ivec rshift(const ivec v1, const ivec bits){
      return _mm512_srlv_epi32(v1,bits);
    }
 
    static ivec bitwiseand(const ivec v1, const ivec v2){
      return _mm512_and_epi32(v1, v2); 
    }
   
    static ivec add(const ivec v1, const ivec v2){
      return _mm512_add_epi32(v1, v2);
    }

    static ivec mul(const ivec v1, const ivec v2){
      return _mm512_mullo_epi32(v1, v2);
    }

      /* Logical */

    static imask cmplt(const ivec v1, const ivec v2){
      return _mm512_cmp_epi32_mask(v1, v2, _MM_CMPINT_LT);
    }

    static imask cmpgt(const ivec v1, const ivec v2){
      return _mm512_cmp_epi32_mask(v1, v2, _MM_CMPINT_GT);
    }

    /*
      Floating Point functions
    */

    static fvec cast(const fvec v){
      return v;
    }

    static fvec load(const fval * buff){
      return _mm512_load_ps(buff);
    }

    static fvec load1(const fval * buff){
      return _mm512_extload_ps(buff, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
    }

    static fvec load4(const fval * buff){
      return _mm512_extload_ps(buff, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE);
    }

    static fvec mask_load4(const fvec base, const fmask mask, const fval * buff){
      return _mm512_mask_extload_ps(base, mask, buff, _MM_UPCONV_PS_NONE, 
                                    _MM_BROADCAST_4X16, _MM_HINT_NONE);
    }
    
    static fvec mask_mov(const fvec src, const fmask mask, const fvec v){
      return _mm512_mask_mov_ps(src, mask, v);
    }
    

    static void mask_store(fval *buff, const fmask mask, const fvec v){
      _mm512_mask_store_ps(buff, mask, v);
    }

    static void mask_packstore(fval *buff, const fmask mask, const fvec v){
      _mm512_mask_packstorelo_ps(buff, mask, v);
    }

    static fvec setzero(){
      return _mm512_setzero_ps();
    }

    static fvec set1(const fval val){
      return _mm512_set1_ps(val);
    }

    static fvec logather(const ivec idx, const fval * buffer){
      return _mm512_i32gather_ps(idx, buffer, sizeof(fval));
    }

    static fvec mask_logather(const fvec v0, const fmask mask, const ivec idx, const fval *buffer){
      return _mm512_mask_i32gather_ps(v0, mask, idx, buffer, sizeof(fval));
    }

    static ivec cast_int(const fvec v){
      return _mm512_castps_si512(v);
    }
    
    static ivec cast_int2(const fvec v){
      return cast_int(v);
    }


    static fvec castlo(const fvec v){
      return v;
    }

    static int loscatter(const fval *buffer, ivec idx, fvec v){
      return 0;
    }

      /* Arithmetic */

    static fvec add(const fvec v1, const fvec v2){
      return _mm512_add_ps(v1, v2); // v1 + v2
    }

    static fvec mask_add(const fvec v0, const fmask mask, const fvec v1, const fvec v2){
      return _mm512_mask_add_ps(v0, mask, v1, v2); // v1 + v2 // v0 if no mask
    }

    static fvec sub(const fvec v1, const fvec v2){
      return _mm512_sub_ps(v1, v2); // v1 - v2
    }

    static fvec mask_sub(const fvec v0, const fmask mask, const fvec v1, const fvec v2){
      return _mm512_mask_sub_ps(v0, mask, v1, v2); // v1 - v2 // v0 if no mask
    }

    static fvec mul(const fvec v1, const fvec v2){
      return _mm512_mul_ps(v1, v2); // v1 .* v2
    }

    static fvec fmadd(const fvec v1, const fvec v2, const fvec v3){
      return _mm512_fmadd_ps(v1, v2, v3); // v1 .* v2 + v3
    }

    static fvec fnmadd(const fvec v1, const fvec v2, const fvec v3){
      return _mm512_fnmadd_ps(v1, v2, v3); // - v1 .* v2 + v3
    }

    static fvec mask3_fnmadd(const fvec v1, const fvec v2, const fvec v3, const fmask mask){
      return _mm512_mask3_fmadd_ps(v1, v2, v3, mask); // - v1 .* v2 + v3 // v3 if no mask
    }

    static fvec mask_mul(const fvec v0, const fmask mask, const fvec v1, const fvec v2){
      return _mm512_mask_mul_ps(v0, mask, v1, v2); // v1 .* v2  // v0 if no mask
    }

    static fvec div(const fvec v1, const fvec v2){
      return _mm512_div_ps(v1, v2); // v1 ./ v2
    }

    static fvec sqrt(const fvec v){
      return _mm512_sqrt_ps(v);
    }

    static fvec exp(const fvec v){
      return _mm512_exp_ps(v);
    }

    static fvec recip(const fvec v){
      return _mm512_recip_ps(v);
    }

    static fval mask_reduce_add(const fmask mask, const fvec v){
      return _mm512_mask_reduce_add_ps(mask, v);
    }


    static fvec mask_accu(const fvec accu, const fmask mask, const fvec v){
      return mask_add(accu, mask, accu, v);
    }


      /* Logical */

    static fmask cmplt(const fvec v1, const fvec v2){
      return _mm512_cmp_ps_mask(v1, v2, _MM_CMPINT_LT);
    }

    static fmask cmple(const fvec v1, const fvec v2){
      return _mm512_cmp_ps_mask(v1, v2, _MM_CMPINT_LE);
    }

    static fmask cmpgt(const fvec v1, const fvec v2){
      return _mm512_cmp_ps_mask(v1, v2, _MM_CMPINT_GT);
    }


    struct fvec4{
    fvec4(): x(setzero()), y(setzero()), z(setzero()), w(setzero()){}
      fvec x, y, z, w;
    };
    
     /* Vector Routines */

    static fvec rsq(fvec4 r){
      fvec ans = mul(r.x, r.x);
      ans = fmadd(r.y, r.y, ans);
      ans = fmadd(r.z, r.z, ans);
      return ans;      
    }

    static fvec4 swizzleload( fval *buff, const ival *jidx, const ival mymask){
      fvec tmpj04 =  load4( buff + 4 * jidx[0] );
      fvec tmpj15 =  load4( buff + 4 * jidx[1] );
      fvec tmpj26 =  load4( buff + 4 * jidx[2] );
      fvec tmpj37 =  load4( buff + 4 * jidx[3] );

      if (mymask > 0x000F){
        tmpj04 =  mask_load4(tmpj04, 0x00F0, buff + 4 * jidx[4] );
        tmpj15 =  mask_load4(tmpj15, 0x00F0, buff + 4 * jidx[5] );
        tmpj26 =  mask_load4(tmpj26, 0x00F0, buff + 4 * jidx[6] );
        tmpj37 =  mask_load4(tmpj37, 0x00F0, buff + 4 * jidx[7] );
        if (mymask > 0x00FF){
          tmpj04 =  mask_load4(tmpj04, 0x0F00, buff + 4 * jidx[8] );
          tmpj15 =  mask_load4(tmpj15, 0x0F00, buff + 4 * jidx[9] );
          tmpj26 =  mask_load4(tmpj26, 0x0F00, buff + 4 * jidx[10] );
          tmpj37 =  mask_load4(tmpj37, 0x0F00, buff + 4 * jidx[11] );
          if (mymask > 0x0FFF){
            tmpj04 =  mask_load4(tmpj04, 0xF000, buff + 4 * jidx[12] );
            tmpj15 =  mask_load4(tmpj15, 0xF000, buff + 4 * jidx[13] );
            tmpj26 =  mask_load4(tmpj26, 0xF000, buff + 4 * jidx[14] );
            tmpj37 =  mask_load4(tmpj37, 0xF000, buff + 4 * jidx[15] );
          }
        }
      }

      fvec xz01 = _mm512_mask_swizzle_ps(tmpj04, 0xAAAA, tmpj15, _MM_SWIZ_REG_CDAB);
      fvec xz23 = _mm512_mask_swizzle_ps(tmpj26, 0xAAAA, tmpj37, _MM_SWIZ_REG_CDAB);
      fvec yw01 = _mm512_mask_swizzle_ps(tmpj15, 0x5555, tmpj04, _MM_SWIZ_REG_CDAB);
      fvec yw23 = _mm512_mask_swizzle_ps(tmpj37, 0x5555, tmpj26, _MM_SWIZ_REG_CDAB);

      fvec4 pos;

      pos.x = _mm512_mask_swizzle_ps(xz01, 0xCCCC, xz23, _MM_SWIZ_REG_BADC);
      pos.y = _mm512_mask_swizzle_ps(yw01, 0xCCCC, yw23, _MM_SWIZ_REG_BADC);
      pos.z = _mm512_mask_swizzle_ps(xz23, 0x3333, xz01, _MM_SWIZ_REG_BADC);
      pos.w = _mm512_mask_swizzle_ps(yw23, 0x3333, yw01, _MM_SWIZ_REG_BADC);

      return pos;
    }

    static int swizzlesubstore( fval * buff, const fvec4 fmic, const ival *jidx, const fmask mymask){
      fvec xz01 = _mm512_mask_swizzle_ps(fmic.x, 0xCCCC, fmic.z, _MM_SWIZ_REG_BADC);
      fvec yw01 = _mm512_mask_swizzle_ps(fmic.y, 0xCCCC, fmic.w, _MM_SWIZ_REG_BADC);
      fvec xz23 = _mm512_mask_swizzle_ps(fmic.z, 0x3333, fmic.x, _MM_SWIZ_REG_BADC);
      fvec yw23 = _mm512_mask_swizzle_ps(fmic.w, 0x3333, fmic.y, _MM_SWIZ_REG_BADC);
              
      fvec tmpj04 = _mm512_mask_swizzle_ps(xz01, 0xAAAA, yw01, _MM_SWIZ_REG_CDAB);
      fvec tmpj15 = _mm512_mask_swizzle_ps(yw01, 0x5555, xz01, _MM_SWIZ_REG_CDAB);
      fvec tmpj26 = _mm512_mask_swizzle_ps(xz23, 0xAAAA, yw23, _MM_SWIZ_REG_CDAB);
      fvec tmpj37 = _mm512_mask_swizzle_ps(yw23, 0x5555, xz23, _MM_SWIZ_REG_CDAB);
      // Load the current force values
      fvec fj04 =  load4( buff + 4 * jidx[0]);
      fvec fj15 =  load4( buff + 4 * jidx[1]);
      fvec fj26 =  load4( buff + 4 * jidx[2]);              
      fvec fj37 =  load4( buff + 4 * jidx[3]);              
      
      if (mymask > 0x000F){
        fj04 =  mask_load4(fj04, 0x00F0, buff + 4 * jidx[4]);
        fj15 =  mask_load4(fj15, 0x00F0, buff + 4 * jidx[5]);                
        fj26 =  mask_load4(fj26, 0x00F0, buff + 4 * jidx[6]);                
        fj37 =  mask_load4(fj37, 0x00F0, buff + 4 * jidx[7]);
        if (mymask > 0x00FF){
          fj04 =  mask_load4(fj04, 0x0F00, buff + 4 * jidx[8] );
          fj15 =  mask_load4(fj15, 0x0F00, buff + 4 * jidx[9] );
          fj26 =  mask_load4(fj26, 0x0F00, buff + 4 * jidx[10] );
          fj37 =  mask_load4(fj37, 0x0F00, buff + 4 * jidx[11] );
          if (mymask > 0x0FFF){
            fj04 =  mask_load4(fj04, 0xF000, buff + 4 * jidx[12] );
            fj15 =  mask_load4(fj15, 0xF000, buff + 4 * jidx[13] );
            fj26 =  mask_load4(fj26, 0xF000, buff + 4 * jidx[14] );
            fj37 =  mask_load4(fj37, 0xF000, buff + 4 * jidx[15] );
          }
        }
      }  
     
              
      // Subtract the new values
      fj04 = sub(fj04,tmpj04);
      fj15 = sub(fj15,tmpj15);
      fj26 = sub(fj26,tmpj26);
      fj37 = sub(fj37,tmpj37);
      
      // Store new Vector
      mask_packstore(buff + 4 * jidx[0], 0x000F, fj04);
      mask_packstore(buff + 4 * jidx[1], 0x000F, fj15);
      mask_packstore(buff + 4 * jidx[2], 0x000F, fj26);
      mask_packstore(buff + 4 * jidx[3], 0x000F, fj37);
      if (mymask > 0x000F){
        mask_packstore(buff + 4 * jidx[4], 0x00F0, fj04);
        mask_packstore(buff + 4 * jidx[5], 0x00F0, fj15);
        mask_packstore(buff + 4 * jidx[6], 0x00F0, fj26);
        mask_packstore(buff + 4 * jidx[7], 0x00F0, fj37);
        if (mymask > 0x00FF){
          mask_packstore(buff + 4 * jidx[8], 0x0F00, fj04);
          mask_packstore(buff + 4 * jidx[9], 0x0F00, fj15);
          mask_packstore(buff + 4 * jidx[10], 0x0F00, fj26);
          mask_packstore(buff + 4 * jidx[11], 0x0F00, fj37);
          if (mymask > 0x0FFF){
            mask_packstore(buff + 4 * jidx[12], 0xF000, fj04);
            mask_packstore(buff + 4 * jidx[13], 0xF000, fj15);
            mask_packstore(buff + 4 * jidx[14], 0xF000, fj26);
            mask_packstore(buff + 4 * jidx[15], 0xF000, fj37);
          }
        }

      }
      return 0; //Or error
    }


    static void print(const ivec v){
      __declspec(align(64)) int imem[16];
      _mm512_store_epi32((void *)imem, v);
      for (int i=0; i<VL; i++)
        printf("%d ",imem[i]);
      printf("\n");
    }
    static void print(const fvec v){
      __declspec(align(64)) fval imem[VL];
      mask_store(imem, BASEMASK, v);
      for (int i=0; i<VL; i++)
        printf("%f ",imem[i]);
      printf("\n");
    }


  };


  template<>
    struct vector_op<double> {
    typedef __m512i ivec;
    typedef __m512d fvec;
    typedef double  fval;
    typedef int     ival;
    typedef __mmask8 fmask;
    typedef __mmask16 imask;
    static const int VL = 8;
    static const int BASEMASK = 0xFF;
      

    /*
      Integer functions (common)
    */

    static ivec load(const ival * buff){
      return _mm512_load_epi32(buff);
    }

    static ivec set1(const int val){
      return _mm512_set1_epi32(val);
    }

    static ivec mask_gather( const ivec v0, const imask mask, const ivec idx,  const ival *buff){
      return _mm512_mask_i32gather_epi32(v0, mask, idx, buff, sizeof(ival));
    }

    static void mask_store(ival *buff, const imask mask, const ivec v){
      _mm512_mask_store_epi32(buff, mask, v);
    }

    static void mask_packstore(ival *buff, const imask mask, const ivec v){
      _mm512_mask_packstorelo_epi32(buff, mask, v);
    }
       /* Arithmetic */

    static ivec rshift(const ivec v1, const ivec bits){
      return _mm512_srlv_epi32(v1,bits);
    }
 
    static ivec bitwiseand(const ivec v1, const ivec v2){
      return _mm512_and_epi32(v1, v2); 
    }
   
    static ivec add(const ivec v1, const ivec v2){
      return _mm512_add_epi32(v1, v2);
    }

    static ivec mul(const ivec v1, const ivec v2){
      return _mm512_mullo_epi32(v1, v2);
    }

      /* Logical */

    static imask cmplt(const ivec v1, const ivec v2){
      return _mm512_cmp_epi32_mask(v1, v2, _MM_CMPINT_LT);
    }

    static imask cmpgt(const ivec v1, const ivec v2){
      return _mm512_cmp_epi32_mask(v1, v2, _MM_CMPINT_GT);
    }

    /*
      Floating Point functions
    */

    static fvec cast(const fvec v){
      return v;
    }

    static fvec cast(const __m512 v){
      return _mm512_castps_pd(v);
    }


    static fvec load(const fval * buff){
      return _mm512_load_pd(buff);
    }


    static fvec load1(const fval * buff){
      return _mm512_extload_pd(buff, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    }

    static fvec load4(const fval * buff){
      return _mm512_extload_pd(buff, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, _MM_HINT_NONE);
    }

    static fvec mask_load4(const fvec base, const fmask mask, const fval * buff){
      return _mm512_mask_extload_pd(base, mask, buff, _MM_UPCONV_PD_NONE, 
                                    _MM_BROADCAST_4X8, _MM_HINT_NONE);
    }

    static fvec mask_mov(const fvec src, const fmask mask, const fvec v){
      return _mm512_mask_mov_pd(src, mask, v);
    }

    static void mask_store(fval *buff, const fmask mask, const fvec v){
      _mm512_mask_store_pd(buff, mask, v);
    }

    static void mask_packstore(fval *buff, const fmask mask, const fvec v){
      _mm512_mask_packstorelo_pd(buff, mask, v);
    }

    static fvec setzero(){
      return _mm512_setzero_pd();
    }

    static fvec set1(const fval val){
      return _mm512_set1_pd(val);
    }

    static fvec logather(const ivec idx, const fval * buffer){
      return _mm512_i32logather_pd(idx, buffer, sizeof(fval));
    }

    static fvec mask_logather(const fvec v0, const fmask mask, const ivec idx, const fval *buffer){
      return _mm512_mask_i32logather_pd(v0, mask, idx, buffer, sizeof(fval));
    }

    static ivec cast_int(const fvec v){
      __declspec(align(64)) int vtmp[16];
      mask_packstore(vtmp, 0x5555, _mm512_castps_si512(_mm512_castpd_ps(v)));
      ivec ans = load(vtmp);
      return ans;
    }

    static ivec cast_int2(const fvec v){
      return _mm512_castps_si512(_mm512_cvtpd_pslo(v));
    }

    static fvec castlo(const vector_op<float>::fvec v){
      return _mm512_cvtpslo_pd(v);

    }

    static fvec castlo(fvec v){
      return v;

    }


    static fvec casthi(const vector_op<float>::fvec v){
      __m512 tmp =_mm512_mask_permute4f128_ps(v, 0xFFFF, v, _MM_PERM_BADC );
      return _mm512_cvtpslo_pd(tmp);
    }

    static fvec mask_accu(const fvec accu, const fmask mask, const fvec v){
      return mask_add(accu, mask, accu, v);
    }

    static fvec mask_accu(const fvec accu, const imask mask, const __m512 v){
      const fmask maskhi = mask >> 8;
      const fmask masklo = 0xFF & mask;
      fvec vlo = castlo(v);
      fvec vhi = casthi(v);
      fvec ans = mask_add(accu, maskhi, accu, vhi);
      ans = mask_add(ans, masklo, ans, vlo);
      return ans;
    }

    static int loscatter(fval *buffer, const ivec idx, fvec v){
      _mm512_i32loscatter_pd(buffer, idx, v, sizeof(fval));
      return 0;
    }


      /* Arithmetic */

    static fvec add(const fvec v1, const fvec v2){
      return _mm512_add_pd(v1, v2); // v1 + v2
    }

    static fvec mask_add(const fvec v0, const fmask mask, const fvec v1, const fvec v2){
      return _mm512_mask_add_pd(v0, mask, v1, v2); // v1 + v2 // v0 if no mask
    }

    static fvec sub(const fvec v1, const fvec v2){
      return _mm512_sub_pd(v1, v2); // v1 - v2
    }

    static fvec mask_sub(const fvec v0, const fmask mask, const fvec v1, const fvec v2){
      return _mm512_mask_sub_pd(v0, mask, v1, v2); // v1 - v2 // v0 if no mask
    }

    static fvec mul(const fvec v1, const fvec v2){
      return _mm512_mul_pd(v1, v2); // v1 .* v2
    }

    static fvec fmadd(const fvec v1, const fvec v2, const fvec v3){
      return _mm512_fmadd_pd(v1, v2, v3); // v1 .* v2 + v3
    }

    static fvec fnmadd(const fvec v1, const fvec v2, const fvec v3){
      return _mm512_fnmadd_pd(v1, v2, v3); // - v1 .* v2 + v3
    }

    static fvec mask3_fnmadd(const fvec v1, const fvec v2, const fvec v3, const fmask mask){
      return _mm512_mask3_fmadd_pd(v1, v2, v3, mask); // - v1 .* v2 + v3 // v3 if no mask
    }

    static fvec mask_mul(const fvec v0, const fmask mask, const fvec v1, const fvec v2){
      return _mm512_mask_mul_pd(v0, mask, v1, v2); // v1 .* v2  // v0 if no mask
    }

    static fvec div(const fvec v1, const fvec v2){
      return _mm512_div_pd(v1, v2); // v1 ./ v2
    }

    static fvec sqrt(const fvec v){
      return _mm512_sqrt_pd(v);
    }

    static fvec exp(const fvec v){
      return _mm512_exp_pd(v);
    }

    static fvec recip(const fvec v){
      return _mm512_recip_pd(v);
    }

    static fval mask_reduce_add(const fmask mask, const fvec v){
      return _mm512_mask_reduce_add_pd(mask, v);
    }

      /* Logical */

    static fmask cmplt(const fvec v1, const fvec v2){
      return _mm512_cmp_pd_mask(v1, v2, _MM_CMPINT_LT);
    }

    static fmask cmple(const fvec v1, const fvec v2){
      return _mm512_cmp_pd_mask(v1, v2, _MM_CMPINT_LE);
    }


    static fmask cmpgt(const fvec v1, const fvec v2){
      return _mm512_cmp_pd_mask(v1, v2, _MM_CMPINT_GT);
    }


    struct fvec4{
    fvec4(): x(setzero()), y(setzero()), z(setzero()), w(setzero()){}
      fvec x, y, z, w;
    };
    
     /* Vector Routines */

    static fvec rsq(const fvec4 r){
      fvec ans = mul(r.x, r.x);
      ans = fmadd(r.y, r.y, ans);
      ans = fmadd(r.z, r.z, ans);
      return ans;      
    }


    static fvec4 swizzleload( const fval *buff, const ival *jidx, const ival mymask){
      fvec tmpj04 =  load4( buff + 4 * jidx[0] );
      fvec tmpj15 =  load4( buff + 4 * jidx[1] );
      fvec tmpj26 =  load4( buff + 4 * jidx[2] );
      fvec tmpj37 =  load4( buff + 4 * jidx[3] );

      if (mymask > 0xF){
        tmpj04 =  mask_load4(tmpj04, 0xF0, buff + 4 * jidx[4] );
        tmpj15 =  mask_load4(tmpj15, 0xF0, buff + 4 * jidx[5] );
        tmpj26 =  mask_load4(tmpj26, 0xF0, buff + 4 * jidx[6] );
        tmpj37 =  mask_load4(tmpj37, 0xF0, buff + 4 * jidx[7] );
      }

      fvec xz01 = _mm512_mask_swizzle_pd(tmpj04, 0xAA, tmpj15, _MM_SWIZ_REG_CDAB);
      fvec xz23 = _mm512_mask_swizzle_pd(tmpj26, 0xAA, tmpj37, _MM_SWIZ_REG_CDAB);
      fvec yw01 = _mm512_mask_swizzle_pd(tmpj15, 0x55, tmpj04, _MM_SWIZ_REG_CDAB);
      fvec yw23 = _mm512_mask_swizzle_pd(tmpj37, 0x55, tmpj26, _MM_SWIZ_REG_CDAB);

      fvec4 pos;

      pos.x = _mm512_mask_swizzle_pd(xz01, 0xCC, xz23, _MM_SWIZ_REG_BADC);
      pos.y = _mm512_mask_swizzle_pd(yw01, 0xCC, yw23, _MM_SWIZ_REG_BADC);
      pos.z = _mm512_mask_swizzle_pd(xz23, 0x33, xz01, _MM_SWIZ_REG_BADC);
      pos.w = _mm512_mask_swizzle_pd(yw23, 0x33, yw01, _MM_SWIZ_REG_BADC);

      return pos;
    }



    static int swizzlesubstore( fval * buff, fvec4 fmic, const ival *jidx, const fmask mymask){
      fvec xz01 = _mm512_mask_swizzle_pd(fmic.x, 0xCC, fmic.z, _MM_SWIZ_REG_BADC);
      fvec yw01 = _mm512_mask_swizzle_pd(fmic.y, 0xCC, fmic.w, _MM_SWIZ_REG_BADC);
      fvec xz23 = _mm512_mask_swizzle_pd(fmic.z, 0x33, fmic.x, _MM_SWIZ_REG_BADC);
      fvec yw23 = _mm512_mask_swizzle_pd(fmic.w, 0x33, fmic.y, _MM_SWIZ_REG_BADC);
              
      fvec tmpj04 = _mm512_mask_swizzle_pd(xz01, 0xAA, yw01, _MM_SWIZ_REG_CDAB);
      fvec tmpj15 = _mm512_mask_swizzle_pd(yw01, 0x55, xz01, _MM_SWIZ_REG_CDAB);
      fvec tmpj26 = _mm512_mask_swizzle_pd(xz23, 0xAA, yw23, _MM_SWIZ_REG_CDAB);
      fvec tmpj37 = _mm512_mask_swizzle_pd(yw23, 0x55, xz23, _MM_SWIZ_REG_CDAB);
      // Load the current force values
      fvec fj04 =  load4( buff + 4 * jidx[0]);
      fvec fj15 =  load4( buff + 4 * jidx[1]);
      fvec fj26 =  load4( buff + 4 * jidx[2]);              
      fvec fj37 =  load4( buff + 4 * jidx[3]);              

      if (mymask > 0x0F){
        fj04 =  mask_load4(fj04, 0xF0, buff + 4 * jidx[4]);
        fj15 =  mask_load4(fj15, 0xF0, buff + 4 * jidx[5]);                
        fj26 =  mask_load4(fj26, 0xF0, buff + 4 * jidx[6]);                
        fj37 =  mask_load4(fj37, 0xF0, buff + 4 * jidx[7]);
      }
              
      // Subtract the new values
      fj04 = sub(fj04,tmpj04);
      fj15 = sub(fj15,tmpj15);
      fj26 = sub(fj26,tmpj26);
      fj37 = sub(fj37,tmpj37);
      
      // Store new Vector
      mask_packstore(buff + 4 * jidx[0], 0x0F, fj04);
      mask_packstore(buff + 4 * jidx[1], 0x0F, fj15);
      mask_packstore(buff + 4 * jidx[2], 0x0F, fj26);
      mask_packstore(buff + 4 * jidx[3], 0x0F, fj37);
      if (mymask > 0xF){
        mask_packstore(buff + 4 * jidx[4], 0xF0, fj04);
        mask_packstore(buff + 4 * jidx[5], 0xF0, fj15);
        mask_packstore(buff + 4 * jidx[6], 0xF0, fj26);
        mask_packstore(buff + 4 * jidx[7], 0xF0, fj37);
      }
      return 0; //Or error
    }

    static int swizzlesubstore( fval * buff, vector_op<float>::fvec4 fmic, const ival *jidx, 
                                const imask mymask){

      fmask maskhi = mymask >> 8;
      fmask masklo = 0xFF & mymask;
      fvec4 ftmp;
      ftmp.x = castlo(fmic.x);
      ftmp.y = castlo(fmic.y);
      ftmp.z = castlo(fmic.z);
      ftmp.w = castlo(fmic.w);
      swizzlesubstore( buff, ftmp, jidx, masklo);

      if (maskhi){
        ftmp.x = casthi(fmic.x);
        ftmp.y = casthi(fmic.y);
        ftmp.z = casthi(fmic.z);
        ftmp.w = casthi(fmic.w);
        swizzlesubstore( buff, ftmp, jidx + 8, maskhi);
      }

      return 0;
    }

      
    static void print(const ivec v){
      __declspec(align(64)) int imem[16];
      _mm512_store_epi32((void *)imem, v);
      for (int i=0; i<VL; i++)
        printf("%d ",imem[i]);
      printf("\n");
    }

    static void print(const fvec v){
      __declspec(align(64)) fval imem[VL];
      mask_store(imem, BASEMASK, v);
      for (int i=0; i<VL; i++)
        printf("%f ",imem[i]);
      printf("\n");
    }


  };



#endif //__MIC__

} //NAmespace

