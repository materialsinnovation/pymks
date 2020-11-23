/***
 *  $Id$
 **
 *  File: performance_indicators_abs.hpp
 *  Created: May 9, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapathysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef PERFORMANCE_INDICATORS_ABS_HPP
#define PERFORMANCE_INDICATORS_ABS_HPP

#include "graspi_types.hpp" 

#include <cmath>

namespace graspi {

  struct foo_w_abs{
      double Lexp;
      foo_w_abs(double L = 100):Lexp(L){ }
      double operator()(double d)const{ return exp(-1.0*d/Lexp); }
  };
  struct foo_no_w_abs{
      foo_no_w_abs(){ }
      double operator()(double d)const{ return 1.0; }
  };

  template<typename WFoo>
  inline double wf_abs(const vertex_colors_t& C, const dim_a_t& d_a, WFoo wf,
		       double pixelsize){
      double w_abs = 0;
      unsigned int total_n = d_a.nx * d_a.ny;

      if( (d_a.nz == 0) || (d_a.nz == 1) ){//2D
	  for(unsigned int j = 0; j < d_a.ny; j++){
	      for(unsigned int i = 0; i < d_a.nx; i++){
		  int id = i + d_a.nx * j;
		  if(C[id] == BLACK){
		      double h_diff = (double)(d_a.ny-j)*pixelsize;
		      w_abs += wf(h_diff);
		  }//BLACK
	      }//i
	  }//j
      }else{//3D
	  total_n *= d_a.nz;;
	  for(unsigned int k = 0; k < d_a.nz; k++){
	      for(unsigned int j = 0; j < d_a.ny; j++){
		  for(unsigned int i = 0; i < d_a.nx; i++){
		      unsigned int id = i + d_a.nx * ( j + d_a.ny * k);
		      if(C[id] == BLACK){
			  double h_diff = (double)(d_a.nz-k)*pixelsize;
			  w_abs += wf(h_diff);
		      }//BLACK
		  }//i
	      }//j
	  }//k
      }//end-3D

      return w_abs/total_n;
  }

} // namespace graspi

#endif // PERFORMANCE_INDICATORS_HPP
