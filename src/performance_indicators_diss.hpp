/***
 *  $Id$
 **
 *  File: performance_indicators_diss.hpp
 *  Created: May 9, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapathysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef PERFORMANCE_INDICATORS_DISS_HPP
#define PERFORMANCE_INDICATORS_DISS_HPP

#include <climits>
#include <sstream>

#include "graspi_types.hpp"
#include "graph_dijkstra.hpp"
#include "graspi_predicates.hpp"
#include <boost/graph/filtered_graph.hpp>


namespace graspi {

  struct foo_w_diss{
      double A1,B1,C1;
      foo_w_diss(){ A1=6.265; B1=-23.0; C1=17.17; }
      double operator()(double d)const{
	  return A1*exp(-((d-B1)/C1)*((d-B1)/C1));
      }
  };

  inline int
  identify_n_vertices_within_distance( const std::vector<float>& d,
				       double Ld){
      int n_Ld = 0;
      for(unsigned int i = 0; i < d.size(); i++){
	  if( (d[i] < Ld) && (d[i] > 0) ) n_Ld++;
      }
      return n_Ld;
  }

  inline double
  identify_weighted_vertices_within_distance( const std::vector<float>& d,
					      double Ld){
      double wn_Ld = 0;
      foo_w_diss wfoo;

      for(unsigned int i = 0; i < d.size(); i++){
	  double d_i = d[i];
	  double wd_i = wfoo(d_i);
	  if( (d_i < Ld) && (d_i > 0) ) wn_Ld+= wd_i;
      }
      return wn_Ld;
  }

  inline std::pair<int,double>
  identify_n_weighted_vertices_within_distance( const std::vector<float>& d,
						double Ld){
      double wn_Ld = 0;
      int n_Ld = 0;
      foo_w_diss wfoo;

      for(unsigned int i = 0; i < d.size(); i++){
	  double d_i = d[i];
	  double wd_i = wfoo(d_i);
	  if( (d_i < Ld) && (d_i > 0) ){
	      wn_Ld+= wd_i;
	      n_Ld++;
	  }
      }
      return std::pair<int,double>(n_Ld,wn_Ld);
  }


  inline std::pair<double,double>
  wf_diss(
	  graph_t* G, const dim_g_t& d_g, const vertex_colors_t& C,
	  const edge_weights_t& W, const vertex_ccs_t& vCC,
	  const ccs_t& CC,
	  double Ld,
	  const std::string& filename_ColorToGreen,
	  const std::string& filename_WColorToGreen,
	  COLOR color = BLACK,
	  COLOR green = GREEN
	  ){
      int n_color = 0;
      int n_color_Ld = 0;
      double wn_color_Ld = 0;

      connect_color_green pred(*G,C,color,green);
      unsigned int n = boost::num_vertices(*G);
      vertex_t int_id = d_g.id(green);
      std::vector<float> d(n);

      determine_shortest_distances( G, W, int_id, pred, d);

      foo_w_diss wfoo;
      std::ostringstream oss_out_d;
      std::ostringstream oss_out_wd;
      for (unsigned int i = 0; i < d.size(); i++) {
	  unsigned int c = C[i];
	  if (c == color) n_color++;
	  if ( ( c == color )
	       && ( fabs(d[i]) < std::numeric_limits<float>::max() )
	       ) {
	      double d_i = d[i];
	      oss_out_d  << d_i       << std::endl;
	      oss_out_wd << d_i << " " << wfoo(d_i) << std::endl;
	  }
      }
      std::ofstream f_out(filename_ColorToGreen.c_str());
      std::string buffer = oss_out_d.str();
      int size = oss_out_d.str().size();
      f_out.write (buffer.c_str(),size);
      f_out.close();
      f_out.open(filename_WColorToGreen.c_str());
      buffer = oss_out_wd.str();
      size = oss_out_wd.str().size();
      f_out.write (buffer.c_str(),size);
      f_out.close();

      std::pair<int,double> pLd
	  = identify_n_weighted_vertices_within_distance(d,Ld);
      n_color_Ld = pLd.first;
      wn_color_Ld = pLd.second;

#ifdef DEBUG
      std::cout << "[DEBUG] Number of " << color << "vertices: "
		<< n_color << std::endl
		<< "[DEBUG] Number of " << color << "vertices in "
		<< Ld << " distance to green: "
		<< n_color_Ld << std::endl;
#endif

      return std::pair<double, double>(
				       (double)wn_color_Ld/n_color,
				       (double)n_color_Ld/n_color
				       );
  }


  inline std::pair<double,double>
  wf_diss(
	  graph_t* G, const dim_g_t& d_g, const vertex_colors_t& C,
	  const edge_weights_t& W, const vertex_ccs_t& vCC,
	  const ccs_t& CC,
	  double Ld,
	  COLOR color = BLACK,
	  COLOR green = GREEN
	  ){
      int n_color = 0;
      int n_color_Ld = 0;
      double wn_color_Ld = 0;

      connect_color_green pred(*G,C,color,green);
      unsigned int n = boost::num_vertices(*G);
      vertex_t int_id = d_g.id(green);
      std::vector<float> d(n);

      determine_shortest_distances( G, W, int_id, pred, d);

      foo_w_diss wfoo;
      for (unsigned int i = 0; i < d.size(); i++) {
	  unsigned int c = C[i];
	  if (c == color) n_color++;
      }

      std::pair<int,double> pLd
	  = identify_n_weighted_vertices_within_distance(d,Ld);
      n_color_Ld = pLd.first;
      wn_color_Ld = pLd.second;

      return std::pair<double, double>(
				       (double)wn_color_Ld/n_color,
				       (double)n_color_Ld/n_color
				       );
  }


}
#endif
