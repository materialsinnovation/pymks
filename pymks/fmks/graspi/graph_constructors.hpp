/***
 *  $Id$
 **
 *  File: graph_constructors.hpp
 *  Created: May 8, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapathysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef GRAPH_CONSTRUCTORS_HPP
#define GRAPH_CONSTRUCTORS_HPP

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "graspi_types.hpp"


namespace graspi {

  bool build_graph(std::istream& is,
		   graph_t*& G,
		   dim_g_t& d_g,
		   vertex_colors_t& C,
		   edge_weights_t& W,
		   edge_colors_t& L);

  bool build_graph(const std::string& name,
			  graph_t*& G,
			  dim_g_t& d_g,
			  vertex_colors_t& C,
			  edge_weights_t& W,
			  edge_colors_t& L) ;
// {
//       std::ifstream f(name.c_str());
//       if (!f) return false;
//       return build_graph(f, G, d_g, C, W, L);
//   } // build_graph


  template <typename Container>
  bool read_array(const std::string& name,
		  Container& M, dim_a_t& d_a, dim_g_t& d_g){
      std::ifstream f(name.c_str());
      if (!f) return false;
      std::string str;
      getline(f,str);
      std::istringstream iss(str);
      iss >> d_a.nx >> d_a.ny >> d_a.nz;
      if(d_a.nz == 0) d_a.nz = 1;
      d_g.n_bulk = d_a.nx * d_a.ny * d_a.nz;
      unsigned int n_total =  d_g.n_total();
      M.resize(n_total);
      for(unsigned int i=0; i< d_g.n_bulk; i++){
	  if (!f) return false;
	  f >> M[i];
      }
      f.close();
   return true;
  }

  bool build_graph(graph_t*& G, const dim_g_t& d_g,
		   vertex_colors_t& C, dim_a_t& d_a,
		   edge_weights_t& W,
		   edge_colors_t& L,
		   double pixelsize = 1.0, bool if_per_on_size = false);

  void initlize_colors_meta_vertices( vertex_colors_t& C, const dim_g_t& d_g );

  template <typename Container>
  bool update_graph(const Container& M, const dim_a_t& d,
		    graph_t& G, dim_g_t& d_g,
		    vertex_colors_t& C,
		    edge_weights_t& W,
		    edge_colors_t& L);

  void add_edge_to_graph(int s, int t, char o, double w,
			 int green_vertex, int dgreen_vertex, int lgreen_vertex,
			 graph_t* G,
			 vertex_colors_t& C,
			 edge_weights_t& W,
			 edge_colors_t& L);

  void make_update_edge_with_meta_vertex( int s, int meta_t,
					  double w, char o,
					  graph_t* G,
					  edge_weights_t& W,
					  edge_colors_t& L);

  int compute_pos_2D(int i_x, int i_y,
		     const dim_a_t& d_a,
		     bool if_per_on_sides = false );

  int compute_pos_3D(int i_x, int i_y, int i_z,
		     const dim_a_t& d_a,
		     bool if_per_on_sides = false );

  void generate_ngbr(int i, int j, int k,
		     const dim_a_t& d_a,
		     std::pair<int,char>* ngbr,
		     bool if_per_on_sides = false );

} // graspi


#endif // GRAPH_CONSTRUCTORS_HPP


