/***
 *  $Id$
 **
 *  File: graspi_types.hpp
 *  Created: May 8, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapthysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef GRASPI_TYPES_HPP
#define GRASPI_TYPES_HPP

#include <map>
#include <vector>
#include <boost/graph/adjacency_list.hpp>

#define COLOR unsigned int

#define BLACK   0 // donor     (D)
#define WHITE   1 // acceptor  (A)
#define GREY    3 // mixed     (D+A)

#define BLUE   10 // cathode   (neg/bottom)
#define RED    20 // anode     (pos/top)

#define GREEN  30 // interface (I D/A)
#define DGREEN 31 // interface (I D/D+A)
#define LGREEN 32 // interface (I A/D+A)

namespace graspi {

  // graph type
  typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS,
				boost::no_property, boost::no_property> graph_t;

  typedef boost::graph_traits<graph_t>::vertex_descriptor vertex_t;
  typedef boost::graph_traits<graph_t>::edge_descriptor   edge_descriptor_t;

  // vertex properties
  typedef std::vector<COLOR> vertex_colors_t;
  typedef std::vector<int>   vertex_ccs_t;

  // edge properties
  typedef std::map<std::pair<int,int> ,char >         edge_colors_t;
  typedef std::map<edge_descriptor_t, float>          edge_map_t;
  typedef boost::associative_property_map<edge_map_t> edge_weights_t;

  // cc type
  struct CC;
  typedef std::vector<CC> ccs_t;
    
  // descriptor type;
    typedef std::pair<float,std::string> desc_t;
    
  // descriptor dimension of graph vertices
  struct dim_g_t {
      unsigned int n_phases;           //number of phases (currently 2 or 3)
      unsigned int n_bulk;             //all BLACK,WHITE and GREY vertices
      unsigned int n_meta_basic;       //basic meta vertices (anode, cathode)
      unsigned int n_meta_interfacial; //interfacial meta vertices between
                                       // (BLACK/WHITE), (BLACK/GREY), (WHITE/GREY)
      explicit dim_g_t(unsigned int n_phases = 2,
		       unsigned int bulk = 0,
		       unsigned int meta = 2)
	  : n_phases(n_phases), n_bulk(bulk), n_meta_basic(2), n_meta_interfacial(1) {
	  if(n_phases == 3) n_meta_interfacial = 3;
      }

      int n_total()     const{ return n_bulk+n_meta_basic+n_meta_interfacial; }
      int n_meta()      const{ return n_meta_basic; }
      int n_meta_int()  const{ return n_meta_interfacial; }
      int n_meta_total()const{ return n_meta_basic+n_meta_interfacial; }

      int id(COLOR c)const{
	  if(c == BLUE)   return id_blue();
	  if(c == RED)    return id_red();
	  if(c == GREEN)  return id_green();
	  if(c == DGREEN)  return id_dgreen();
	  if(c == LGREEN)  return id_lgreen();
          return -1;
      }

      int id_blue() const{ return n_bulk;   }
      int id_red()  const{ return n_bulk+1; }
      int id_green()const{ return n_bulk+2; }
      int id_dgreen()const{ return n_bulk+3; }
      int id_lgreen()const{ return n_bulk+4; }
  }; // struct dim_g_t

  // dimension descriptor
  struct dim_a_t {
      explicit dim_a_t(unsigned int x = 1, unsigned int y = 1,
		       unsigned int z = 1)
	  : nx(x), ny(y), nz(z) { }

      unsigned int nx;
      unsigned int ny;
      unsigned int nz;

  }; // struct dim_t

} // namespace graspi

#endif // GRASPI_TYPES_HPP
