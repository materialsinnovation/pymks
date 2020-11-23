/***
 *  $Id$
 **
 *  File: graph_cc.hpp
 *  Created: May 9, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapthysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef GRAPH_CC_HPP
#define GRAPH_CC_HPP

#include "graspi_types.hpp"
#include "graspi_predicates.hpp"

#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>

namespace graspi{

  inline void identify_connected_components(
					    graspi::graph_t* G,
					    const vertex_colors_t& C,
					    vertex_ccs_t& vCC ){

      connect_same_color pred(*G, C);
      boost::filtered_graph<graspi::graph_t,connect_same_color> FG(*G, pred);

      int size_of_G = boost::num_vertices(*G);
      vCC.resize(size_of_G, 0);

      boost::connected_components(FG, &vCC[0]);
  }//identify_connected_components

  inline int determine_number_of_CCs(
				     const graspi::vertex_ccs_t& vCC,
				     const graspi::dim_g_t& d_g){

      int max_index = 0;
      // find max index
      for (unsigned int i = 0; i < vCC.size(); ++i) {
		  std::cout << "id,cc: " << i << " " << vCC[i] << std::endl;
	  if(vCC[i] > max_index)
	      max_index = vCC[i];
      }
      int number_of_conn_comp =  max_index + 1;
      // correct nOfCC - subtract meta vertices that are identified as separate CC
      return number_of_conn_comp - d_g.n_meta_total();
  }//determine_number_of_CCs



  inline void
  determine_basic_stats_of_ccs( graspi::graph_t* G,
				const graspi::dim_g_t& d_g,
				graspi::ccs_t& CCs,
				const graspi::vertex_colors_t& C,
				const graspi::vertex_ccs_t&   vCCs ){

      std::vector<int>::const_iterator it = max_element( vCCs.begin(),
							 vCCs.end() );
      CCs.resize( (*it)+1 );

#ifdef DEBUG
      std::cout << "Total number of connected components" << *it << std::endl;
#endif
      for (unsigned int  i = 0; i < vCCs.size(); i++){
	  CCs[vCCs[i]].color = C[i];
	  CCs[vCCs[i]].size++;
      }

      unsigned int size_of_G = boost::num_vertices(*G);

      // determine all vertices connected to the bottom electrode
      std::set<int> comp_conn_to_electrode;
      graspi::vertex_t source = d_g.id_blue(); // bottom electrode index
      graspi::graph_t::adjacency_iterator u, u_end;
      boost::tie(u, u_end) = boost::adjacent_vertices(source, *G);
      for (; u != u_end; ++u) {
	  comp_conn_to_electrode.insert(vCCs[*u]);
      }

#ifdef DEBUG
      std::cout << "Id of connected components connected to bottom " << std::endl;
      copy(comp_conn_to_electrode.begin(), comp_conn_to_electrode.end(),
	   std::ostream_iterator<int>(std::cout, " "));
      std::cout << std::endl;
#endif

      // pass info about connectivity to bottom electrode to data of components
      for(unsigned int  i = 0; i < CCs.size(); i++ ){
	  if( comp_conn_to_electrode.find(i) != comp_conn_to_electrode.end() )
	      CCs[i].if_connected_to_electrode += 1;
      }

      // determine all vertices connected to the top electrode
      comp_conn_to_electrode.clear();
      source = d_g.id_red(); // top electrode index
      boost::tie(u, u_end) = boost::adjacent_vertices(source, *G);
      for (; u != u_end; ++u) {
	  comp_conn_to_electrode.insert(vCCs[*u]);
      }

#ifdef DEBUG
      std::cout << "Id of connected components conn to top " << std::endl;
      std::set<int>::iterator its = comp_conn_to_electrode.begin();
      for (int i = 0; i < comp_conn_to_electrode.size(); i++){
	  int id = *its;
	  std::cout << id << " " << CCs[id].color << std::endl;
	  its++;
      }
      std::cout << std::endl;
#endif


      // pass info about connectivity to top electrode to data of components
      for(unsigned int  i = 0; i < CCs.size(); i++ ){
	  if( comp_conn_to_electrode.find(i) != comp_conn_to_electrode.end() )
	      CCs[i].if_connected_to_electrode += 2;
      }
  }//determine_basic_stats_of_ccs




}
#endif
