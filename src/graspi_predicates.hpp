/***
 *  $Id$
 **
 *  File: graspi_predicates.hpp
 *  Created: May 9, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapthysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef GRASPI_PREDICATES_HPP
#define GRASPI_PREDICATES_HPP

#include "graspi_types.hpp"
#include "graspi_cc.hpp"

namespace graspi{

  // Predicate used to identify connected components
  // consisting of one color vertices.
  // This predicate identifies edges connecting vertices of the same color
  class connect_same_color {
  public:
      connect_same_color() : G_(0), vertex_colors_(0) { }

      connect_same_color(const graspi::graph_t& G,
			 const graspi::vertex_colors_t& vertex_colors)
	  : G_(&G), vertex_colors_(&vertex_colors){ }

      bool operator()(const graspi::edge_descriptor_t& e) const {

	  return
	      ( (*vertex_colors_)[boost::source(e, *G_)]
		==
		(*vertex_colors_)[boost::target(e, *G_)]
		);
      }

  private:
      const graspi::graph_t* G_;
      const graspi::vertex_colors_t* vertex_colors_;
  }; // class connect_same_color


  // Predicate used to filter original graph to find the shortest distances
  // from source vertex - GREEN to all BLACK/GREY (color_).
  // Predicate is used to mask edges in G other than
  // (BLACK,BLACK) and (BLACK,GREEN) and (color,color).
  // Since source vertex is GREEN we allow edges to exists between
  // GREEN and BLACK.
  class connect_color_green {
  public:
      connect_color_green() : G_(0), vertex_colors_(0) { }

      connect_color_green(const graph_t& G,
			  const vertex_colors_t& C,
			  COLOR color, COLOR green = GREEN)
	  : green_(green), color_(color), G_(&G), vertex_colors_(&C)  { }

      bool operator()(const edge_descriptor_t& e) const {
	  if ( ( ( (*vertex_colors_)[boost::source(e, *G_)] == green_ )
		 && ( (*vertex_colors_)[boost::target(e, *G_)] == color_ ) )
	       ||
	       ( ( (*vertex_colors_)[boost::target(e, *G_)] == green_ )
		 && ( (*vertex_colors_)[boost::source(e, *G_)] == color_ ) )
	       )
	      return true;
	  return (
		  (*vertex_colors_)[boost::source(e, *G_)]
		  ==
		  (*vertex_colors_)[boost::target(e, *G_)]
		  );
      }

  private:
      COLOR green_;
      COLOR color_;
      const graph_t* G_;
      const vertex_colors_t* vertex_colors_;
  }; // class connect_black_green


  class connect_same_color_and_relevant_meta_vertex {
  public:
      connect_same_color_and_relevant_meta_vertex ()
	  : G_(0), vertex_colors_(0) { }
      connect_same_color_and_relevant_meta_vertex (const graph_t& G,
						   const vertex_colors_t& C)
	  : G_(&G), vertex_colors_(&C) { }

      bool operator()(const edge_descriptor_t& e) const {
	  if ( ((*vertex_colors_)[boost::source(e, *G_)]
		+
		(*vertex_colors_)[boost::target(e, *G_)]  )
	       == 20) return true; //black+red
	  if ( ((*vertex_colors_)[boost::source(e, *G_)]
		+
		(*vertex_colors_)[boost::target(e, *G_)]  )
	       == 11) return true; //white+blue
	  if ( ((*vertex_colors_)[boost::source(e, *G_)]
		+
		(*vertex_colors_)[boost::target(e, *G_)]  )
	       == 23) return true; //grey+red
	  if ( ((*vertex_colors_)[boost::source(e, *G_)]
		+
		(*vertex_colors_)[boost::target(e, *G_)]  )
	       == 13) return true; //grey+blue

	  return ((*vertex_colors_)[boost::source(e, *G_)]
		  ==
		  (*vertex_colors_)[boost::target(e, *G_)]);
      }

  private:
      const graph_t* G_;
      const vertex_colors_t* vertex_colors_;
  }; // class connect_same_color_and_relevant_meta_vertex


  class connect_relevant_meta_vertex {
  public:
      connect_relevant_meta_vertex()
	  : G_(0), vertex_colors_(0), vertex_ccs_(0), ccs_(0), c_(0) { }
      void change_color(COLOR c){ c_ = c; }

      connect_relevant_meta_vertex(const graph_t& G,
				   const vertex_colors_t& C,
				   const vertex_ccs_t& vCC,
				   const ccs_t& CC,
				   COLOR color)
	  : G_(&G), vertex_colors_(&C), vertex_ccs_(&vCC),
	    ccs_(&CC), c_(color) { }

      bool operator()(const vertex_t& vertex_id) const {
	  if( (*vertex_colors_)[vertex_id] != c_) return false;

	  if (
	      ( c_ == BLACK ) &&
	      ( (*ccs_)[(*vertex_ccs_)[vertex_id]].if_connected_to_top() )
	      ) return true;
	  if (
	      ( c_ == WHITE ) &&
	      ( (*ccs_)[(*vertex_ccs_)[vertex_id]].if_connected_to_bottom() )
	      ) return true;
	  return false;
      }
  private:
      const graph_t* G_;
      const vertex_colors_t* vertex_colors_;
      const vertex_ccs_t* vertex_ccs_;
      const ccs_t* ccs_;
      COLOR c_;
  }; // connect_relevant_meta_vertex

    
    // Predicate used to identify descriptor of a given name
    class find_desc_of_name {
    public:
        find_desc_of_name():desc_(" "){}
        find_desc_of_name(const std::string desc_name)
        : desc_(desc_name){ }
        
        void set_desc(const std::string& desc){desc_=desc;}
        
        bool operator()(const std::pair<float,std::string>& p_desc) const {
            return p_desc.second == desc_;
        }
        
    private:
        std::string desc_;
    }; // class find_desc_of_name

}//graspi-namespace

#endif
