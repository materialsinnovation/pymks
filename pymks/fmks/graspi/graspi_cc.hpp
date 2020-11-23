/***
 *  $Id$
 **
 *  File: graspi_cc.hpp
 *  Created: May 9, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapthysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef GRASPI_CC_HPP
#define GRASPI_CC_HPP

#include "graspi_types.hpp"

namespace graspi{

  struct CC{
      COLOR color;
      int size;
      int if_connected_to_electrode;
      // 0 - does not connected, 1 - connected to bottom
      // 2 - connected to top,   3 - connected to both

      CC():color(0), size(0),
	   if_connected_to_electrode(0) { }

      bool if_connected_to_top_or_bottom()const{
	  return (if_connected_to_electrode > 0 );
      }
      bool if_connected_to_top_and_bottom()const{
	  return (if_connected_to_electrode == 3 );
      }

      bool if_connected_to_top()const{
	  if( (if_connected_to_electrode == 2)
	      ||
	      (if_connected_to_electrode == 3) ) return true;
	  return false;
      }

      bool if_connected_to_bottom()const{
	  if( (if_connected_to_electrode == 1)
	      ||
	      (if_connected_to_electrode == 3) ) return true;
	  return false;
      }

      friend std::ostream& operator << (std::ostream& os, const CC& c){
	  return os << " color: " << c.color << " size: " << c.size
		    << " if connected: " << c.if_connected_to_electrode
		    <<  std::endl;
      }
  };//struct CC

}//graspi-namespace

#endif
