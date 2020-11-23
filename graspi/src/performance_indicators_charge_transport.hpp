/***
 *  $Id$
 **
 *  File: performance_indicators_charge_transport.hpp
 *  Created: May 9, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapathysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef PERFORMANCE_INDICATORS_CT_HPP
#define PERFORMANCE_INDICATORS_CT_HPP

#include "graspi_types.hpp"
#include "graspi_descriptors.hpp"

namespace graspi {

  std::pair<int,int> find_useful_cc(
                    graspi::DESC& descriptors,
				    graspi::graph_t* G,
				    const vertex_colors_t& C,
				    const vertex_ccs_t& vCC,
				    const ccs_t& CC){

      int n_of_white_ccs = 0;
      int n_of_grey_ccs = 0;
      int n_of_black_ccs = 0;
      int n_of_white_ccs_conn_to_bottom = 0; // usefull white components
      int n_of_black_ccs_conn_to_top = 0;    // usefull black components
      int n_of_grey_ccs_conn_to_bottom = 0;  // usefull grey components
      int n_of_grey_ccs_conn_to_top = 0;     // usefull grey components
      int n_of_grey_ccs_conn_to_both = 0;    // usefull grey components
      // number of usefull white/black cells
      // number of all white/black cells -> fraction of useful
      int n_of_white_vertices = 0;
      int n_of_white_vertices_conn_to_bottom = 0;
      int n_of_grey_vertices = 0;
      int n_of_grey_vertices_conn_to_bottom = 0;
      int n_of_grey_vertices_conn_to_top = 0;
      int n_of_grey_vertices_conn_to_both = 0;
      int n_of_black_vertices = 0;
      int n_of_black_vertices_conn_to_top = 0;

      int flag_n_phases = 2;

      for(unsigned int i = 0; i < CC.size(); i++){
	  if( CC[i].color == WHITE){
	      n_of_white_ccs++;
	      n_of_white_vertices += CC[i].size;
	      if( CC[i].if_connected_to_bottom() ) {
	      n_of_white_ccs_conn_to_bottom++;
	      n_of_white_vertices_conn_to_bottom += CC[i].size;
	      }
          }
	  if( CC[i].color == BLACK){
	      n_of_black_ccs++;
	      n_of_black_vertices += CC[i].size;
	      if ( CC[i].if_connected_to_top() ){
		  n_of_black_ccs_conn_to_top++;
		  n_of_black_vertices_conn_to_top += CC[i].size;
	      }
	  }
	  if( CC[i].color == GREY){
	      flag_n_phases = 3;
	      n_of_grey_ccs++;
	      n_of_grey_vertices += CC[i].size;
	      if ( CC[i].if_connected_to_top() ){
		  n_of_grey_ccs_conn_to_top++;
		  n_of_grey_vertices_conn_to_top += CC[i].size;
	      }
	      if ( CC[i].if_connected_to_bottom() ){
		  n_of_grey_ccs_conn_to_bottom++;
		  n_of_grey_vertices_conn_to_bottom += CC[i].size;
	      }
	      if ( CC[i].if_connected_to_top_and_bottom() ){}
		  n_of_grey_ccs_conn_to_both++;
		  n_of_grey_vertices_conn_to_both += CC[i].size;
	  }

      }
//      os << "[STATS] Number of black connected components: " << n_of_black_ccs << std::endl
//	 << "[STATS] Number of white connected components: " <<  n_of_white_ccs << std::endl;

      descriptors.update_desc("STAT_CC_D",n_of_black_ccs);
      descriptors.update_desc("STAT_CC_A",n_of_white_ccs);

      
      if( flag_n_phases == 3){
//          os << "[STATS] Number of grey connected components: " << n_of_grey_ccs << std::endl;
          descriptors.update_desc("STAT_CC_M",n_of_grey_ccs);
      }

//      os << "[STATS] Number of black connected components connected to top: "
//	 << n_of_black_ccs_conn_to_top << std::endl
//	 << "[STATS] Number of white connected components connected to bottom: "
//	 << n_of_white_ccs_conn_to_bottom << std::endl;
      
      descriptors.update_desc("STAT_CC_D_An",n_of_black_ccs_conn_to_top);
      descriptors.update_desc("STAT_CC_A_Ca",n_of_white_ccs_conn_to_bottom);

      
      if( flag_n_phases == 3){
//	  os << "[STATS] Number of grey connected components connected to top: "
//	     << n_of_grey_ccs_conn_to_top << std::endl
//	     << "[STATS] Number of grey connected components connected to bottom: "
//	     << n_of_grey_ccs_conn_to_bottom << std::endl
//	     << "[STATS] Number of grey connected components connected to both: "
//          << n_of_grey_ccs_conn_to_both << std::endl;
          descriptors.update_desc("STAT_CC_M_An",n_of_grey_ccs_conn_to_top);
          descriptors.update_desc("STAT_CC_M_Ca",n_of_grey_ccs_conn_to_bottom);
          descriptors.update_desc("STAT_CC_M_AnCa",n_of_grey_ccs_conn_to_both);

      }

//     os << "[STATS] Number of vertices: "
//	 << n_of_black_vertices + n_of_white_vertices << std::endl;
      descriptors.update_desc("STAT_n",n_of_black_vertices + n_of_white_vertices);


//      os << "[STATS] Number of black vertices: "
//	 << n_of_black_vertices << std::endl
//	 << "[STATS] Number of white vertices: "
//	 << n_of_white_vertices << std::endl;
      
      descriptors.update_desc("STAT_n_D",n_of_black_vertices);
      descriptors.update_desc("STAT_n_A",n_of_white_vertices);
      
      
      if(flag_n_phases == 3){
//      os << "[STATS] Number of grey vertices: " << n_of_grey_vertices << std::endl;
          descriptors.update_desc("STAT_n_M",n_of_grey_vertices);
      }
      
//      os << "[F ABS] Fraction of black vertices: "
//	 << (double)n_of_black_vertices
//	  / (n_of_black_vertices+n_of_white_vertices+n_of_grey_vertices) << std::endl;

      descriptors.update_desc("ABS_f_D",(double)n_of_black_vertices
                              / (n_of_black_vertices+n_of_white_vertices+n_of_grey_vertices));

      
      if(flag_n_phases == 3)
 //         os << "[F ABS] Fraction of grey vertices: "
 //         << (double)n_of_grey_vertices
 //         / (n_of_black_vertices+n_of_white_vertices+n_of_grey_vertices) << std::endl;
          descriptors.update_desc("ABS_f_M",(double)n_of_grey_vertices
                                           / (n_of_black_vertices+n_of_white_vertices+n_of_grey_vertices));
      
      if(flag_n_phases == 3){
//         os << "[F CT] Fraction of black and grey vertices connected to top: "
//          << (double)(n_of_black_vertices_conn_to_top+n_of_grey_vertices_conn_to_top)
//          /(n_of_black_vertices+n_of_grey_vertices)
//          << std::endl
//          << "[F CT] Fraction of white and grey vertices connected to bottom: "
//          << (double)(n_of_white_vertices_conn_to_bottom+n_of_grey_vertices_conn_to_bottom)
//          /(n_of_white_vertices+n_of_grey_vertices)
//          << std::endl;
          descriptors.update_desc("CT_f_conn_DM_An",(double)(n_of_black_vertices_conn_to_top+n_of_grey_vertices_conn_to_top)
                                            /(n_of_black_vertices+n_of_grey_vertices) );
          descriptors.update_desc("CT_f_conn_AM_Ca",(double)(n_of_white_vertices_conn_to_bottom+n_of_grey_vertices_conn_to_bottom)
                                            /(n_of_white_vertices+n_of_grey_vertices));
          
      }else{
          
//          os << "[F CT] Fraction of useful vertices - w/o islands: " <<
//          (double)( n_of_black_vertices_conn_to_top
//                   +
//                   n_of_white_vertices_conn_to_bottom
//                   ) / (n_of_black_vertices+n_of_white_vertices)
//          << std::endl;
//          os << "[F CT] Fraction of black vertices connected to top: "
//          << (double)n_of_black_vertices_conn_to_top/n_of_black_vertices
//          << std::endl
//          << "[F CT] Fraction of white vertices connected to bottom: "
//          << (double)n_of_white_vertices_conn_to_bottom/n_of_white_vertices
//          << std::endl;
          
          descriptors.update_desc("CT_f_conn_D",(double)(n_of_black_vertices_conn_to_top+n_of_white_vertices_conn_to_bottom) / (n_of_black_vertices+n_of_white_vertices));
          descriptors.update_desc("CT_f_conn_D_An",(double)n_of_black_vertices_conn_to_top/n_of_black_vertices);
          descriptors.update_desc("CT_f_conn_A_Ca",(double)n_of_white_vertices_conn_to_bottom/n_of_white_vertices);

          
      }
      return std::pair<int,int>(n_of_black_vertices_conn_to_top,
                                n_of_white_vertices_conn_to_bottom);
  }

    
    
    
  inline std::pair<int,int>
  identify_complementary_paths_from_green(
					  graph_t* G,
					  const vertex_colors_t& C,
					  const edge_colors_t& L,
					  const vertex_ccs_t& vCC,
					  const ccs_t& CC ){

      int n_1st_order_edges_conn_blue_red = 0;
      int n_1st_order_edges = 0;

      graph_t::edge_iterator e, e_end;
      boost::tie(e, e_end) = boost::edges(*G);

      for (; e != e_end; ++e) {
	  vertex_t s = boost::source(*e, *G);
	  vertex_t t = boost::target(*e, *G);
	  std::pair<int,int> p = std::pair<int,int>(std::min(s,t),
						    std::max(s,t));
	  edge_colors_t::const_iterator edge = L.find(p);
	  char edge_color = edge->second;

	  // find first order edges connectiong black and white
	  if(
	     ( (C[s] + C[t]) == 1 )  && ( edge_color == 'f' )
	     ){
	      n_1st_order_edges++;

	      vertex_t black_vertex = s;
	      vertex_t white_vertex = t;
	      if( C[t] == BLACK ) {
		  black_vertex = t;
		  white_vertex = s;
	      }

	      if(
		 (CC[vCC[black_vertex] ].if_connected_to_top() )
		 &&
		 (CC[vCC[white_vertex] ].if_connected_to_bottom() )
		 ){
		  n_1st_order_edges_conn_blue_red++;
	      }
	  }//if first order int
      } // for e

      return std::pair<int,int>(n_1st_order_edges,
				n_1st_order_edges_conn_blue_red);
  }//identify_complementary_paths_from_green


  inline void
  compute_shortest_distance_from_sourceC_to_targetC(
						    COLOR sourceC,
						    COLOR targetC,
						    graph_t* G,
						    const dim_g_t& d_g,
						    const vertex_colors_t& C,
						    const edge_weights_t& W,
						    std::vector<float>& d,
						    std::string filename = ""
						    ){
      vertex_t source = d_g.id(sourceC);
      connect_same_color_and_relevant_meta_vertex pred(*G,C);
      determine_shortest_distances( G, W, source, pred, d);

      if(filename.size() != 0){
	  std::ostringstream oss_out;
	  for (unsigned int i = 0; i < d.size(); i++) {
	      unsigned int c = C[i];
	      if ( ( c == targetC )
		   && ( fabs(d[i]) < std::numeric_limits<float>::max() )
		   ) {
		  oss_out << d[i] << std::endl;
	      }
	  }
	  std::ofstream f_out(filename.c_str());
	  std::string buffer = oss_out.str();
	  int size = oss_out.str().size();
	  f_out.write (buffer.c_str(),size);
	  f_out.close();
      }
  }//compute_shortest_distance_from_black_to_red

    
    
    
  inline void
  identify_useful_triple_black_white_green( graph_t* G,
					    const vertex_colors_t& C,
					    const vertex_ccs_t& vCC,
					    const ccs_t& CC,
					    std::set<int>& id_blacks_conn_green_red,
					    std::set<int>& id_whites_conn_green_blue
					    ){
      //identify set of all black vertices with connection to green and path to red
      //identify set of all white vertices with connection to green and path to blue
      connect_relevant_meta_vertex pred_black(*G, C, vCC, CC, BLACK);
      connect_relevant_meta_vertex pred_white(*G, C, vCC, CC, WHITE);

      graph_t::edge_iterator e, e_end;
      boost::tie(e, e_end) = boost::edges(*G);
      for (; e != e_end; ++e) {
	  vertex_t s = boost::source(*e, *G);
	  vertex_t t = boost::target(*e, *G);
	  if (
	      (( C[s] == GREEN ) && ( C[t] == BLACK ))
	      ||
	      (( C[t] == GREEN ) && ( C[s] == BLACK ) )
	      ){
	      if( pred_black(t) ) id_blacks_conn_green_red.insert(t);
	      if( pred_black(s) ) id_blacks_conn_green_red.insert(s);
	  }
	  if (
	      (( C[s] == GREEN ) && ( C[t] == WHITE ))
	      ||
	      (( C[t] == GREEN ) && ( C[s] == WHITE ) )
	      ){
	      if( pred_white(t) ) id_whites_conn_green_blue.insert(t);
	      if( pred_white(s) ) id_whites_conn_green_blue.insert(s);
	  }
      }
  }//identify_useful_triple_black_white_green


  inline void
  identify_black_vertices_connected_to_green(
					     const std::vector<float>& d_red,
					     const std::set<int>& id_blacks_conn_green_red
					     ){

      std::string filename("DistancesGreenToRedViaBlack.txt");
      std::ostringstream oss_g_r_b;

//      unsigned int black_int = id_blacks_conn_green_red.size();
      std::vector<float> d_green_to_red_via_black;
      std::set<int>::iterator iter(id_blacks_conn_green_red.begin());
      std::set<int>::iterator  end(id_blacks_conn_green_red.end());

      for (; iter != end; ++iter) {
	  d_green_to_red_via_black.push_back( d_red[*iter ] ) ;
	  oss_g_r_b <<  d_red[*iter] << std::endl;
      }

      copy( d_green_to_red_via_black.begin(), d_green_to_red_via_black.end(),
	    std::ostream_iterator<float>(oss_g_r_b, "\n"));

      std::ofstream f_out_g_r_b(filename.c_str());
      std::string buffer = oss_g_r_b.str();
      int size = buffer.size();
      f_out_g_r_b.write (buffer.c_str(),size);
      f_out_g_r_b.close();
  }//identify_black_vertices_connected_to_green

  inline void print_distances_of_ids(
				     const std::vector<float>& d,
				     const std::set<int>& ids,
				     const std::string& filename
				     ){

      std::ostringstream oss;

      std::set<int>::iterator iter(ids.begin());
      std::set<int>::iterator  end(ids.end());

      for (; iter != end; ++iter) {
	  oss <<  d[*iter] << std::endl;
      }

      std::ofstream f_out(filename.c_str());
      std::string buffer = oss.str();
      int size = buffer.size();
      f_out.write(buffer.c_str(),size);
      f_out.close();
  }//identify_black_vertices_connected_to_green


  inline double
  determine_and_print_tortuosity( const graspi::vertex_colors_t& vertex_colors,
				  const dim_a_t& d_a,
				  double pixelsize,
				  const std::vector<float>& d,
				  const std::string& filename_t,
				  const std::string& filename_id_t,
				  COLOR c_source,
				  COLOR c_target){
      std::ostringstream oss_out_t;
      std::ostringstream oss_out_id_t;

      int tort_1 = 0;
      int n_useful = 0; // useful means vertices with positive and finite distances
      double t = 0.0;
      double position_source = d_a.ny - 1;
      double eps = 1.0/d_a.ny;
      if(d_a.nz > 1) {
	  position_source = d_a.nz - 1;
	  eps = 1.0/d_a.nz;
      }

      if(c_source == BLUE) position_source = 0;

      int total_n = d_a.nx * d_a.ny;

      if(	(d_a.nz == 0) || (d_a.nz == 1) ){//2D
	  for(unsigned int j = 0; j < d_a.ny; j++){
	      for(unsigned int i = 0; i < d_a.nx; i++){
		  int id = i + d_a.nx * j;
		  if(d[id]< 0.5*std::numeric_limits<float>::max()){
		      if(vertex_colors[id] == c_target){
			  n_useful++;
			  double h_diff
			      = fabs((double)(position_source-j))*pixelsize;

			  if ( fabs(h_diff) < 1e-20 ) t = 1.0;
			  else t = d[id]/h_diff;

			  if (fabs(t-1.0) < eps) {
			      t = 1.0;
			      tort_1++;
			  }
			  oss_out_t << t << std::endl;
			  oss_out_id_t << id << " " << t << " "
				       << d[id] << " " << h_diff << std::endl;
		      }
		  }
	      }//i
	  }//j
      }else{//3D
	  total_n *= d_a.nz;
	  for(unsigned int k = 0; k < d_a.nz; k++){
	      for(unsigned int j = 0; j < d_a.ny; j++){
		  for(unsigned int i = 0; i < d_a.nx; i++){
		      int id = i + d_a.nx * ( j + d_a.ny * k);
		      if(d[id]< 0.5*std::numeric_limits<float>::max()){
			  if(vertex_colors[id] == c_target){
			      n_useful++;
			      double h_diff
				  = fabs((double)(position_source-k))*pixelsize;

			      if(fabs(h_diff)<1e-20)  t = 1.0;
			      else t = d[id]/h_diff;

			      if (fabs(t-1.0) < eps) {
				  t = 1.0;
				  tort_1++;
			      }
			      oss_out_t << t << std::endl;
			      oss_out_id_t << id << " " << t << " "
					   << d[id] << " " << h_diff << std::endl;
			  }
		      }
		  }//i
	      }//j
	  }//k
      }//end-3D


      std::ofstream f_out(filename_t.c_str());
      std::string buffer = oss_out_t.str();
      int size = oss_out_t.str().size();
      f_out.write (buffer.c_str(),size);
      f_out.close();
      f_out.open(filename_id_t.c_str());
      buffer = oss_out_id_t.str();
      size = oss_out_id_t.str().size();
      f_out.write (buffer.c_str(),size);
      f_out.close();

      if(n_useful == 0) return 0;

      return (double)tort_1/n_useful;
  }


  inline double
  determine_tortuosity( const graspi::vertex_colors_t& vertex_colors,
			const dim_a_t& d_a,
			double pixelsize,
			const std::vector<float>& d,
			COLOR c_source,
			COLOR c_target){

      int tort_1 = 0;
      int n_useful = 0; // useful means vertices with positive and finite distances
      double t = 0.0;
      double position_source = d_a.ny - 1;
      double eps = 1.0/d_a.ny;
      if(d_a.nz > 1) {
	  position_source = d_a.nz - 1;
	  eps = 1.0/d_a.nz;
      }

      if(c_source == BLUE) position_source = 0;

      int total_n = d_a.nx * d_a.ny;

      if(	(d_a.nz == 0) || (d_a.nz == 1) ){//2D
	  for(unsigned int j = 0; j < d_a.ny; j++){
	      for(unsigned int i = 0; i < d_a.nx; i++){
		  int id = i + d_a.nx * j;
		  if(d[id]< 0.5*std::numeric_limits<float>::max()){
		      if(vertex_colors[id] == c_target){
			  n_useful++;
			  double h_diff
			      = fabs((double)(position_source-j))*pixelsize;

			  if ( fabs(h_diff) < 1e-20 ) t = 1.0;
			  else t = d[id]/h_diff;

			  if (fabs(t-1.0) < eps) {
			      t = 1.0;
			      tort_1++;
			  }
		      }
		  }
	      }//i
	  }//j
      }else{//3D
	  total_n *= d_a.nz;
	  for(unsigned int k = 0; k < d_a.nz; k++){
	      for(unsigned int j = 0; j < d_a.ny; j++){
		  for(unsigned int i = 0; i < d_a.nx; i++){
		      int id = i + d_a.nx * ( j + d_a.ny * k);
		      if(d[id]< 0.5*std::numeric_limits<float>::max()){
			  if(vertex_colors[id] == c_target){
			      n_useful++;
			      double h_diff
				  = fabs((double)(position_source-k))*pixelsize;

			      if(fabs(h_diff)<1e-20)  t = 1.0;
			      else t = d[id]/h_diff;

			      if (fabs(t-1.0) < eps) {
				  t = 1.0;
				  tort_1++;
			      }
			  }
		      }
		  }//i
	      }//j
	  }//k
      }//end-3D


      if(n_useful == 0) return 0;

      return (double)tort_1/n_useful;
  }



}//graspi-namespace
#endif
