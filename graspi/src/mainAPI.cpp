/***
 *  $Id$
 **
 *  File: graspi.hpp
 *  Created: May 8, 2012
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2012 Olga Wodo, Baskar Ganapathysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#include <cstdlib>
#include <iostream>
#include "time_check.hpp"
#include "graspi_types.hpp"
#include "graph_constructors.hpp"
#include "graph_io.hpp"
#include "graph_cc.hpp"
#include "performance_indicators.hpp"
#include "graspi_descriptors.hpp"
#include "graspi.hpp"


int main(int argc, char** argv){

     /**********************************************************************
     * Basic variables to parse command line (filename and flags)          *
     * Followed by basic command line parsing                              *
     *                                                                     *
     **********************************************************************/

    int infile_flag = -1;    // format 0=array, 1-graph
    std::string infile_name; //filename to read data from
    float pixelsize = 1.0;
    bool if_per = 0;         // if periodic BC (0-false, 1-true)
    int n_of_phases = 2;    // number of phases (default 2)
    std::string res_path("./");

    for (int i = 1; i < (argc-1); i++){
	std::string param(argv[i]);
	if (param == std::string("-g")) {
	    infile_flag = 1;
	    infile_name = argv[i + 1];
	    i++;
	} else if (param == std::string("-a")) {
	    infile_flag = 0;
	    infile_name = argv[i + 1];
	    i++;
	} else if (param == std::string("-s")) {
	    pixelsize = atof(argv[i + 1]);
	    i++;
	} else if (param == std::string("-p")) {
	    if_per = atoi(argv[i + 1]);
	    i++;
	}else if (param == std::string("-n")) {
	    n_of_phases = atoi(argv[i + 1]);
	    i++;
	}else if (param == std::string("-r")) {
	    res_path = argv[i + 1];
	    i++;
	}
    }
    std::string log_filename = res_path + std::string("graspi.log");
    std::ofstream d_log(log_filename.c_str());
    d_log << "[STATUS] 1/5" << std::endl;
    d_log << "[STATUS] Command line read " << std::endl;
#ifdef DEBUG
    std::cout << "-------------------------------------------" << std::endl
	      << "(1) Command line read!"
	      << " infile_flag:" << infile_flag << std::endl
	      << " infile_name:" << infile_name << std::endl
	      << " pixelsize:" << pixelsize << std::endl
	      << " if_per:" << if_per << std::endl
	      << " n_of_phases:" << n_of_phases << std::endl
	      << " res_path" << res_path << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
#endif

    if ( (argc == 1) || (infile_flag == -1) ) {
	std::cout << std::endl
		  << "GraSPI accepts input data in two formats:"
		  << " graph, and array." << std::endl
		  << "For more information check documentation"
		  << std::endl << std::endl;
	std::cout << argv[0] << " -g <filen.graphe> " << std::endl;
	std::cout << argv[0] << " -a <file.txt> (row-major order) "
		  << "-s <pixelSize> (default 1) "
		  << "-p <{0,1}> (default 0-false) "
		  << "-n <{2,3}> (default 2-{D,A}) "
		  << "-r path where store results (default ./) "
		  << std::endl << std::endl;
	return 0;
    }


    /**********************************************************************
     * Graph definition and declaration                                    *
     *(list of containers to store labels of vertices and edges etc)       *
     *                                                                     *
     **********************************************************************/
    graspi::dim_g_t d_g(n_of_phases,0,3); //structure storing basic dimensions of G
    graspi::dim_a_t d_a;                     //structure storing color array dimensions
    graspi::vertex_colors_t vertex_colors;   //container storing colors of vertices

    /***********************************************************************
     * Graph construction                                                  *
     *                                                                     *
     **********************************************************************/

    if(infile_flag == 0){
	if( !graspi::read_array(infile_name, vertex_colors, d_a, d_g) ){
	    std::cout << "Problem with input file - "
		      << "Reading input file with array! "
		      << std::endl;
	    return -1;
	}
    
    }
    
    std::vector<graspi::desc_t> descriptors=graspi::compute_descriptors(vertex_colors, d_a.nx, d_a.ny, d_a.nz, pixelsize, if_per, res_path);
    

    
    return 0;
}
