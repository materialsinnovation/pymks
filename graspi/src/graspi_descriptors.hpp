/***
 *  $Id$
 **
 *  File: graspi_descriptors.hpp
 *  Created: September 28, 2020
 *
 *  Author: Olga Wodo, Baskar Ganapathysubramanian
 *  Copyright (c) 2020 Olga Wodo, Baskar Ganapathysubramanian
 *  See accompanying LICENSE.
 *
 *  This file is part of GraSPI.
 */

#ifndef GRASPI_DESCRIPTORS_HPP
#define GRASPI_DESCRIPTORS_HPP


namespace graspi {

    struct DESC{
        std::vector<desc_t> desc;
        find_desc_of_name find_desc;

        explicit DESC(){}
        
        // set of methods to initiate and
        void initiate_descriptors_2_phase(){
            std::pair <float, std::string> p_desc;
            p_desc.first = -1;  p_desc.second = "STAT_n";// number_of_vertices
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_e";//number_of_interface_edges
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_n_D";// number_of_black_vertices
            desc.push_back(p_desc);

            p_desc.first = -1;  p_desc.second = "STAT_n_A";// number_of_white_vertices
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_D";// number_of_black_connected_components
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_A";// number_of_white_connected_components
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_D_An";// number_of_black_connected_components_connected_to_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_A_Ca";// number_of_white_connected_components_connected_to_bottom
            desc.push_back(p_desc);

            p_desc.first = -1;  p_desc.second = "ABS_wf_D";//weighted_fraction_of_black_vertices
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "ABS_f_D";// fraction_of_black_vertices
            desc.push_back(p_desc);

            p_desc.first = -1;  p_desc.second = "DISS_wf10_D";//weighted_fraction_of_black_vertices_in_10_distance_to_interface
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "DISS_f10_D";//fraction_of_black_vertices_in_10_distance_to_interface
            desc.push_back(p_desc);
//            p_desc.first = -1;  p_desc.second = "DISS_f2_D";// fraction_of_black_vertices_in_2_distance_to_interface
//           desc.push_back(p_desc);

//            p_desc.first = -1;  p_desc.second = "CT_f_conn";// fraction_of_useful_vertices_-_w/o_islands
//            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_e_conn";//fraction_of_interface_with_complementary_paths_to_bottom_and_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_conn_D_An";// fraction_of_black_vertices_connected_to_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_conn_A_Ca";// fraction_of_white_vertices_connected_to_bottom
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_e_conn";//number_of_interface_edges_with_complementary_paths
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_e_D_An";//number_of_black_interface_vertices_with_path_to_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_e_A_Ca";//number_of_white_interface_vertices_with_path_to_bottom
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_D_tort1";//fraction_of_black_vertices_with_straight_rising_paths_(t=1)
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_A_tort1";//fraction_of_white_vertices_with_straight_rising_paths_(t=1)
            desc.push_back(p_desc);
//            p_desc.first = -1;  p_desc.second = "CT_wtort_D";// weighted_fraction_of_black_vertices_with_preferably_straight_rising_paths
//            desc.push_back(p_desc);
//            p_desc.first = -1;  p_desc.second = "CT_wtort_A";// weighted_fraction_of_white_vertices_with_preferably_straight_rising_paths
//            desc.push_back(p_desc);
//            p_desc.first = -1;  p_desc.second = "CT_n_A_adj_Ca";// number_of_white_vertices_in_direct_contact_with_blue
//            desc.push_back(p_desc);
//            p_desc.first = -1;  p_desc.second = "CT_n_D_adj_An";// number_of_black_vertices_in_direct_contact_with_red
//            desc.push_back(p_desc);
            
        }
        
        void print_descriptors_2_phase(std::ostream& os){

            for (unsigned int i=0; i< desc.size();i++){
                if (fabs(desc[i].first +1 ) > 1e-10)
                    os << desc[i].second << " " << desc[i].first << std::endl;
            }
        }
        

        // set of methods to initiate and
        void initiate_descriptors_3_phase(){
            std::pair <float, std::string> p_desc;
            p_desc.first = -1;  p_desc.second = "STAT_n";//number_of_vertices
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_e";//number_of_interface_edges
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_n_D";//number_of_black_vertices
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_n_A";//number_of_white_vertices
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_D";//number_of_black_connected_components
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_A";//number_of_white_connected_components
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_D_An";//number_of_black_connected_components_connected_to_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "STAT_CC_A_Ca";//number_of_white_connected_components_connected_to_bottom
            desc.push_back(p_desc);
            
            p_desc.first = -1;  p_desc.second = "ABS_wf_D";//weighted_fraction_of_black_vertices
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "ABS_f_D";//fraction_of_black_vertices
            desc.push_back(p_desc);
            
            p_desc.first = -1;  p_desc.second = "DISS_wf10_D";//weighted_fraction_of_black_vertices_in_10_distance_to_interface
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "DISS_f10_D";//fraction_of_black_vertices_in_10_distance_to_interface
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "DISS_f2_D";// fraction_of_black_vertices_in_2_distance_to_interface
            desc.push_back(p_desc);
            
            p_desc.first = -1;  p_desc.second = "CT_f_conn_D";//fraction_of_useful_vertices_-_w/o_islands
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_e_conn";//fraction_of_interface_with_complementary_paths_to_bottom_and_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_conn_D_An";//fraction_of_black_vertices_connected_to_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_conn_A_Ca";//fraction_of_white_vertices_connected_to_bottom
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_e_conn";//number_of_interface_edges_with_complementary_paths
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_e_D_An";//number_of_black_interface_vertices_with_path_to_top
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_e_A_Ca";//number_of_white_interface_vertices_with_path_to_bottom
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_D_tort1";//fraction_of_black_vertices_with_straight_rising_paths_(t=1)
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_f_A_tort1";//fraction_of_white_vertices_with_straight_rising_paths_(t=1)
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_wtort_D";// weighted_fraction_of_black_vertices_with_preferably_straight_rising_paths
            desc.push_back(p_desc);
            p_desc.first = -1;  p_desc.second = "CT_wtort_A";// weighted_fraction_of_white_vertices_with_preferably_straight_rising_paths
            desc.push_back(p_desc);
//            p_desc.first = -1;  p_desc.second = "CT_n_A_adj_Ca";// number_of_white_vertices_in_direct_contact_with_blue
//            desc.push_back(p_desc);
//            p_desc.first = -1;  p_desc.second = "CT_n_D_adj_An";// number_of_black_vertices_in_direct_contact_with_red
//            desc.push_back(p_desc);
            
        }

        
        bool update_desc(std::string desc_name,float value){
            
            find_desc.set_desc(desc_name);
            std::vector< std::pair<float, std::string> >::iterator it;
            it = std::find_if( desc.begin(), desc.end(), find_desc );
            if (it == desc.end())
                return false;
            else{
                (*it).first=value;
                return true;
            }
        }
        
        
        
        
    };// struct DESC
}


#endif // GRASPI_HPP
