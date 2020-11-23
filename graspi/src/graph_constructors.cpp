#include <iostream>
#include "graph_constructors.hpp"

bool graspi::build_graph(const std::string& name,
			graph_t*& G,
			dim_g_t& d_g,
			vertex_colors_t& C,
			edge_weights_t& W,
			edge_colors_t& L) {
    std::ifstream f(name.c_str());
    if (!f) return false;
    return build_graph(f, G, d_g, C, W, L);
} // build_graph


bool graspi::build_graph(graph_t*& G, const dim_g_t& d_g,
		 vertex_colors_t& C, dim_a_t& d_a,
		 edge_weights_t& W,
		 edge_colors_t& L,
		 double pixelsize, bool if_per_on_size);

void graspi::initlize_colors_meta_vertices( vertex_colors_t& C, const dim_g_t& d_g ){
    
//    std::cerr << "Test: " << C.size() << " " << d_g.n_bulk << std::endl;
    
    C[d_g.n_bulk]   = BLUE;  // bottom electrode
    C[d_g.n_bulk+1] = RED;   // top electrode
    C[d_g.n_bulk+2] = GREEN; // meta vertex - I D/A
    if(d_g.n_phases == 3){
	C[d_g.n_bulk+3] = DGREEN; // meta vertex - I D/D+A
	C[d_g.n_bulk+4] = LGREEN; // meta vertex - I A/D+A
    }
}

template <typename Container>
bool graspi::update_graph(const Container& M, const dim_a_t& d,
		  graph_t& G, dim_g_t& d_g,
		  vertex_colors_t& C,
		  edge_weights_t& W,
		  edge_colors_t& L) {
    // no error checking
    C.resize( M.size()+d_g.n_meta() );
    for (unsigned int i = 0; i < M.size(); ++i) C[i] = M[i];

    //      initlize_color_meta_vertices( C, d_g );
    return true;
} // update_graph


/*
 * Function adds/updates edges to the graph
 * between interface meta-vertex and bulk vertices
 * with corresponding attributes
 * s      - source vertex to be checked
 * meta_t - meta vertex to which edge is linked
 * w,o    - attributes of new edges
 */

bool graspi::build_graph(std::istream& is,
			 graph_t*& G,
			 dim_g_t& d_g,
			 vertex_colors_t& C,
			 edge_weights_t& W,
			 edge_colors_t& L){

    // n is size of graphe w/o special vertices
    unsigned int n = 0;

    if(!is) {
    	std::cerr << "Problem with istream!" << std::endl;
    	return false;
    }

    int n_of_vertices = 1;
    int green_vertex, lgreen_vertex, dgreen_vertex;

    int i=0;
    do {
	std::string tmp;
	getline(is,tmp);
	std::istringstream iss(tmp);
	if(!is) {
	    std::cerr << "Problem with input file - line " << i << std::endl;
	    delete G;
	    return false;
	}

	if(i==0){
	    iss >> n_of_vertices;
	    n = n_of_vertices;
	    d_g.n_bulk = n_of_vertices;
	    G = new graspi::graph_t( d_g.n_total() );
	    C.resize(d_g.n_total(), 13);
	    initlize_colors_meta_vertices(C,d_g);
	    green_vertex = d_g.n_bulk+2;
	    if(d_g.n_phases == 3){
		dgreen_vertex = d_g.n_bulk+3;
		lgreen_vertex = d_g.n_bulk+4;
	    }
	}

	if( (i>0) && ( i <= n_of_vertices ) ){
	    int s = 0;
	    iss >> s;
	    iss >> C[s];
	    int t = 0;
	    double w = 0.0;
	    char o;
	    do{
		iss >> t;
		iss >> w;
		iss >> o;

		if(!iss ) break;

		if (t == -3) continue;
		if (t < 0) {
		    t = n-t-1;
		    w = 0;
		    o = 's';
		}
		add_edge_to_graph(s, t, o, w,
				  green_vertex, dgreen_vertex, lgreen_vertex,
				  G,C,W,L);

	    }while(iss);

	}

	if( i > n_of_vertices ){
	    int s = 0;
	    iss >> s;
	    iss >> C[s];
	    int t = 0;
	    double w = 0.0;
	    char o;
	    do{
		iss >> t;
		iss >> w;
		iss >> o;

		if(!iss ) break;

		if (t == -3) continue;
		if (t < 0) {
		    t = n-t-1;
		    w = 0;
		    o = 's';
		}
		add_edge_to_graph(s, t, o, w,
				  green_vertex, dgreen_vertex, lgreen_vertex,
				  G,C,W,L);
	    }while(iss);
	}
	if (i >= (n_of_vertices+2) ) break;
	i++;
    }while(is);


    return true;
}//build_graph



bool graspi::build_graph(graph_t*& G, const dim_g_t& d_g,
			 vertex_colors_t& C, dim_a_t& d_a,
			 edge_weights_t& W,
			 edge_colors_t& L,
			 double pixelsize, bool if_per_on_size
			 ) {

    G = new graspi::graph_t( d_g.n_total() );
    int n = d_g.n_bulk;
#ifdef DEBUG
    std::cout << "n: " << n << std::endl;
#endif

    if(d_a.nz == 0) d_a.nz = 1;
    initlize_colors_meta_vertices( C, d_g );
    int green_vertex, lgreen_vertex, dgreen_vertex;
    green_vertex = d_g.n_bulk+2;
    if(d_g.n_phases == 3){
	dgreen_vertex = d_g.n_bulk+3;
	lgreen_vertex = d_g.n_bulk+4;
    }

    int ngbr_size = 8;
    if(d_a.nz > 1) ngbr_size = 26;
    std::pair<int,char>* ngbr = new std::pair<int,char> [ngbr_size];

    for(unsigned int k = 0; k < d_a.nz; k++){
	for(unsigned int j = 0; j < d_a.ny; j++){
	    for(unsigned int i = 0; i < d_a.nx; i++){
		generate_ngbr(i,j,k,d_a,ngbr,if_per_on_size);

		int id = i + d_a.nx * ( j + d_a.ny * k );
		int s = id;
		for (int i_ngbr=0; i_ngbr < ngbr_size; i_ngbr++){
		    int t = ngbr[i_ngbr].first;
		    char o = ngbr[i_ngbr].second;
		    double w = 1.0 * pixelsize;
		    if( o == 's') w = sqrt(2.0) * pixelsize;
		    if( o == 't') w = sqrt(3.0) * pixelsize;

		    if (t == -3) continue;
		    // for metavertices corresponding to electrode correct indices
		    // old: cathode has index:-1 in the inputdata
		    // new: cathode has index: n_bulk+1 (appended at the end)
		    // similar operation applied to anode
		    // anode has index : n_bulk+2
		    if (t < 0){
			t = n-t-1;
			o = 's';
			w = 0.0;
		    }
#ifdef DEBUG
		    std::cerr << "( " << s << "," << t << " )"
			      << o << " " << w << " , "
			      << C[s] << " " << C[t] << std::endl;
#endif
		    add_edge_to_graph(s, t, o, w,
				      green_vertex,
				      dgreen_vertex,
				      lgreen_vertex,
				      G,C,W,L);
		}
	    }//i
	}//j
    }//k

    delete[] ngbr;

    return true;
} // build_graph


void graspi::add_edge_to_graph(int s, int t, char o, double w,
			       int green_vertex,
			       int dgreen_vertex,
			       int lgreen_vertex,
			       graph_t* G,
			       vertex_colors_t& C,
			       edge_weights_t& W,
			       edge_colors_t& L) {

    graspi::edge_descriptor_t e;
    bool e_res = false;

    boost::tie(e, e_res) = boost::edge(s, t, *G);
    if (e_res == false) {
	boost::tie(e, e_res) = boost::add_edge(s, t, *G);
	std::pair<int,int> p = std::pair<int,int>(std::min(s,t),
						  std::max(s,t));
	W[e] = w;
	L[p] = o;
    }

    if(C[s]+C[t] == 1){
	//add edge between white and green
	make_update_edge_with_meta_vertex( s, green_vertex,
					   w, o, G, W, L);
	//add edge between black and green
	make_update_edge_with_meta_vertex( t, green_vertex,
					   w, o, G, W, L);
    }//I D/A

    if(C[s]+C[t] == 3){
	//add edge between black and dgreen
	make_update_edge_with_meta_vertex( s, dgreen_vertex,
					   w, o, G, W, L);
	//add edge between grey and green
	make_update_edge_with_meta_vertex( t, dgreen_vertex,
					   w, o, G, W, L);
    }//I D/D+A

    if(C[s]+C[t] == 4){
	//add edge between white and lgreen
	make_update_edge_with_meta_vertex( s, lgreen_vertex,
					   w, o, G, W, L);
	//add edge between grey and lgreen
	make_update_edge_with_meta_vertex( t, lgreen_vertex,
					   w, o, G, W, L);
    }//I A/D+A
}

void graspi::make_update_edge_with_meta_vertex( int s, int meta_t,
						double w, char o,
						graph_t* G,
						edge_weights_t& W,
						edge_colors_t& L)
{

    //add edges between source and target_metavertex
    graspi::edge_descriptor_t e;
    bool e_res = false;
    boost::tie(e, e_res) = boost::edge(s, meta_t, *G);
    if (e_res == false) {
	std::pair<int,int> p = std::pair<int,int>(std::min(s,meta_t),
						  std::max(s,meta_t));
	boost::tie(e, e_res) = boost::add_edge(s, meta_t, *G);
	W[e] = w/2.0;
	L[p]=o;
    }else{ // if edge exist check if current distance is shorter
	// and change accordingly
	float existing_distance = boost::get(W, e);
	if( (w/2.0) < existing_distance){
	    std::pair<int,int> p = std::pair<int,int>(std::min(s,meta_t),
						      std::max(s,meta_t));
	    W[e] = w/2.0;
	    L[p] = o;
	}
    }
}


// determine the position in row-wise array on the basis of index i_x and i_y
int graspi::compute_pos_2D(int i_x, int i_y,
			   const dim_a_t& d_a,
			   bool if_per_on_sides){

    if( i_y == -1)    return -1;
    if( i_y == d_a.ny )  return -2;

    if (if_per_on_sides){
	if ( i_x == -1 )     i_x = d_a.nx-1;
	if ( i_x == d_a.nx ) i_x = 0;
    }else{
	if( ( i_x == -1 ) || (i_x == d_a.nx) ) return -3;
    }

    return i_x + d_a.nx * i_y ;
}


// determine the position in row-wise array on the basis of index i_x and i_y
int graspi::compute_pos_3D(int i_x, int i_y, int i_z,
			   const dim_a_t& d_a,
			   bool if_per_on_sides){

    if( i_z == -1)    return -1;
    if( i_z == d_a.nz )  return -2;

    if (if_per_on_sides){
	if ( i_x == -1 )     i_x = d_a.nx-1;
	if ( i_x == d_a.nx ) i_x = 0;
	if ( i_y == -1 )     i_y = d_a.ny-1;
	if ( i_y == d_a.ny ) i_y = 0;
    }else{
	if(
	   ( i_x == -1 ) || ( i_x == d_a.nx )
	   ||
	   ( i_y == -1 ) || ( i_y == d_a.ny )
	   )
	    return -3;
    }

    return i_x + d_a.nx * (i_y + d_a.ny * i_z) ;
}


// generate 2D neighborhood for 2D case
void graspi::generate_ngbr(int i, int j, int k,
			   const dim_a_t& d_a,
			   std::pair<int,char>* ngbr,
			   bool if_per_on_sides ){
    // 2D
    if( (d_a.nz == 0) || (d_a.nz == 1) ) {
	k = 0;
	ngbr[0].first = compute_pos_2D(i  ,j+1, d_a, if_per_on_sides);	// ngbr N
	ngbr[1].first = compute_pos_2D(i+1,j+1, d_a, if_per_on_sides);	// ngbr NE
	ngbr[2].first = compute_pos_2D(i+1,j  , d_a, if_per_on_sides);	// ngbr E
	ngbr[3].first = compute_pos_2D(i+1,j-1, d_a, if_per_on_sides);	// ngbr ES
	ngbr[4].first = compute_pos_2D(i  ,j-1, d_a, if_per_on_sides);	// ngbr S
	ngbr[5].first = compute_pos_2D(i-1,j-1, d_a, if_per_on_sides);	// ngbr SW
	ngbr[6].first = compute_pos_2D(i-1,j  , d_a, if_per_on_sides);	// ngbr W
	ngbr[7].first = compute_pos_2D(i-1,j+1, d_a, if_per_on_sides);	// ngbr WN

	ngbr[0].second = 'f'; // ngbr N
	ngbr[1].second = 's'; // ngbr NE
	ngbr[2].second = 'f'; // ngbr E
	ngbr[3].second = 's'; // ngbr ES
	ngbr[4].second = 'f'; // ngbr S
	ngbr[5].second = 's'; // ngbr SW
	ngbr[6].second = 'f'; // ngbr W
	ngbr[7].second = 's'; // ngbr WN
    }else{// 3D
	ngbr[0].first = compute_pos_3D(i  ,j+1, k, d_a, if_per_on_sides);	// ngbr N
	ngbr[1].first = compute_pos_3D(i+1,j+1, k, d_a, if_per_on_sides);	// ngbr NE
	ngbr[2].first = compute_pos_3D(i+1,j  , k, d_a, if_per_on_sides);	// ngbr E
	ngbr[3].first = compute_pos_3D(i+1,j-1, k, d_a, if_per_on_sides);	// ngbr ES
	ngbr[4].first = compute_pos_3D(i  ,j-1, k, d_a, if_per_on_sides);	// ngbr S
	ngbr[5].first = compute_pos_3D(i-1,j-1, k, d_a, if_per_on_sides);	// ngbr SW
	ngbr[6].first = compute_pos_3D(i-1,j  , k, d_a, if_per_on_sides);	// ngbr W
	ngbr[7].first = compute_pos_3D(i-1,j+1, k, d_a, if_per_on_sides);	// ngbr WN

	ngbr[0].second = 'f'; // ngbr N
	ngbr[1].second = 's'; // ngbr NE
	ngbr[2].second = 'f'; // ngbr E
	ngbr[3].second = 's'; // ngbr ES
	ngbr[4].second = 'f'; // ngbr S
	ngbr[5].second = 's'; // ngbr SW
	ngbr[6].second = 'f'; // ngbr W
	ngbr[7].second = 's'; // ngbr WN

	ngbr[8].first  = compute_pos_3D(i  ,j+1, k+1, d_a, if_per_on_sides);	// ngbr N
	ngbr[9].first  = compute_pos_3D(i+1,j+1, k+1, d_a, if_per_on_sides);	// ngbr NE
	ngbr[10].first = compute_pos_3D(i+1,j  , k+1, d_a, if_per_on_sides);	// ngbr E
	ngbr[11].first = compute_pos_3D(i+1,j-1, k+1, d_a, if_per_on_sides);	// ngbr ES
	ngbr[12].first = compute_pos_3D(i  ,j-1, k+1, d_a, if_per_on_sides);	// ngbr S
	ngbr[13].first = compute_pos_3D(i-1,j-1, k+1, d_a, if_per_on_sides);	// ngbr SW
	ngbr[14].first = compute_pos_3D(i-1,j  , k+1, d_a, if_per_on_sides);	// ngbr W
	ngbr[15].first = compute_pos_3D(i-1,j+1, k+1, d_a, if_per_on_sides);	// ngbr WN
	ngbr[16].first = compute_pos_3D(i,j,     k+1, d_a, if_per_on_sides);	// ngbr WN

	ngbr[8].second  = 's'; // ngbr N
	ngbr[9].second  = 't'; // ngbr NE
	ngbr[10].second = 's'; // ngbr E
	ngbr[11].second = 't'; // ngbr ES
	ngbr[12].second = 's'; // ngbr S
	ngbr[13].second = 't'; // ngbr SW
	ngbr[14].second = 's'; // ngbr W
	ngbr[15].second = 't'; // ngbr WN
	ngbr[16].second = 'f'; // ngbr WN

	ngbr[17].first = compute_pos_3D(i  ,j+1, k-1, d_a, if_per_on_sides);	// ngbr N
	ngbr[18].first = compute_pos_3D(i+1,j+1, k-1, d_a, if_per_on_sides);	// ngbr NE
	ngbr[19].first = compute_pos_3D(i+1,j  , k-1, d_a, if_per_on_sides);	// ngbr E
	ngbr[20].first = compute_pos_3D(i+1,j-1, k-1, d_a, if_per_on_sides);	// ngbr ES
	ngbr[21].first = compute_pos_3D(i  ,j-1, k-1, d_a, if_per_on_sides);	// ngbr S
	ngbr[22].first = compute_pos_3D(i-1,j-1, k-1, d_a, if_per_on_sides);	// ngbr SW
	ngbr[23].first = compute_pos_3D(i-1,j  , k-1, d_a, if_per_on_sides);	// ngbr W
	ngbr[24].first = compute_pos_3D(i-1,j+1, k-1, d_a, if_per_on_sides);	// ngbr WN
	ngbr[25].first = compute_pos_3D(i,j,     k-1, d_a, if_per_on_sides);	// ngbr WN

	ngbr[17].second = 's'; // ngbr N
	ngbr[18].second = 't'; // ngbr NE
	ngbr[19].second = 's'; // ngbr E
	ngbr[20].second = 't'; // ngbr ES
	ngbr[21].second = 's'; // ngbr S
	ngbr[22].second = 't'; // ngbr SW
	ngbr[23].second = 's'; // ngbr W
	ngbr[24].second = 't'; // ngbr WN
	ngbr[25].second = 'f'; // ngbr WN

    }

}

