#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>

bool isNotDigit(int c)
{
    return !isdigit(c);
}

int main(int argc, char* argv[])try{

    if(argc!=3) {
	std::cerr << "Usage: plt2arr <in_filename.plt> <out_filename.txt> "
		  << std::endl;
	return 0;
    }

    bool verbose = false;
    //    verbose = true;

    std::ifstream f_in(argv[1]);
    std::ofstream f_out(argv[2]);
    std::ostringstream oss_out_1;
    std::ostringstream oss_out_2;

    if(! f_in) {
	std::cerr << "Problem with inputfile " << std::endl;
	return -1;
    }

    // plt format components
    int nsd = 0;
    int nofElements = 0;
    int nofNodes = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;

    std::string elType;
    do{

	std::string tmp;
	getline(f_in,tmp);

	if( (!f_in) && (!f_in.eof()) ) {
	    std::cerr << "Problem with the stream!!!" << tmp << std::endl;
	    return false;
	}

	// read header - begin
	std::size_t pos_var = tmp.find("VARIABLE");
	if( pos_var != std::string::npos){
	    std::istringstream iss;
	    iss.str(tmp);
	    if(verbose) std::clog << "[reading line:] " << tmp << std::endl;
	    std::string token;
	    while( iss >> token ){
		if( (token[0] == '"') || (token[token.size()-1] == '"') )
		    if(
		       (token[1] == 'x') || (token[1] == 'y')
		       || (token[1] == 'z')
		       ||
		       (token[1] == 'X') || (token[1] == 'Y')
		       || (token[1] == 'Z')
		       ) nsd++;
	    }
	    if(verbose) std::clog << "[extracting data] nsd: " << nsd 
				  << std::endl;
	}

	int ix=1,iy=1,iz=1;
	std::size_t pos_zone = tmp.find("ZONE");
	if( pos_zone != std::string::npos){
	    std::istringstream iss;
	    iss.str(tmp);
	    if(verbose) std::clog << "[reading line:] " << tmp << std::endl;

	    size_t found = tmp.find("F=POINT");
	    if (found!=std::string::npos){
		std::string token;
		while( iss >> token ){
		    if ( token[0] == 'I') {
			token.resize( std::remove_if(token.begin(),
						     token.end(),
						     isNotDigit)
				      - token.begin()
				      );
			ix=std::atoi(token.c_str());
		    }
		    if ( token[0] == 'J') {
			token.resize( std::remove_if(token.begin(),
						     token.end(),
						     isNotDigit)
				      - token.begin()
				      );
			iy=std::atoi(token.c_str());
		    }
		    if ( token[0] == 'K') {
			token.resize( std::remove_if(token.begin(),
						     token.end(),
						     isNotDigit)
				      - token.begin()
				      );
			iz=std::atoi(token.c_str());
		    }
		    nofNodes=ix*iy*iz;
		}
	    }

	    found = tmp.find("QUADRILATERAL");
	    std::size_t found1 = tmp.find("BRICK");

	    if ((found!=std::string::npos) || (found1!=std::string::npos) ){

		std::string token;
		while( iss >> token ){
		    if ( (token[0] == 'N') || (token[0] == 'I') ){
			token.resize( std::remove_if(token.begin(), token.end(),
						     isNotDigit) - token.begin());
			nofNodes=std::atoi(token.c_str());
			if(verbose) std::clog << "[extracting data] nOfNodes: "
					      <<  nofNodes << std::endl;
		    }
		    if ( ( token[0] == 'E') && (token[1] != 'T') ) {
			token.resize( std::remove_if(token.begin(), token.end(),
						     isNotDigit) - token.begin());
			nofElements=std::atoi(token.c_str());
			if(verbose) std::clog << "[extracting data] nOfElements: "
					      <<  nofElements << std::endl;
		    }
		    if ( ( token[0] == 'E') && (token[1] == 'T') ) {
			elType.resize(token.size()-3);
			std::copy(token.begin()+3, token.end(), elType.begin() );
			if(verbose) std::clog << "[extracting data] elTyle: "
					      <<  elType << std::endl;
			if( (elType != "QUADRILATERAL" )
			    && (elType != "BRICK" ) ){
			    std::cerr << "This type of element is not supported!"
				      << std::endl;
			    return -1;
			}
		    }
		}

		if(verbose) std::cerr << "(phase2arr) nsd: " << nsd
				      << " nofNodes: " << nofNodes
				      << " nofElements: " << nofElements
				      << std::endl;

		double x,y,z;
		double x_prev, y_prev,z_prev = 0;
		int phase;
		for(int i = 0; i < nofNodes; i++){
		    f_in >> x;
		    if(nsd > 1) f_in >> y;
		    if(nsd > 2) f_in >> z;
		    f_in >> phase;
		    if(nsd > 2){
			if(fabs(z-z_prev) > 1e-20){
			    nz++;
			    ny = 0;
			    oss_out_2 << std::endl;
			}
		    }
		    if(nsd > 1){
			if(fabs(y-y_prev) > 1e-20){
			    ny++;
			    nx = 0;
			    oss_out_2 << std::endl;
			}
		    }
		    if(fabs(x-x_prev) > 1e-20) nx++;
		    x_prev = x;
		    if(nsd > 1) y_prev = y;
		    if(nsd > 2) z_prev = z;

		    oss_out_2 << phase << " ";
		}
		if(nsd == 3) nz++;
		if(nsd == 2) ny++;
		if(nsd == 1) nx++;
		break;
	    }
	}// read header - end

    }while(!f_in.eof());

    oss_out_1 << nx << " " << ny << " " << nz << std::endl;

    std::string buffer = oss_out_1.str();
    int size = oss_out_1.str().size();
    f_out.write (buffer.c_str(),size);

    buffer = oss_out_2.str();
    size = oss_out_2.str().size();
    f_out.write (buffer.c_str(),size);

    f_out.close();


    return 0;
 }
 catch(std::bad_alloc e){
     std::cerr << "Problem with memory allocation " << e.what();
     return -1;
 }
 catch(...){
     std::cerr << "Unknown error" << std::endl;
     return -1;
 }

