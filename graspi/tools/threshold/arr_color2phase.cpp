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

    if(argc!=4) {
	std::cerr << "Usage: arr_color2phase <in_filename.txt> <out_filename.txt> <threshold>"
		  << std::endl;
	return 0;
    }

    bool verbose = false;
    //    verbose = true;

    std::ifstream f_in(argv[1]);
    std::ofstream f_out(argv[2]);
    int threshold = atoi(argv[3]);
    std::ostringstream oss_out_1;
    std::ostringstream oss_out_2;

    if(! f_in) {
	std::cerr << "Problem with inputfile " << std::endl;
	return -1;
    }

    // array format components
    int nx = 0;
    int ny = 0;
    int nz = 0;

    std::string tmp;
    getline(f_in,tmp);
    std::istringstream iss(tmp);
    iss >> nx;
    iss >> ny;
    if(iss) iss >> nz;

    int nofNodes = nx * ny;
    if(nz != 0) nofNodes *= nz;

    if( (!f_in) && (!f_in.eof()) ) {
	std::cerr << "Problem with the stream!!!" << tmp << std::endl;
	return false;
    }

    if(verbose) std::cerr << "(arr_color2phase)"
			  << " nofNodes: " << nofNodes
			  << std::endl;

    int phase;
    for(int i = 0; i < nofNodes; i++){
	f_in >> phase;
	if (phase < threshold) oss_out_2 << "1 ";
	else oss_out_2 << "0 ";
    }// read header - end

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

