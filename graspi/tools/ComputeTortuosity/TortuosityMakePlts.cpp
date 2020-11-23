#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <limits>

bool isNotDigit(int c)
{
    return !isdigit(c);
}


int main(int argc, char* argv[])try{

    if(argc!=4) {
	std::cerr << "Usage:" << argv[0] << " <inputFileWithPhase> <FileWithAllDistances> <phaseToInvestigate>" << std::endl;
	return 0;
    }

    bool verbose = false;

    std::ifstream f_in_phase     (argv[1]);
    std::ifstream f_in_distances (argv[2]);
    int           phaseToInvestigate = std::atoi(argv[3]);

    std::ofstream f_out("tortuosity.plt");
    std::ostringstream f_out_1;

    if(! f_in_phase) {
	std::cerr << "Problem with inputfile: phase " << std::endl;
	return -1;
    }

    if(! f_in_distances) {
	std::cerr << "Problem with inputfile: distances " << std::endl;
	return -1;
    }

    std::vector<double> vreal_distance, vshortest_distance, vtortuosity;
    std::vector<double> vphase;

    std::string tmp;
    int nsd = 3;
    int nx, ny, nz = 0;
    getline(f_in_phase,tmp);
    std::istringstream iss(tmp);
    iss >> nx;
    iss >> ny;
    if(iss) iss >> nz;
    if( (nz == 0) || (nz == 1) ) {nz = 1; nsd = 2;}
    int ntotal = nx * ny * nz;

    vphase.resize(ntotal);
    vreal_distance.resize(ntotal);
    vshortest_distance.resize(ntotal);
    vtortuosity.resize(ntotal,-1);

    int p;
    for(int i = 0; i < ntotal; i++){
	f_in_phase >> p;
	vphase[i]=p;
    }

    int id;
    double tortuosity, real_distance, shortest_distance;
    do{
	//	std::cerr << id << std::endl;
	f_in_distances >> id >> tortuosity >> real_distance >> shortest_distance;
	vshortest_distance[id]=shortest_distance;
	vreal_distance[id]=real_distance;
	vtortuosity[id]=tortuosity;
	if(!f_in_distances) {
	    break;
	}
    }while(!f_in_distances.eof());



    f_out_1 << "TITLE = Tort " << std::endl;
    if(nsd == 3){
	f_out_1 << "VARIABLES = \"X\", \"Y\", \"Z\", \"phase\", \"tort\", \"realDist\", \"shortestDist\" " << std::endl;
	f_out_1 << "ZONE I=" << nx << ", J=" << ny << ", K=" << nz << ", DATAPACKING=POINT" << std::endl;
    }
    if(nsd == 2){
	f_out_1 << "VARIABLES = \"X\", \"Y\", \"phase\", \"tort\", \"realDist\", \"shortestDist\" " << std::endl;
	f_out_1 << "ZONE I=" << nx << ", J=" << ny << ", DATAPACKING=POINT" << std::endl;
    }


    for (int k = 0; k < nz; k++){
	for (int j = 0; j < ny; j++){
	    for (int i = 0; i < nx; i++){
		int id = i + nx * ( j + ny * k);
		f_out_1 << i << " " << j << " " ;
		if (nsd == 3 ) f_out_1 << k << " ";
		f_out_1 << vphase[id] << " " << vtortuosity[id] << " " << vreal_distance[id] << " " << vshortest_distance[id] << std::endl;
	    }
	}
    }

    int size = f_out_1.str().size();
    f_out.write(f_out_1.str().c_str(),size);
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

