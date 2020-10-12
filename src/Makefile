all: graspi graspiAPI


# graspi requires boost library, install and add to you path or explicitly provide the path here. 
# graspi at the time of release was tested with version boost_1_72_0
BOOSTLIB="/Users/owodo/Packages/boost_1_72_0"

COPT = -I${BOOSTLIB} -O3   #-g3 -DDEBUG

graspi: main.cpp graph_constructors.cpp
	g++ -o graspi -Wno-deprecated ${COPT} main.cpp graph_constructors.cpp

graspiAPI: mainAPI.cpp graph_constructors.cpp
	g++ -o graspiAPI -Wno-deprecated ${COPT} mainAPI.cpp graph_constructors.cpp


clean:
	rm -rf *.o graspi graspiAPI
