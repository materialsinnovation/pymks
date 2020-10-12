## GraSPI: what it is and what it does?

GraSPI is a tool for morphology analysis. The first application was in the field of organic solar cells, where morphology affect the final performance of device. GraSPI helped us to quantify various morphological features. However, GraSPI can be used for any other application, assuming that descriptor used in GraSPI are physically relevant. We decided to share to code

## Small example

See documentaion 

## Major features 

GraSPI computes the set of descriptors/features that can used for data analytics (e.g., to establish process-structure-property maps). GraSPI is implemented in C++ and uses bash as a scripting tool to encode basic analysis pipelines. GraSPI aims to handle multiple input formats, and tools to convert various formats it to internal data structures. 

## Installation instructions 

GraSPI requires boost library. 
To compile go to src file, you find there Makefile. If you install boost library in location that is not searchable, you need to explicity provide the path. 


Type make all


## How to cite GraSPI

@article{wodo2012graph,  
  title={A graph-based formulation for computational characterization of bulk heterojunction morphology},  
  author={Wodo, Olga and Tirthapura, Srikanta and Chaudhary, Sumit and Ganapathysubramanian, Baskar},  
  journal={Organic Electronics},  
  volume={13},  
  number={6},  
  pages={1105--1113},  
  year={2012}  
}
 
## Support 

This project is supported by [National Science Foundation][NSF], award number [1906344][QRM award]

## License 

GraSPI is distributed under BSD license - for more details see the LICENSE.txt file in the main folder of the project. 

## Changelog 

We keep updating GraSPI, here we log the major changes:

* June 2020, the first release of GraSPI, with the capabilities to analyze two phase morphology


## Community information 

If you want to some additional features, send us email - see Contact info at the bottom. Here, we will keep the list of requests that we plan to introduce, and our planned extensions. 

## Contributor Information

GraSPI has been developing at two institutions: 

* Olga Wodo, University at Buffalo 
* Baskar Ganapathysubramian,  Iowa State Univiersity 

## Contact info

Any questions and queries can be directed to Olga: olgawodo at buffalo dot edu.  


[NSF]: https://www.nsf.gov/
[QRM award]: https://www.nsf.gov/awardsearch/showAward?AWD_ID=1906344&HistoricalAwards=false# graspi
# graspi
