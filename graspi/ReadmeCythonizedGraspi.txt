 requirements/configuration:
	in setupGraspiCython.py change the path to the boost accordingly:
             include_dirs=[numpy.get_include(), '/Users/owodo/Packages/boost_1_72_0', 'src'],

step 1:
	python3 setupGraspiCython.py build_ext -i
	As an outcome you should get graspi.cpython-ARCH.so object in the folder
step 2:
	python3 testCythonizedGraSPI.py


