
The script parameterize-gbvi_2.py combines the earlier scripts into one script
in an effort to avoid code duplication that can hamper maintenance.

The script also uses SWIG-wrapped code to calculate the GB/VI solute-solvent
interaction energy; the associated calls to OpenMM are not used.
The setup call for the wrapper code is  gbvi.addMolecules() in the Python method
writeGBVIParameterFile() The call, gbvi.getGBVIEnergy(), is made in the 
Python method compute_gbvi_energy() to get the energy.

Quick overview of flow associated w/ wrapper-code:

A file w/ the static info for all the molecules is
output by the Python method writeGBVIParameterFile(). 
The wrapper code gbvi.addMolecules() reads this file and save
the info in various data structures.  
A sample of the format of the file generated in writeGBVIParameterFile() 
is given below for methane; the file will contain similar entries for all
molecules obtained from the sdf file:

        name methane
        atoms  5
            0   C  -1.0867772e-01    4    4    -2.9504299e-07  -1.0309815e-05 2.8741360e-06
            1   H   2.7365677e-02    7    8     5.5406272e-02   7.9960823e-02 4.9648654e-02
            2   H   2.7326990e-02    7    8     6.8328631e-02  -8.1335044e-02 -2.5374085e-02
            3   H   2.7000926e-02    7    8    -7.7822578e-02  -3.7350243e-02 6.6930377e-02
            4   H   2.6984129e-02    7    8    -4.5912036e-02   3.8734773e-02 -9.1207826e-02
        
Atom entries:
    column 1.   index
    column 2.   atom type (C, H, ...)
    column 3.   atom charge
    column 4.   radius index
    column 5.   gamma index
    column 6.   x-coordinate of the atom
    column 7.   y-coordinate     "
    column 8.   z-coordinate     "

The radius and gamma indices are used to get the radius and gamma parameter
values when calculating the GB/VI energy. For each energy calculation, a molecule index and
radii and gamma arrays containing the current radii and gamma value are passed to the the wrapper routine
gbvi.getGBVIEnergy(). gbvi.getGBVIEnergy() pulls the radii/gamma values out of
the input arrays using the index values passed in the setup file and calculates/returns the GB/VI energy.


Setup of SWIG code:

# Compile gbvi SWIG wrapper code

/bin/rm -rf gbvi_wrap.cxx  gbvi.py gbvi.o gbvi_wrap _gbvi.o
g++ -I. gbvi.cpp -o gbvi
swig -v -c++ -python gbvi.i
g++ -c gbvi.cpp gbvi_wrap.cxx -DSWIG -fPIC -I/home/friedrim/source/python/include/python2.6
g++ -shared gbvi.o gbvi_wrap.o -o _gbvi.so

JDC: On Mac: 

g++ -I. gbvi.cpp -o gbvi
swig -v -c++ -python gbvi.i
g++ -c gbvi.cpp gbvi_wrap.cxx -DSWIG -fPIC -I/Library/Frameworks/EPD64.framework/Versions/Current/include/python2.7/
g++ -dynamiclib -undefined suppress -flat_namespace gbvi.o gbvi_wrap.o -o _gbvi.so

Note: _gbvi.so should be in the run directory.




# Sample script run (short description of arguments below):

/home/friedrim/source/python/bin/python /home/friedrim/source/gbff/examples/gbvi2/parameterize-gbvi_2.py  --types /home/friedrim/source/gbff/examples/gbvi/parameters/gbvi_reduced.types --parameters /home/friedrim/source/gbff/examples/gbvi/parameters/gbvi-am1bcc_reduced.parameters  --molecules /home/friedrim/source/gbff/examples/gbvi/datasets/neutrals.sdf  --mcmcDb /home/friedrim/source/gbff/examples/gbvi/results/MCMCReduced_0.80_30  --burnIn 100  --iterations 800  --includeSoluteDielectricAsParameter 1  --train_fraction 0.8  --seed 30  --performFit 1  --allMolecules 0  --analyzeParameters 0 >& /home/friedrim/source/gbff/examples/gbvi/results/zParameterizeReduced_0.80_30


python ~/examples/gbvi2/parameterize-gbvi_2.py
   --types      /home/friedrim/source/gbff/examples/gbvi/parameters/gbvi_reduced.types 
   --parameters /home/friedrim/source/gbff/examples/gbvi/parameters/gbvi-am1bcc_reduced.parameters  
   --molecules  /home/friedrim/source/gbff/examples/gbvi/datasets/neutrals.sdf  
   --mcmcDb     /home/friedrim/source/gbff/examples/gbvi/results/MCMCReduced_0.80_30  
   --burnIn 1000                                          # number of burn-in iterations
   --iterations 10000                                     # number of MCMC iterations
   --includeSoluteDielectricAsParameter 1                 # if set, include solute dielectric as parameter
   --train_fraction 0.8                                   # fraction of molecules to be used in training set
   --seed 30                                              # random number generator seed for partitioning molecules between training/test sets
   --performFit 1                                         # if nonzero, then perform MCMC fit; if zero, skip fitting (used for analyzing previous runs)
   --allMolecules 0                                       # if value is <= 0, then all molecules in sdf file are used. If allMolecules=x and x > 0, then only first 
                                                          # x molecules are used; this is useful for debugging/testing new code when you don't need to
                                                          # run through all molecules
   --analyzeParameters 0                                  # analyze parameters (calculate means, variances, ...) from previous fits/runs
                                                          # if set
---

John Chodera notes 2012-07-23

For last step on OS X:

swig -v -c++ -python gbvi.i
g++ -arch x86_64 -arch i386 -I. gbvi.cpp -o gbvi
g++ -arch x86_64 -c gbvi.cpp gbvi_wrap.cxx -DSWIG -fPIC -I/Library/Frameworks/EPD64.framework/Versions/Current/include/python2.7/
ld -arch x86_64 -arch i386 -bundle -flat_namespace -undefined suppress -o _gbvi.so gbvi.o gbvi_wrap.o

python2.7 parameterize-gbvi_2.py --types ../gbvi/parameters/gbvi.types --parameters ../gbvi/parameters/gbvi-am1bcc.parameters --molecules ../gbvi/datasets/solvation.sdf --iterations 150 --mcmcDb MCMC.txt

