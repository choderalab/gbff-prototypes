/* example.i */

%include "typemaps.i"
%include "factory.i"
%include "std_string.i"
%include "std_vector.i"

namespace std {
  %template(vectord) vector<double>;
  %template(vectorddd) vector< vector< vector<double> > >;
  %template(vectori) vector<int>;
  %template(vectorii) vector < vector<int> >;
  %template(vectorstring) vector<string>;
};

%module gbvi
%{
     /* Put header files here or function declarations like below */
     extern void addMolecules( char* fileName );
     extern double getGBVIEnergyFromFile( int moleculeIndex, char* fileName );
     extern  double getGBVIEnergy( int moleculeIndex, double soluteDielectric, const std::vector<double> radiiVector, const std::vector<double> gammaVector );
%}
 
 extern void addMolecules( char* fileName );
 extern double getGBVIEnergyFromFile( int moleculeIndex, char* fileName );
 extern  double getGBVIEnergy( int moleculeIndex, double soluteDielectric, const std::vector<double> radiiVector, const std::vector<double> gammaVector );
