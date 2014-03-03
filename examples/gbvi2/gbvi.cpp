
/* Portions copyright (c) 2006-2009 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 *
 * Calculate GB/VI energy for molecule(s)
 *
 * Intended for use w/ Python scripts
 *
 * code can be Swigged for Python 
 *
 *     swig -v -c++ -python gbvi.i
 *     g++ -c gbvi.cpp gbvi_wrap.cxx -DSWIG -fPIC -I{Python include directory: ../python/include/python2.6}
 *     g++ -shared gbvi.o gbvi_wrap.o -o _gbvi.so
 *
 * Use case:
 *
 *  Python:
 *
 *     read in mol file w/ atoms/bonds via call to addMolecules( molFileName )
 *
 *         the info for each molecule is stored in the global vector:
               std::vector<GbviMolecule> globalMolecules;
 *
 *     call getGBVIEnergy( moleculeIndex, soluteDielectric, radiiVector, gammaVector )
 *          to get the GB/VI energy for the molecule e/ index=moleculeIndex in the input mol file
 *          soluteDielectric, radiiVector, gammaVector are the parameters to be used in the calculations
 *
 *          fixed parameters (solvent dielectric, ...) are stored in  the global instance of GbviParameters, globalGbviParameters
 * Standalone
 *
 *      g++ -I. gbvi.cpp -o gbvi
 *      gbvi -parameterFile gbviParameterFile.txt -deltaParameterFile deltaParameterFile.txt
 *
 *      see comments above main() for more details of standalone use case
 *
 */

#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <vector>
#include <map>
#include <sstream>
#include <fstream>

using namespace std;

// use double precision

#define RealOpenMM     double

#define SQRT           sqrt
#define POW            pow
#define EXP            exp
#define FABS           fabs

// shared indicies for delta[x/y/z/] 

static const int XIndex             = 0;
static const int YIndex             = 1;
static const int ZIndex             = 2;
static const int R2Index            = 3;
static const int RIndex             = 4;
static const int LastDeltaRIndex    = 5;

// the following typedefs are used in parsing parameter files

typedef std::vector<std::string> StringVector;
typedef StringVector::iterator StringVectorI;
typedef StringVector::const_iterator StringVectorCI;

typedef std::vector<StringVector> StringVectorVector;

typedef std::map< std::string, std::string > MapStringString;
typedef MapStringString::iterator MapStringStringI;
typedef MapStringString::const_iterator MapStringStringCI;

// copy of OpenMMVec3

#ifndef OPENMM_VEC3_H_
#define OPENMM_VEC3_H_

#include <cassert>

/**
 * This class represents a three component vector.  It is used for storing positions,
 * velocities, and forces.
 */

class OpenMMVec3 {

public:
    /**
     * Create a OpenMMVec3 whose elements are all 0.
     */
    OpenMMVec3() {
        data[0] = data[1] = data[2] = 0.0;
    }
    /**
     * Create a OpenMMVec3 with specified x, y, and z components.
     */
    OpenMMVec3(double x, double y, double z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double operator[](int index) const {
        assert(index >= 0 && index < 3);
        return data[index];
    }
    double& operator[](int index) {
        assert(index >= 0 && index < 3);
        return data[index];
    }
    
    // Arithmetic operators
    
    // unary plus
    OpenMMVec3 operator+() const {
        return OpenMMVec3(*this);
    }
    
    // plus
    OpenMMVec3 operator+(const OpenMMVec3& rhs) const {
        const OpenMMVec3& lhs = *this;
        return OpenMMVec3(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]);
    }
    
    OpenMMVec3& operator+=(const OpenMMVec3& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        data[2] += rhs[2];
        return *this;
    }

    // unary minus
    OpenMMVec3 operator-() const {
        const OpenMMVec3& lhs = *this;
        return OpenMMVec3(-lhs[0], -lhs[1], -lhs[2]);
    }
    
    // minus
    OpenMMVec3 operator-(const OpenMMVec3& rhs) const {
        const OpenMMVec3& lhs = *this;
        return OpenMMVec3(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]);
    }

    OpenMMVec3& operator-=(const OpenMMVec3& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        data[2] -= rhs[2];
        return *this;
    }

    // scalar product
    OpenMMVec3 operator*(double rhs) const {
        const OpenMMVec3& lhs = *this;
        return OpenMMVec3(lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs);
    }

    OpenMMVec3& operator*=(double rhs) {
        data[0] *= rhs;
        data[1] *= rhs;
        data[2] *= rhs;
        return *this;
    }
    
    // dot product
    double dot(const OpenMMVec3& rhs) const {
        const OpenMMVec3& lhs = *this;
        return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
    }

    // cross product
    OpenMMVec3 cross(const OpenMMVec3& rhs) const {
        return OpenMMVec3(data[1]*rhs[2]-data[2]*rhs[1], data[2]*rhs[0]-data[0]*rhs[2], data[0]*rhs[1]-data[1]*rhs[0]);
    }
    
private:
    double data[3];
};

template <class CHAR, class TRAITS>
std::basic_ostream<CHAR,TRAITS>& operator<<(std::basic_ostream<CHAR,TRAITS>& o, const OpenMMVec3& v) {
    o<<'['<<v[0]<<", "<<v[1]<<", "<<v[2]<<']';
    return o;
}

#endif /*OPENMM_VEC3_H_*/

/**---------------------------------------------------------------------------------------

    Open file

    @param fileName             file name
    @param mode                 file mode: "r", "w", "a"
    @param log                  optional logging file reference

    @return file pttr or NULL if file not opened

    --------------------------------------------------------------------------------------- */

static FILE* openFile( const std::string& fileName, const std::string& mode, FILE* log ){

    // ---------------------------------------------------------------------------------------

    FILE* filePtr;

#ifdef _MSC_VER
    fopen_s( &filePtr, fileName.c_str(), mode.c_str() );
#else
    filePtr = fopen( fileName.c_str(), mode.c_str() );
#endif

    if( log ){
        (void) fprintf( log, "openFile: file=<%s> %sopened w/ mode=%s.\n", fileName.c_str(), (filePtr == NULL ? "not " : ""), mode.c_str() );
        (void) fflush( log );
    }    
    return filePtr;
}

/**---------------------------------------------------------------------------------------

   Report whether a number is a nan or infinity

   @param number               number to test
   @return 1 if number is  nan or infinity; else return 0

   --------------------------------------------------------------------------------------- */

int isNan( double number ){
    return (number != number || 
            number ==  std::numeric_limits<double>::infinity() || 
            number == -std::numeric_limits<double>::infinity()) ? 1 : 0; 
}

/**---------------------------------------------------------------------------------------
 *
 * Set string field if in map
 * 
 * @param  argumentMap            map to check
 * @param  fieldToCheck           key
 * @param  fieldToSet             field to set
 *
 * @return 1 if argument set, else 0
 *
   --------------------------------------------------------------------------------------- */

static int setStringFromMap( MapStringString& argumentMap, std::string fieldToCheck, std::string& fieldToSet ){

// ---------------------------------------------------------------------------------------

   MapStringStringCI check = argumentMap.find( fieldToCheck );
   if( check != argumentMap.end() ){
      fieldToSet = check->second; 
      return 1;
   }
   return 0;
}

/**---------------------------------------------------------------------------------------
 *
 * Set int field if in map
 * 
 * @param  argumentMap            map to check
 * @param  fieldToCheck           key
 * @param  fieldToSet             field to set
 *
 * @return 1 if argument set, else 0
 *
   --------------------------------------------------------------------------------------- */

static int setIntFromMap( MapStringString& argumentMap, std::string fieldToCheck, int& fieldToSet ){

// ---------------------------------------------------------------------------------------

   MapStringStringCI check = argumentMap.find( fieldToCheck );
   if( check != argumentMap.end() ){
      fieldToSet = atoi( check->second.c_str() ); 
      return 1;
   }
   return 0;
}

/**---------------------------------------------------------------------------------------

 * Set float field if in map
 * 
 * @param  argumentMap            map to check
 * @param  fieldToCheck           key
 * @param  fieldToSet             field to set
 *
 * @return 1 if argument set, else 0
 *
   --------------------------------------------------------------------------------------- */

static int setFloatFromMap( MapStringString& argumentMap, std::string fieldToCheck, float& fieldToSet ){

// ---------------------------------------------------------------------------------------

   static const std::string methodName             = "setFloatFromMap";

// ---------------------------------------------------------------------------------------

   MapStringStringCI check = argumentMap.find( fieldToCheck );
   if( check != argumentMap.end() ){
      fieldToSet = static_cast<float>(atof( check->second.c_str() )); 
      return 1;
   }
   return 0;
}

/**---------------------------------------------------------------------------------------
 *
 * Set double field if in map
 * 
 * @param  argumentMap            map to check
 * @param  fieldToCheck           key
 * @param  fieldToSet             field to set
 *
 * @return 1 if argument set, else 0
 *
   --------------------------------------------------------------------------------------- */

static int setDoubleFromMap( MapStringString& argumentMap, std::string fieldToCheck, double& fieldToSet ){

// ---------------------------------------------------------------------------------------

   static const std::string methodName             = "setDoubleFromMap";

// ---------------------------------------------------------------------------------------

   MapStringStringCI check = argumentMap.find( fieldToCheck );
   if( check != argumentMap.end() ){
      fieldToSet = atof( check->second.c_str() ); 
      return 1;
   }
   return 0;
}

/**---------------------------------------------------------------------------------------

 * Append command-line arguments to argumentMap
 * 
 * @param  numberOfArguments      number of arguments (argc)
 * @param  argv                   arguments
 * @param  argumentMap            map to add arguments to
 *
 *
   --------------------------------------------------------------------------------------- */

static void appendInputArgumentsToArgumentMap( int numberOfArguments, char* argv[], MapStringString& argumentMap ){

// ---------------------------------------------------------------------------------------

    for( int ii = 1; ii < numberOfArguments; ii += 2 ){ 
        char* key        = argv[ii];
        if( *key == '-' )key++;
        argumentMap[key] = (ii+1) < numberOfArguments ? argv[ii+1] : "NA";
    }    

    return;
}
  
/**---------------------------------------------------------------------------------------

   Replacement of sorts for strtok()
   Used to parse parameter file lines

   @param lineBuffer           string to tokenize
   @param delimiter            token delimter

   @return number of args

   --------------------------------------------------------------------------------------- */

static char* strsepLocal( char** lineBuffer, const char* delimiter ){

   // ---------------------------------------------------------------------------------------

   char *s;
   const char *spanp;
   int c, sc; 
   char *tok;
   
   // ---------------------------------------------------------------------------------------

   s = *lineBuffer;
   if( s == NULL ){
      return (NULL);
   }
   
   for( tok = s;; ){
      c     = *s++;
      spanp = delimiter;
      do {
         if( (sc = *spanp++) == c ){
            if( c == 0 ){
               s = NULL;
            } else {
               s[-1] = 0;
            }
            *lineBuffer = s;
            return( tok );
         }
      } while( sc != 0 );
   }
}  

/**---------------------------------------------------------------------------------------

   Tokenize a string

   @param lineBuffer           string to tokenize
   @param tokenArray           upon return vector of tokens
   @param delimiter            token delimter

   @return number of tokens

   --------------------------------------------------------------------------------------- */

static int tokenizeString( char* lineBuffer, StringVector& tokenArray, const std::string delimiter ){

   // ---------------------------------------------------------------------------------------

   char *ptr_c = NULL;

   for( ; (ptr_c = strsepLocal( &lineBuffer, delimiter.c_str() )) != NULL; ){
      if( *ptr_c ){
         tokenArray.push_back( std::string( ptr_c ) );
      }
   }

   return (static_cast<int>(tokenArray.size()));
}

/**---------------------------------------------------------------------------------------

    Read a line from a file and tokenize into an array of strings

    @param filePtr              file to read from
    @param tokens               array of token strings
    @param lineCount            line count
    @param log                  optional file ptr for logging

    @return ptr to string containing line

    --------------------------------------------------------------------------------------- */

static int readLine( FILE* filePtr, StringVector& tokens, int* lineCount, FILE* log ){

// ---------------------------------------------------------------------------------------

    std::string delimiter                    = " \r\n";
    const int bufferSize                     = 4096;
    char buffer[bufferSize];

// ---------------------------------------------------------------------------------------

    char* isNotEof = fgets( buffer, bufferSize, filePtr );
    if( isNotEof ){
       (*lineCount)++;
       tokenizeString( buffer, tokens, delimiter );
       return 1;
    } else {
       return 0;
    }    

}

/**---------------------------------------------------------------------------------------

   Read a line from a file and tokenize into an array of strings

   @param filePtr              file to read from
   @param tokens               array of token strings
   @param lineCount            line count
   @param log                  optional file ptr for logging

   @return ptr to string containing line

   --------------------------------------------------------------------------------------- */

static char* readLineFromFile( FILE* filePtr, StringVector& tokens ){

// ---------------------------------------------------------------------------------------

   std::string delimiter                    = " \n";
   const int bufferSize                     = 4096;
   char buffer[bufferSize];

// ---------------------------------------------------------------------------------------

   char* isNotEof = fgets( buffer, bufferSize, filePtr );
   if( isNotEof ){
      tokenizeString( buffer, tokens, delimiter );
   }
   return isNotEof;

}

/**---------------------------------------------------------------------------------------

   Read a file

   @param fileName             file name 
   @param fileContents         output file contents as a vector of string vectors:

                               fileContents[line] = string vector of tokens

   --------------------------------------------------------------------------------------- */

static void readFileLineByLine( std::string fileName, StringVectorVector& fileContents ){

// ---------------------------------------------------------------------------------------

    // initialize fileContents vector and open file

    fileContents.resize(0);
    FILE* filePtr = fopen( fileName.c_str(), "r" );
    if( filePtr ){
        StringVector firstLine;
        char* isNotEof = readLineFromFile( filePtr, firstLine);
        fileContents.push_back( firstLine );
        while( isNotEof ){
            StringVector lineTokens;
            isNotEof = readLineFromFile( filePtr, lineTokens );
            fileContents.push_back( lineTokens );
        }
        (void) fclose( filePtr );
    } else {
        (void) fprintf( stderr, "Could not open file=<%s> for reading.\n", fileName.c_str() );
        (void) fflush( stderr );
    }

    return;
}

/**---------------------------------------------------------------------------------------

   Read positions from file

   @param fileName             file name
                               file header:
                                   'Positions 340' or '340 Positions' if 340 particles positions in
                                   file
   @param positions            output vector of position coordinates: 

   --------------------------------------------------------------------------------------- */

static void readPositionsFromFile( const std::string& fileName, std::vector<OpenMMVec3>& positions ){

    StringVectorVector fileContents;
    readFileLineByLine( fileName, fileContents );
    int readCoordinates     = -1;
    int numberOfCoordinates = 0;
    for( unsigned int ii = 0; ii < fileContents.size(); ii++ ){
        if( fileContents[ii].size() < 2 )continue;

        // 'Positions 340' or '340 Positions'

        if( strcmp( "Positions", fileContents[ii][0].c_str() ) == 0 ){
            readCoordinates      = ii + 1;
            numberOfCoordinates  = atoi( fileContents[ii][1].c_str() );
            break;        
        } else if( strcmp( "Positions", fileContents[ii][1].c_str() ) == 0 ){
            readCoordinates      = ii + 1;
            numberOfCoordinates  = atoi( fileContents[ii][0].c_str() );
            break;        
        }
    }

    //(void) fprintf( stderr, "readCoordinates=%d numberOfCoordinates=%d\n", readCoordinates, numberOfCoordinates );

    if( readCoordinates > 0 && readCoordinates < fileContents.size() && numberOfCoordinates > 0 ){
        for( unsigned int ii = readCoordinates; ii < fileContents.size() && positions.size() < numberOfCoordinates; ii++ ){
            double xValue = atof( fileContents[ii][1].c_str() );
            double yValue = atof( fileContents[ii][2].c_str() );
            double zValue = atof( fileContents[ii][3].c_str() );

            OpenMMVec3 values;
            values[0]     = xValue;
            values[1]     = yValue;
            values[2]     = zValue;

            positions.push_back( values );
        }
    }

    return;
}

/**---------------------------------------------------------------------------------------

   Compute quintic spline value and associated derviative

   @param x                   value to compute spline at
   @param rl                  lower cutoff value
   @param ru                  upper cutoff value
   @param outValue            value of spline at x
   @param outDerivative       value of derivative of spline at x

   --------------------------------------------------------------------------------------- */

static void quinticSpline( RealOpenMM x, RealOpenMM rl, RealOpenMM ru, RealOpenMM* outValue, RealOpenMM* outDerivative ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM one           = static_cast<RealOpenMM>(   1.0 );
   static const RealOpenMM minusSix      = static_cast<RealOpenMM>(  -6.0 );
   static const RealOpenMM minusTen      = static_cast<RealOpenMM>( -10.0 );
   static const RealOpenMM minusThirty   = static_cast<RealOpenMM>( -30.0 );
   static const RealOpenMM fifteen       = static_cast<RealOpenMM>(  15.0 );
   static const RealOpenMM sixty         = static_cast<RealOpenMM>(  60.0 );

   // ---------------------------------------------------------------------------------------

   RealOpenMM numerator    = x  - rl;
   RealOpenMM denominator  = ru - rl;
   RealOpenMM ratio        = numerator/denominator;
   RealOpenMM ratio2       = ratio*ratio;
   RealOpenMM ratio3       = ratio2*ratio;

   *outValue               = one + ratio3*(minusTen + fifteen*ratio + minusSix*ratio2);
   *outDerivative          = ratio2*(minusThirty + sixty*ratio + minusThirty*ratio2)/denominator;
}

/**---------------------------------------------------------------------------------------

   Compute Born radii based on Eq. 3 of Labute paper [JCC 29 p. 1693-1698 2008])
   and quintic splice switching function

   @param atomicRadius3       atomic radius cubed
   @param bornSum             Born sum (volume integral)
   @param bornRadius          output Born radius
   @param switchDeriviative   output switching function deriviative

   --------------------------------------------------------------------------------------- */

static void computeBornRadiiUsingQuinticSpline( RealOpenMM atomicRadius3, RealOpenMM bornSum,
                                                RealOpenMM quinticLowerLimitFactor,
                                                RealOpenMM quinticUpperBornRadiusLimit,
                                                RealOpenMM* bornRadius, RealOpenMM* switchDeriviative ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM zero          = static_cast<RealOpenMM>(  0.0 );
   static const RealOpenMM one           = static_cast<RealOpenMM>(  1.0 );
   static const RealOpenMM minusOne      = static_cast<RealOpenMM>( -1.0 );
   static const RealOpenMM minusThree    = static_cast<RealOpenMM>( -3.0 );
   static const RealOpenMM oneEighth     = static_cast<RealOpenMM>(  0.125 );
   static const RealOpenMM minusOneThird = static_cast<RealOpenMM>( (-1.0/3.0) );
   static const RealOpenMM three         = static_cast<RealOpenMM>(  3.0 );

   static const char* methodName         = "computeBornRadiiUsingQuinticSpline";

   // ---------------------------------------------------------------------------------------

   // R                = [ S(V)*(A - V) ]**(-1/3)

   // S(V)             = 1                                 V < L
   // S(V)             = qSpline + U/(A-V)                 L < V < A
   // S(V)             = U/(A-V)                           U < V 

   // dR/dr            = (-1/3)*[ S(V)*(A - V) ]**(-4/3)*[ d{ S(V)*(A-V) }/dr

   // d{ S(V)*(A-V) }/dr   = (dV/dr)*[ (A-V)*dS/dV - S(V) ]

   //  (A - V)*dS/dV - S(V)  = 0 - 1                             V < L

   //  (A - V)*dS/dV - S(V)  = (A-V)*d(qSpline) + (A-V)*U/(A-V)**2 - qSpline - U/(A-V) 

	//                        = (A-V)*d(qSpline) - qSpline        L < V < A**(-3)

   //  (A - V)*dS/dV - S(V)  = (A-V)*U*/(A-V)**2 - U/(A-V) = 0   U < V

   RealOpenMM sum;
   if( bornSum > atomicRadius3*quinticLowerLimitFactor ){
      if( bornSum < atomicRadius3 ){
         RealOpenMM splineValue, splineDerivative;
         quinticSpline( bornSum, quinticLowerLimitFactor, atomicRadius3, &splineValue, &splineDerivative ); 
         sum                 = (atomicRadius3 - bornSum)*splineValue + quinticUpperBornRadiusLimit;
         *switchDeriviative  = splineValue - (atomicRadius3 - bornSum)*splineDerivative;
      } else {   
         sum                = quinticUpperBornRadiusLimit;
         *switchDeriviative = zero;
      }
   } else {
      sum                = atomicRadius3 - bornSum; 
      *switchDeriviative = one;
   }
   *bornRadius = POW( sum, minusOneThird );
}

/**---------------------------------------------------------------------------------------

   Get deltaR and distance and distance**2 between atomI and atomJ (static method)
   deltaR: j - i

   @param atomCoordinatesI    atom i coordinates
   @param atomCoordinatesI    atom j coordinates
   @]param deltaR              deltaX, deltaY, deltaZ, R2, R upon return

   --------------------------------------------------------------------------------------- */

void getDeltaR( const OpenMMVec3& atomCoordinatesI, const OpenMMVec3& atomCoordinatesJ, RealOpenMM* deltaR ){

   // ---------------------------------------------------------------------------------------

   deltaR[XIndex]    = atomCoordinatesJ[0] - atomCoordinatesI[0];
   deltaR[YIndex]    = atomCoordinatesJ[1] - atomCoordinatesI[1];
   deltaR[ZIndex]    = atomCoordinatesJ[2] - atomCoordinatesI[2];

   deltaR[R2Index]   = deltaR[0]*deltaR[0] + deltaR[1]*deltaR[1] + deltaR[2]*deltaR[2];
   deltaR[RIndex]    = static_cast<RealOpenMM>( sqrt( deltaR[R2Index] ) );

}

/**---------------------------------------------------------------------------------------

   Get L (used in analytical solution for volume integrals) 

   @param r                   distance between atoms i & j
   @param R                   atomic radius
   @param S                   scaled atomic radius

   @return L value (Eq. 4 of Labute paper [JCC 29 p. 1693-1698 2008])

   --------------------------------------------------------------------------------------- */

RealOpenMM getL( RealOpenMM r, RealOpenMM x, RealOpenMM S ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM one           = static_cast<RealOpenMM>( 1.0 );
   static const RealOpenMM threeHalves   = static_cast<RealOpenMM>( 1.5 );
   static const RealOpenMM third         = static_cast<RealOpenMM>( (1.0/3.0) );
   static const RealOpenMM fourth        = static_cast<RealOpenMM>( 0.25 );
   static const RealOpenMM eighth        = static_cast<RealOpenMM>( 0.125 );

   // ---------------------------------------------------------------------------------------

   RealOpenMM rInv   = one/r;

   RealOpenMM xInv   = one/x;
   RealOpenMM xInv2  = xInv*xInv;
   RealOpenMM xInv3  = xInv2*xInv;

   RealOpenMM diff2  = (r + S)*(r - S);

   return (threeHalves*xInv)*( (xInv*fourth*rInv) - (xInv2*third) + (diff2*xInv3*eighth*rInv) );
}

/**---------------------------------------------------------------------------------------

   Get volume Eq. 4 of Labute paper [JCC 29 p. 1693-1698 2008])

   @param r                   distance between atoms i & j
   @param R                   atomic radius
   @param S                   scaled atomic radius

   @return volume

   --------------------------------------------------------------------------------------- */

RealOpenMM getVolume( RealOpenMM r, RealOpenMM R, RealOpenMM S ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM zero         = static_cast<RealOpenMM>(  0.0 );
   static const RealOpenMM minusThree   = static_cast<RealOpenMM>( -3.0 );

   RealOpenMM              diff         = (S - R);
   if( FABS( diff ) < r ){

      RealOpenMM lowerBound = (R > (r - S)) ? R : (r - S);
      return (getL( r, (r + S),    S ) -
              getL( r, lowerBound, S ));

   } else if( r <= diff ){

      return getL( r, (r + S), S ) -
             getL( r, (r - S), S ) + 
             POW( R, minusThree );

   } else {
      return zero;
   }
}

/**---------------------------------------------------------------------------------------

   Get partial derivative of L wrt r

   @param r                   distance between atoms i & j
   @param R                   atomic radius
   @param S                   scaled atomic radius

   @return partial derivative based on Eq. 4 of Labute paper [JCC 29 p. 1693-1698 2008])

   --------------------------------------------------------------------------------------- */

RealOpenMM dL_dr( RealOpenMM r, RealOpenMM x, RealOpenMM S ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM one           = static_cast<RealOpenMM>( 1.0 );
   static const RealOpenMM threeHalves   = static_cast<RealOpenMM>( 1.5 );
   static const RealOpenMM threeEights   = static_cast<RealOpenMM>( 0.375 );
   static const RealOpenMM third         = static_cast<RealOpenMM>( (1.0/3.0) );
   static const RealOpenMM fourth        = static_cast<RealOpenMM>( 0.25 );
   static const RealOpenMM eighth        = static_cast<RealOpenMM>( 0.125 );

   // ---------------------------------------------------------------------------------------

   RealOpenMM rInv   = one/r;
   RealOpenMM rInv2  = rInv*rInv;

   RealOpenMM xInv   = one/x;
   RealOpenMM xInv2  = xInv*xInv;
   RealOpenMM xInv3  = xInv2*xInv;

   RealOpenMM diff2  = (r + S)*(r - S);

   return ( (-threeHalves*xInv2*rInv2)*( fourth + eighth*diff2*xInv2 ) + threeEights*xInv3*xInv );
}

/**---------------------------------------------------------------------------------------

   Get partial derivative of L wrt x

   @param r                   distance between atoms i & j
   @param R                   atomic radius
   @param S                   scaled atomic radius

   @return partial derivative based on Eq. 4 of Labute paper [JCC 29 p. 1693-1698 2008])

   --------------------------------------------------------------------------------------- */

RealOpenMM dL_dx( RealOpenMM r, RealOpenMM x, RealOpenMM S ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM one           = static_cast<RealOpenMM>(  1.0 );
   static const RealOpenMM half          = static_cast<RealOpenMM>(  0.5 );
   static const RealOpenMM threeHalvesM  = static_cast<RealOpenMM>( -1.5 );
   static const RealOpenMM third         = static_cast<RealOpenMM>(  (1.0/3.0) );

   // ---------------------------------------------------------------------------------------

   RealOpenMM rInv   = one/r;

   RealOpenMM xInv   = one/x;
   RealOpenMM xInv2  = xInv*xInv;
   RealOpenMM xInv3  = xInv2*xInv;

   RealOpenMM diff   = (r + S)*(r - S);

   return (threeHalvesM*xInv3)*( (half*rInv) - xInv + (half*diff*xInv2*rInv) );
}

/**---------------------------------------------------------------------------------------

   Sgb function

   @param t                   r*r*G_i*G_j

   @return Sgb (top of p. 1694 of Labute paper [JCC 29 p. 1693-1698 2008])

   --------------------------------------------------------------------------------------- */

RealOpenMM Sgb( RealOpenMM t ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM zero    = static_cast<RealOpenMM>( 0.0 );
   static const RealOpenMM one     = static_cast<RealOpenMM>( 1.0 );
   static const RealOpenMM fourth  = static_cast<RealOpenMM>( 0.25 );

   // ---------------------------------------------------------------------------------------

   return ( (t != zero) ? one/SQRT( (one + (fourth*EXP( -t ))/t) ) : zero);
}

/**---------------------------------------------------------------------------------------

   Get Born radii based on Eq. 3 of Labute paper [JCC 29 p. 1693-1698 2008])

   @param atomCoordinates     atomic coordinates
   @param bornRadii           output array of Born radii
   @param chain               not used here

   --------------------------------------------------------------------------------------- */

int computeBornRadii( const std::vector<OpenMMVec3>& atomCoordinates, 
                       const std::vector<RealOpenMM>& atomicRadii,
                       const std::vector<RealOpenMM>& scaledRadii,
                       RealOpenMM quinticLowerLimitFactor,
                       RealOpenMM quinticUpperBornRadiusLimit,
                       std::vector<RealOpenMM>& bornRadii ){

   // ---------------------------------------------------------------------------------------

   static const RealOpenMM zero          = static_cast<RealOpenMM>( 0.0 );
   static const RealOpenMM one           = static_cast<RealOpenMM>( 1.0 );
   static const RealOpenMM minusThree    = static_cast<RealOpenMM>( -3.0 );
   static const RealOpenMM oneEighth     = static_cast<RealOpenMM>( 0.125 );
   static const RealOpenMM minusOneThird = static_cast<RealOpenMM>( (-1.0/3.0) );
   static const RealOpenMM three         = static_cast<RealOpenMM>( 3.0 );

   static const char* methodName         = "computeBornRadii";

   // ---------------------------------------------------------------------------------------

   unsigned int numberOfAtoms                       = atomCoordinates.size();
   //std::vector<RealOpenMM>& switchDeriviatives      = getSwitchDeriviative();

   // ---------------------------------------------------------------------------------------

   FILE* logFile                         = stderr;

   // calculate Born radii

   int nansDetected                      = 0;
   for( unsigned int atomI = 0; atomI < numberOfAtoms; atomI++ ){
     
      RealOpenMM radiusI         = atomicRadii[atomI];
      RealOpenMM sum             = zero;

      // sum over volumes

      for( unsigned int atomJ = 0; atomJ < numberOfAtoms; atomJ++ ){

         if( atomJ != atomI ){

            RealOpenMM deltaR[LastDeltaRIndex];
            getDeltaR( atomCoordinates[atomI], atomCoordinates[atomJ], deltaR );

            RealOpenMM r               = deltaR[RIndex];

            sum                       += getVolume( r, radiusI, scaledRadii[atomJ] );
if( 0 ){ 
            (void) fprintf( logFile, "%d addJ=%d scR=%14.6e %14.6e sum=%14.6e rI=%14.6e r=%14.6e S-R=%14.6e\n",
                            atomI, atomJ, scaledRadii[atomJ], getVolume( r, radiusI, scaledRadii[atomJ] ), sum, 
                            radiusI, r, (scaledRadii[atomJ]-radiusI) );
}

         }
      }

//      (void) fprintf( logFile, "%d Born radius sum=%14.6e %14.6e %14.6e ", 
//                      atomI, sum, POW( radiusI, minusThree ), (POW( radiusI, minusThree ) - sum) );

      RealOpenMM atomicRadius3   = POW( radiusI, minusThree );
      RealOpenMM bornRadius;
#ifdef APPLY_SWITCH
      RealOpenMM switchDeriviative;
      computeBornRadiiUsingQuinticSpline( atomicRadius3, sum, quinticLowerLimitFactor, quinticUpperBornRadiusLimit, &bornRadius, &switchDeriviative );
      bornRadii[atomI]           = bornRadius;
      //switchDeriviatives[atomI]  = switchDeriviative;
#else

      sum                        = atomicRadius3 - sum;
      if( sum <= 0.0 ){
          //(void) fprintf( logFile, "%4d aR=%14.6e sum=%15.7e diff=%15.7e\n", atomI, atomicRadius3, atomicRadius3 - sum, sum );
          nansDetected++;
      } else {
          bornRadii[atomI]      = POW( sum, minusOneThird );
      }

      //switchDeriviatives[atomI]  = one;
#endif

//(void) fprintf( logFile, " br=%14.6e\n", atomI, bornRadii[atomI] );

   }

   return nansDetected;

}

/**---------------------------------------------------------------------------------------

   Get GB/VI energy

   @param bornRadii           Born radii
   @param atomCoordinates     atomic coordinates
   @param partialCharges      partial charges

   @return energy

   --------------------------------------------------------------------------------------- */

RealOpenMM computeBornEnergy( const std::vector<RealOpenMM>& bornRadii, 
                              const std::vector<OpenMMVec3>&    atomCoordinates,
                              const std::vector<RealOpenMM>& atomicRadii,
                              RealOpenMM tau,
                              const std::vector<RealOpenMM>& gammaParameters,
                              const std::vector<RealOpenMM>& partialCharges ){

   // ---------------------------------------------------------------------------------------

   static const char* methodName         = "computeBornEnergy";

   static const RealOpenMM zero          = static_cast<RealOpenMM>( 0.0 );
   static const RealOpenMM one           = static_cast<RealOpenMM>( 1.0 );
   static const RealOpenMM two           = static_cast<RealOpenMM>( 2.0 );
   static const RealOpenMM three         = static_cast<RealOpenMM>( 3.0 );
   static const RealOpenMM four          = static_cast<RealOpenMM>( 4.0 );
   static const RealOpenMM half          = static_cast<RealOpenMM>( 0.5 );
   static const RealOpenMM fourth        = static_cast<RealOpenMM>( 0.25 );
   static const RealOpenMM eighth        = static_cast<RealOpenMM>( 0.125 );

   // ---------------------------------------------------------------------------------------

   const RealOpenMM preFactor           = (-0.5*138.935456);
   const unsigned int numberOfAtoms     = atomCoordinates.size();

   // ---------------------------------------------------------------------------------------

   // Eq.2 of Labute paper [JCC 29 p. 1693-1698 2008]
   // to minimze roundoff error sum cavityEnergy separately since in general much
   // smaller than other contributions

   RealOpenMM energy                 = zero;
   RealOpenMM cavityEnergy           = zero;
   RealOpenMM deltaR[LastDeltaRIndex];

   for( unsigned int atomI = 0; atomI < numberOfAtoms; atomI++ ){
 
      RealOpenMM partialChargeI   = partialCharges[atomI];

      // self-energy term

      RealOpenMM  atomIEnergy     = half*partialChargeI/bornRadii[atomI];

      // cavity term

      RealOpenMM ratio            = (atomicRadii[atomI]/bornRadii[atomI]);
      cavityEnergy               += gammaParameters[atomI]*ratio*ratio*ratio;

      for( unsigned int atomJ = atomI + 1; atomJ < numberOfAtoms; atomJ++ ){

         getDeltaR( atomCoordinates[atomI], atomCoordinates[atomJ], deltaR );

         RealOpenMM r2                 = deltaR[R2Index];
         RealOpenMM t                  = fourth*r2/(bornRadii[atomI]*bornRadii[atomJ]);         
         atomIEnergy                  += partialCharges[atomJ]*Sgb( t )/deltaR[RIndex];

      }

      energy += two*partialChargeI*atomIEnergy;
   }
   energy *= preFactor;
   energy -= cavityEnergy;

if( 0 ){
   (void) fprintf( stderr, "computeBornEnergy: ElectricConstant=%.4e Tau=%.4e e=%.5e eOut=%.5e cavity=%12.5e\n", 
                   preFactor, tau, energy, tau*energy, cavityEnergy );
/*
   for( int atomI = 0; atomI < numberOfAtoms; atomI++ ){
      (void) fprintf( stderr, "bR %d bR=%16.8e q=%16.9e aR=%16.9e gam=%16.9e\n",
                      atomI, bornRadii[atomI], partialCharges[atomI], atomicRadii[atomI], gammaParameters[atomI] );
   }
*/
   (void) fflush( stderr );
}


   return (tau*energy);
 
}

class GbviParameters {
    public:

        GbviParameters( );

        void setGbviParameters( RealOpenMM  soluteDielectric, RealOpenMM  solventDielectric, RealOpenMM  quinticLowerLimitFactor, RealOpenMM  quinticUpperBornRadiusLimit );
        void getGbviParameters( RealOpenMM& soluteDielectric, RealOpenMM& solventDielectric, RealOpenMM& quinticLowerLimitFactor, RealOpenMM& quinticUpperBornRadiusLimit );

        std::vector<RealOpenMM>& getGbviGammaVector( void );
        std::vector<RealOpenMM>& getGbviRadiusVector( void );

        StringVector& getGbviGammaAtomTypeVector( void );
        StringVector& getGbviRadiusAtomTypeVector( void );

        void printGbviParameters( FILE* log );

    private:

        RealOpenMM _soluteDielectric;
        RealOpenMM _solventDielectric;

        RealOpenMM _quinticLowerLimitFactor;
        RealOpenMM _quinticUpperBornRadiusLimit;

        std::vector<RealOpenMM> _gammaParameters;
        std::vector<RealOpenMM> _radiusParameters;

        StringVector _gammaAtomTypeParameters;
        StringVector _radiusAtomTypeParameters;
};

GbviParameters::GbviParameters( ){
    _soluteDielectric                = 1.0;
    _solventDielectric               = 78.3;
    _quinticLowerLimitFactor         = 0.8;
    _quinticUpperBornRadiusLimit     = 5.0;
}

void GbviParameters::setGbviParameters( RealOpenMM soluteDielectric, RealOpenMM solventDielectric, RealOpenMM quinticLowerLimitFactor, RealOpenMM quinticUpperBornRadiusLimit ){
    _soluteDielectric                = soluteDielectric;
    _solventDielectric               = solventDielectric;
    _quinticLowerLimitFactor         = quinticLowerLimitFactor;
    _quinticUpperBornRadiusLimit     = quinticUpperBornRadiusLimit;
}

void GbviParameters::printGbviParameters( FILE* log ){

    (void) fprintf( log, "GbviParameters:\n" );

    RealOpenMM soluteDielectric;
    RealOpenMM solventDielectric;
    RealOpenMM quinticLowerLimitFactor;
    RealOpenMM quinticUpperBornRadiusLimit;
    getGbviParameters( soluteDielectric, solventDielectric, quinticLowerLimitFactor, quinticUpperBornRadiusLimit );

    (void) fprintf( log, "    soluteDielectric            %15.7e\n", soluteDielectric  );
    (void) fprintf( log, "    solventDielectric           %15.7e\n", solventDielectric);
    (void) fprintf( log, "    quinticLowerLimitFactor     %15.7e\n", quinticLowerLimitFactor );
    (void) fprintf( log, "    quinticUpperBornRadiusLimit %15.7e\n", quinticUpperBornRadiusLimit );

    std::vector<RealOpenMM>& radiusVector = getGbviRadiusVector();
    StringVector& radiusAtomTypeVector    = getGbviRadiusAtomTypeVector();
    std::vector<RealOpenMM>& gammaVector  = getGbviGammaVector();
    StringVector& gammaAtomTypeVector     = getGbviGammaAtomTypeVector();

    (void) fprintf( log, "Radius & Gamma parameters\n" );
    for( unsigned int ii = 0; ii < radiusVector.size(); ii++ ){
         (void) fprintf( log, "    %3d %3s %15.7e   %3s %15.7e\n", ii, radiusAtomTypeVector[ii].c_str(), radiusVector[ii], gammaAtomTypeVector[ii].c_str(), gammaVector[ii] );
    }
    (void) fflush( log );
}

std::vector<RealOpenMM>& GbviParameters::getGbviGammaVector( void ){
    return _gammaParameters;
}

std::vector<RealOpenMM>& GbviParameters::getGbviRadiusVector( void ){
    return _radiusParameters;
}

StringVector& GbviParameters::getGbviGammaAtomTypeVector( void ){
    return _gammaAtomTypeParameters;
}

StringVector& GbviParameters::getGbviRadiusAtomTypeVector( void ){
    return _radiusAtomTypeParameters;
}

void GbviParameters::getGbviParameters( RealOpenMM& soluteDielectric, RealOpenMM& solventDielectric, RealOpenMM& quinticLowerLimitFactor, RealOpenMM& quinticUpperBornRadiusLimit ){
    soluteDielectric                = _soluteDielectric;
    solventDielectric               = _solventDielectric;
    quinticLowerLimitFactor         = _quinticLowerLimitFactor;
    quinticUpperBornRadiusLimit     = _quinticUpperBornRadiusLimit;
}

GbviParameters globalGbviParameters;

class GbviBond {
    public:
        GbviBond( );
        void setGbviBond( int particle1, int particle2, RealOpenMM bondLength );
        void getGbviBond( int& particle1, int& particle2, RealOpenMM& bondLength );
    private:
        int _particle1;
        int _particle2;
        RealOpenMM _bondLength;
};

GbviBond::GbviBond( ){
    _particle1  = -1;
    _particle2  = -1;
    _bondLength = 0.0;
}

void GbviBond::getGbviBond( int& particle1, int& particle2, RealOpenMM& bondLength ){
    particle1  = _particle1;
    particle2  = _particle2;
    bondLength = _bondLength;
}

void GbviBond::setGbviBond( int particle1, int particle2, RealOpenMM bondLength ){
    _particle1  = particle1;
    _particle2  = particle2;
    _bondLength = bondLength;
}

class GbviMolecule {

    public:

        GbviMolecule();

        void setName( std::string& name );

        void addAtom( std::string& atomType, RealOpenMM charge, int radiusIndex, int gammaIndex, RealOpenMM x_coord, RealOpenMM y_coord, RealOpenMM z_coord );
        void addBond( int particle1, int particle2, RealOpenMM bondLength );

        void loadParameters( const std::vector<RealOpenMM>& gbviRadiusVector, const std::vector<RealOpenMM>& gbviGammaVector );

        void printMolecule( FILE* filePtr, GbviParameters& gbviParameters, const std::string& idString );

        int                             _index;

        std::string                     _name;

        std::vector<OpenMMVec3>         _atomCoordinates;
        StringVector                    _atomTypes;

        std::vector<int>                _radiiIndices;
        std::vector<int>                _gammaIndices;

        std::vector<RealOpenMM>         _charges;
        std::vector<RealOpenMM>         _atomicRadii;
        std::vector<RealOpenMM>         _scaledRadii;
        std::vector<RealOpenMM>         _gamma;
        std::vector<GbviBond>           _gbviBonds;
        std::vector<RealOpenMM>         _bornRadii;
        double _energy;
};

GbviMolecule::GbviMolecule( ){
    _name   = "NA";
    _energy = 0.0;
}

void GbviMolecule::setName( std::string& name ){
    _name = name;
}

static int maxMoleculePrint            = 0;
void GbviMolecule::loadParameters( const std::vector<RealOpenMM>& gbviRadiusVector, const std::vector<RealOpenMM>& gbviGammaVector ){

    unsigned int numberOfParticles = _atomCoordinates.size();

    _atomicRadii.resize( numberOfParticles );
    _gamma.resize(       numberOfParticles );
    _bornRadii.resize(   numberOfParticles );

    int error = 0;
    for( unsigned int ii = 0; ii < numberOfParticles; ii++ ){
        if( _gammaIndices[ii] >= 0 &&_gammaIndices[ii] < gbviGammaVector.size() ){
            _gamma[ii]        = gbviGammaVector[_gammaIndices[ii]];
        } else {
            (void) fflush( NULL );
            (void) fprintf( stderr, "Invalid gamma index for particle %u: %d max=%u molIndex=%d %s\n",
                            ii, _gammaIndices[ii], gbviGammaVector.size(), _index, _name.c_str() );
            (void) fflush( NULL );
            error++;
        }

        if( _radiiIndices[ii] >= 0 && _radiiIndices[ii] < gbviRadiusVector.size() ){
            _atomicRadii[ii]  = gbviRadiusVector[_radiiIndices[ii]];
        } else {
            (void) fflush( NULL );
            (void) fprintf( stderr, "Invalid radius index for particle %u: %d max=%u; molIndex=%d %s\n",
                            ii, _radiiIndices[ii], gbviRadiusVector.size(), _index, _name.c_str() );
            (void) fflush( NULL );
            error++;
        }

    }

    if( 0 && error && maxMoleculePrint++ < 10 ){
        printMolecule( stderr, globalGbviParameters, "loadParameters");
    }

    if( 0 ){
    (void) fprintf( stderr, "loadParameters: Molecule %d %s atoms=%u bonds=%u\n",
                    _index, _name.c_str(), _atomCoordinates.size(), _gbviBonds.size() );

    for( unsigned int ii = 0; ii < _atomCoordinates.size(); ii++ ){
        (void) fprintf( stderr, "%4d %3s g[%14.5e %3d]    r[%14.5e %3d] scl=%14.5e\n",
                        ii, _atomTypes[ii].c_str(), _gamma[ii], _gammaIndices[ii], _atomicRadii[ii], _radiiIndices[ii], _scaledRadii[ii] );
    }
    (void) fflush( stderr );
    }

    return;
}

void GbviMolecule::addAtom( std::string& atomType, RealOpenMM charge, int radiusIndex, int gammaIndex, RealOpenMM x_coord, RealOpenMM y_coord, RealOpenMM z_coord ){

    _atomTypes.push_back( atomType );
    _atomCoordinates.push_back( OpenMMVec3(  x_coord, y_coord, z_coord ) );
    _charges.push_back( charge );
    _radiiIndices.push_back( radiusIndex );
    _gammaIndices.push_back( gammaIndex );
    
}

void GbviMolecule::addBond( int particle1, int particle2, RealOpenMM bondLength ){

    GbviBond gbviBond;
    gbviBond.setGbviBond( particle1, particle2, bondLength );
    _gbviBonds.push_back( gbviBond );
    
}

void GbviMolecule::printMolecule( FILE* filePtr, GbviParameters& gbviParameters, const std::string& idString = "" ){

    std::vector<RealOpenMM> atomicRadiiParameters = gbviParameters.getGbviRadiusVector();
    std::vector<RealOpenMM> atomicGammaParameters = gbviParameters.getGbviGammaVector();
    double totalQ           = 0.0;
    for( unsigned int ii = 0; ii < _atomCoordinates.size(); ii++ ){
        double charge       = _charges[ii];
        totalQ             += charge;
    }

    (void) fprintf( filePtr, "%s\nMolecule %d %s atoms=%u bonds=%u total charge=%15.7e PE=%15.7e (kJ) %15.7e (kCal)\n", idString.c_str(),
                    _index, _name.c_str(), _atomCoordinates.size(), _gbviBonds.size(), totalQ, _energy, 0.2390057361*_energy );

    for( unsigned int ii = 0; ii < _atomCoordinates.size(); ii++ ){
        double charge       = _charges[ii];
        double radiusI      = _atomicRadii[ii];
        double radiusIp     = atomicRadiiParameters[_radiiIndices[ii]];
        double gamma        = _gamma[ii];
        double gammap       = atomicGammaParameters[_gammaIndices[ii]];
        const char* isOk    = _scaledRadii[ii] > 0.05 && fabs(_scaledRadii[ii] -  radiusI) < 1.5 ? "" : "XXX";
        (void) fprintf( filePtr, "%4d %3s %14.5e  g[%3d %14.5e %14.5e]    r[%3d %14.5e %14.5e]  %14.5e %s ",
                        ii, _atomTypes[ii].c_str(), charge, _gammaIndices[ii], gamma, gammap, _radiiIndices[ii], radiusI, radiusIp, _scaledRadii[ii], isOk );
       
        if( _bornRadii.size() > ii ){
           (void) fprintf( filePtr, "%14.5e ", _bornRadii[ii] );
        }
        (void) fprintf( filePtr, "crd[%14.5e %14.5e %14.5e]\n",
                        _atomCoordinates[ii][0], _atomCoordinates[ii][1], _atomCoordinates[ii][2] );
    }
    (void) fflush( filePtr );

    return;
}

/**---------------------------------------------------------------------------------------

    Read atom block from parameter file:

    Format:
            index type         charge 
                              radiusIndex
                                     gammaIndex                                       coordinates

    Sample:
            atoms  5
                0   C  -1.0867772e-01    4    4    -2.9504299e-07  -1.0309815e-05   2.8741360e-06
                1   H   2.7365677e-02    7    8     5.5406272e-02   7.9960823e-02   4.9648654e-02
                2   H   2.7326990e-02    7    8     6.8328631e-02  -8.1335044e-02  -2.5374085e-02
                3   H   2.7000926e-02    7    8    -7.7822578e-02  -3.7350243e-02   6.6930377e-02
                4   H   2.6984129e-02    7    8    -4.5912036e-02   3.8734773e-02  -9.1207826e-02
            
    @param filePtr              file pointer to parameter file
    @param tokens               array of strings from first line of parameter file for this block of parameters
    @param gbviMolecule         gbviMolecule to add atoms to
    @param lineCount            used to track line entries read from parameter file
    @param inputLog             log file pointer -- may be NULL

    --------------------------------------------------------------------------------------- */

static void readAtoms( FILE* filePtr, const StringVector& tokens, GbviMolecule& gbviMolecule, 
                       int* lineCount, FILE* inputLog ){

// ---------------------------------------------------------------------------------------

    static const std::string methodName      = "readAtoms";
    
// ---------------------------------------------------------------------------------------

    FILE* errorLog = inputLog ? inputLog : stderr;
    if( tokens.size() < 1 ){
        (void) fprintf( errorLog, "%s no atoms number entry\n", methodName.c_str() );
        (void) fflush( errorLog );
        exit(-1);
    }

    int numberOfAtoms = atoi( tokens[1].c_str() );
    if( numberOfAtoms <= 0 || numberOfAtoms > 1000000 ){
        (void) fprintf( errorLog, "%s number of atoms invalid.\n", methodName.c_str(), numberOfAtoms );
        (void) fflush( errorLog );
        exit(-1);
    }

    for( int ii = 0; ii < numberOfAtoms; ii++ ){
       StringVector lineTokens;
       int isNotEof = readLine( filePtr, lineTokens, lineCount, inputLog );
       if( isNotEof && lineTokens.size() > 6 ){
          int tokenIndex       = 0;
          int index            = atoi( lineTokens[tokenIndex++].c_str() );
          std::string atomType =lineTokens[tokenIndex++];
          double charge        = atof( lineTokens[tokenIndex++].c_str() );
          int    radiusIndex   = atoi( lineTokens[tokenIndex++].c_str() );
          int    gammaIndex    = atoi( lineTokens[tokenIndex++].c_str() );
          double x_coord       = atof( lineTokens[tokenIndex++].c_str() );
          double y_coord       = atof( lineTokens[tokenIndex++].c_str() );
          double z_coord       = atof( lineTokens[tokenIndex++].c_str() );
          gbviMolecule.addAtom( atomType, charge, radiusIndex, gammaIndex, x_coord, y_coord, z_coord );
       } else {
          (void) fprintf( errorLog, "%s tokens incomplete at line=%d: expected %d atoms.\n", methodName.c_str(), *lineCount, numberOfAtoms );
          exit(-1);
       }
    }

    return;
}

/**---------------------------------------------------------------------------------------

    Read bond block from parameter file:

    Format:
            index  atom1  atom2   bond distance

    Sample:
            bonds 4
                0      0      1   1.0922442e-01
                1      0      2   1.0920872e-01
                2      0      3   1.0922394e-01
                3      0      4   1.0921750e-01
            
    @param filePtr              file pointer to parameter file
    @param tokens               array of strings from first line of parameter file for this block of parameters
    @param gbviMolecule         gbviMolecule to add atoms to
    @param lineCount            used to track line entries read from parameter file
    @param inputLog             log file pointer -- may be NULL

    --------------------------------------------------------------------------------------- */

static void readBonds( FILE* filePtr, const StringVector& tokens, GbviMolecule& gbviMolecule, 
                       int* lineCount, FILE* inputLog ){

// ---------------------------------------------------------------------------------------

    static const std::string methodName      = "readBonds";
    
// ---------------------------------------------------------------------------------------

    FILE* log = inputLog ? inputLog : stderr;

    if( tokens.size() < 1 ){
       (void) fprintf( log, "%s no bonds number entry???\n", methodName.c_str() );
       (void) fflush( log );
       exit(-1);
    }

    int numberOfBonds = atoi( tokens[1].c_str() );
    if( inputLog ){
       //(void) fprintf( log, "%s number of bonds=%d\n", methodName.c_str(), numberOfBonds);
    }
    for( int ii = 0; ii < numberOfBonds; ii++ ){
        StringVector lineTokens;
        int isNotEof = readLine( filePtr, lineTokens, lineCount, log );
        if( lineTokens.size() > 3 ){
            int tokenIndex       = 0;
            int index            = atoi( lineTokens[tokenIndex++].c_str() );
            int particle1        = atoi( lineTokens[tokenIndex++].c_str() );
            int particle2        = atoi( lineTokens[tokenIndex++].c_str() );
            double bondLength    = atof( lineTokens[tokenIndex++].c_str() );
            gbviMolecule.addBond( particle1, particle2, bondLength );
        } else {
            (void) fprintf( log, "%s tokens incomplete at line=%d\n", methodName.c_str(), *lineCount );
            exit(-1);
        }
    }

    return;
}

/**---------------------------------------------------------------------------------------

    Read parameters for a molecule: create GbviMolecule and calls readAtoms() and readBonds()
                                    append molecule to molecules vector

    @param filePtr              file pointer to parameter file
    @param tokens               array of strings from first line of parameter file for this block of parameters
    @param molecules            list of molecules
    @param lineCount            used to track line entries read from parameter file
    @param inputLog             log file pointer -- may be NULL

    --------------------------------------------------------------------------------------- */

static void readMolecules( FILE* filePtr, const StringVector& tokens, std::vector<GbviMolecule>& molecules,
                           int* lineCount, FILE* inputLog ){

// ---------------------------------------------------------------------------------------

    static const std::string methodName      = "readMolecules";
    
// ---------------------------------------------------------------------------------------

    FILE* log = inputLog ? inputLog : stderr;

    if( tokens.size() < 1 ){
       (void) fprintf( log, "%s no molecules?\n", methodName.c_str() );
       (void) fflush( log );
       exit(-1);
    }

    int numberOfMolecules = atoi( tokens[1].c_str() );
    if( inputLog ){
       (void) fprintf( log, "molecules=%d\n", numberOfMolecules );
    }

    for( int ii = 0; ii < numberOfMolecules; ii++ ){

        GbviMolecule gbviMolecule;

        StringVector lineTokens;
        StringVector atomLineTokens;
        StringVector nameLineTokens;
        StringVector bondLineTokens;
        int isNotEof = readLine( filePtr, lineTokens, lineCount, log );
            isNotEof = readLine( filePtr, nameLineTokens, lineCount, log );
        gbviMolecule.setName( nameLineTokens[1] );
            isNotEof = readLine( filePtr, atomLineTokens, lineCount, log );
 
        readAtoms( filePtr, atomLineTokens, gbviMolecule, lineCount, log );

        isNotEof     = readLine( filePtr, bondLineTokens, lineCount, log );
        readBonds( filePtr, bondLineTokens, gbviMolecule, lineCount, log );

        gbviMolecule._index = ii;
        molecules.push_back( gbviMolecule );
    }

    return;
}

/**---------------------------------------------------------------------------------------

    Read vectors pf parameter value and associated atom types
    used to read in gamma and radius vectors 

    @param filePtr              file pointer to parameter file
    @param tokens               array of strings from first line of parameter file for this block of parameters
                                line format: ... <numberOfTokens> <int value1> <int value2> ...<int valueN>,
                                where N=<numberOfTokens> 
    @param parameterVector      output vector of parameters
    @param atomTypeVector       output vector of atom types (one for each parameter)
    @param lineCount            used to track line entries read from parameter file
    @param inputLog             log file pointer -- may be NULL

    --------------------------------------------------------------------------------------- */

static void readGammaRadiusVector( FILE* filePtr, const StringVector& tokens,
                                   std::vector<double>& parameterVector,
                                   StringVector& atomTypeVector, int* lineCount,
                                   FILE* inputLog ){

// ---------------------------------------------------------------------------------------

    static const std::string methodName      = "readGammaRadiusVector";
    
// ---------------------------------------------------------------------------------------

    FILE* log = inputLog ? inputLog : stderr;

    if( tokens.size() < 1 ){ 
        char buffer[1024];
        (void) fprintf( log, "%s no entries?\n", methodName.c_str() );
        (void) fflush( log );
        exit(-1);
    }
 
    int numberToRead = atoi( tokens[1].c_str() );
    if( inputLog ){
        (void) fprintf( log, "%s number to read: %d\n", methodName.c_str(), numberToRead );
        (void) fflush( log );
    }

    parameterVector.resize( numberToRead );
    atomTypeVector.resize( numberToRead );

    for( int ii = 0; ii < numberToRead; ii++ ){
        StringVector lineTokens;
        int isNotEof = readLine( filePtr, lineTokens, lineCount, log );
        if( lineTokens.size() > 1 ){
            int tokenIndex       = 0;
            std::string atomType = lineTokens[tokenIndex++];
            int index            = atoi( lineTokens[tokenIndex++].c_str() );
            double pValue        = atof( lineTokens[tokenIndex++].c_str() );
            atomTypeVector[ii]   = atomType;
            parameterVector[ii]  = pValue;
        } else {
            (void) fprintf( log, "%s tokens incomplete at line=%d\n", methodName.c_str(), *lineCount );
            exit(-1);
        }
    }    
 
    return;
}

/**---------------------------------------------------------------------------------------

    Read parameter file

    Sample format:

            molecules        10
            molecule 0
            name methane
            atoms  5
                0   C  -1.0867772e-01    4    4    -2.9504299e-07  -1.0309815e-05   2.8741360e-06
                1   H   2.7365677e-02    7    8     5.5406272e-02   7.9960823e-02   4.9648654e-02
                2   H   2.7326990e-02    7    8     6.8328631e-02  -8.1335044e-02  -2.5374085e-02
                3   H   2.7000926e-02    7    8    -7.7822578e-02  -3.7350243e-02   6.6930377e-02
                4   H   2.6984129e-02    7    8    -4.5912036e-02   3.8734773e-02  -9.1207826e-02
            bonds 4
                0      0      1   1.0922442e-01
                1      0      2   1.0920872e-01
                2      0      3   1.0922394e-01
                3      0      4   1.0921750e-01
            

    @param inputParameterFileName   input parameter file name
    @param inputArgumentMap         not used
    @param gbviParameters           load parameters into Vec3 array containing coordinates on output
    @param molecules                output list of molecules
    @param inputLog                 log file pointer -- may be NULL

    --------------------------------------------------------------------------------------- */

void readParameterFile( const std::string& inputParameterFileName, MapStringString& inputArgumentMap,
                        GbviParameters& gbviParameters, std::vector<GbviMolecule>& molecules, 
                        FILE* inputLog ){

// ---------------------------------------------------------------------------------------

    FILE* log = inputLog ? inputLog : stderr;

    // open parameter file
 
    FILE* filePtr = openFile( inputParameterFileName, "r", log );
    if( filePtr == NULL ){
        (void) fprintf( log, "Input parameter file=<%s> could not be opened -- aborting.\n", inputParameterFileName.c_str() );
        (void) fflush( log );
       exit(-1);
    } else if( inputLog ){
        (void) fprintf( log, "Input parameter file=<%s> opened.\n", inputParameterFileName.c_str() );
        (void) fflush( log );
    }

    int lineCount                  = 0;
    int version                    = 0; 
    int isNotEof                   = 1;
    double soluteDielectric;
    double solventDielectric;
    double quinticLowerLimitFactor;
    double quinticUpperBornRadiusLimit;
    gbviParameters.getGbviParameters( soluteDielectric, solventDielectric, quinticLowerLimitFactor, quinticUpperBornRadiusLimit );

    // loop over lines in file

    while( isNotEof ){
 
        // read line and continue if not EOF and tokens found on line
 
        StringVector tokens;
        isNotEof = readLine( filePtr, tokens, &lineCount, log );
 
        if( isNotEof && tokens.size() > 0 ){
 
            std::string field       = tokens[0];
  
            if( inputLog ){
                (void) fprintf( log, "Field=<%s> at line=%d\n", field.c_str(), lineCount );
                (void) fflush( log );
            }
   
            if( field == "Version" ){
                if( tokens.size() > 1 ){
                   version = atoi( tokens[1].c_str() );
                   if( inputLog ){
                       (void) fprintf( log, "Version=%d at line=%d\n", version, lineCount );
                   }
                }
            } else if( field == "molecules" ){
                readMolecules( filePtr, tokens, molecules, &lineCount, log );
            } else if( field == "soluteDielectric" ){
                soluteDielectric = atof( tokens[1].c_str() );
            } else if( field == "solventDielectric" ){
                solventDielectric = atof( tokens[1].c_str() );
            } else if( field == "quinticLowerLimitFactor" ){
                quinticLowerLimitFactor = atof( tokens[1].c_str() );
            } else if( field == "quinticUpperBornRadiusLimit" ){
                quinticUpperBornRadiusLimit = atof( tokens[1].c_str() );
            } else if( field == "radius" ){
                readGammaRadiusVector( filePtr, tokens, gbviParameters.getGbviRadiusVector(),
                                       gbviParameters.getGbviRadiusAtomTypeVector(), &lineCount, log );
            } else if( field == "gamma" ){
                readGammaRadiusVector( filePtr, tokens, gbviParameters.getGbviGammaVector(),
                                       gbviParameters.getGbviGammaAtomTypeVector(), &lineCount, log );
            } else {
                (void) fprintf( log, "Field=<%s> not recognized at line=%d.\n", field.c_str(), lineCount );
                (void) fflush( log );
                exit(-1);
            }
       }
    }
    gbviParameters.setGbviParameters( soluteDielectric, solventDielectric, quinticLowerLimitFactor, quinticUpperBornRadiusLimit );

    if( inputLog ){
        (void) fprintf( log, "Exiting readParameterFile: molecules=%u.\n", molecules.size() );
        gbviParameters.printGbviParameters( log );
        (void) fprintf( log, "Exiting readParameterFile\n" );
        (void) fflush( log );
    }

    return;
}

/**---------------------------------------------------------------------------------------

    Compute scaled radii used in GB/VI algorithm based on covalent bond info
    gbviMolecule._scaledRadii[]

    @param gbviMolecule             input molecule to compute scaled radii
    @param atomicRadiiParameters    input radii

    --------------------------------------------------------------------------------------- */

void computeScaledRadii( GbviMolecule& gbviMolecule, const std::vector<RealOpenMM>& atomicRadiiParameters ){

    unsigned int numberOfParticles              = gbviMolecule._atomCoordinates.size();
    //(void) fprintf( stderr, "Molecule %d %s numberOfParticles=%d\n", gbviMolecule._index, gbviMolecule._name.c_str(), numberOfParticles ); fflush( stderr );

    gbviMolecule._scaledRadii.resize( numberOfParticles );
    std::vector<GbviBond>& gbviBonds            = gbviMolecule._gbviBonds; 
    
    // load 1-2 indicies for each atom 

    std::vector<std::vector<int> > bonded12(numberOfParticles);
    std::vector<RealOpenMM> bondLengths;

    for( unsigned int ii = 0; ii < gbviBonds.size(); ii++ ){

        int particle1;
        int particle2;
        RealOpenMM bondLength;

        gbviBonds[ii].getGbviBond( particle1, particle2, bondLength );
        bonded12[particle1].push_back(ii);
        bonded12[particle2].push_back(ii);
//(void) fprintf( stderr, "getGbviBond: %3u %3d %3u  %d %3u   %12.3e\n", ii, particle1, bonded12[particle1].size(), particle2, bonded12[particle2].size(), bondLength ); fflush( stderr );
        bondLengths.push_back(bondLength);
    }

    int errors = 0;

    // compute scaled radii (Eq. 5 of Labute paper [JCC 29 p. 1693-1698 2008])

    for (int j = 0; j < (int) bonded12.size(); ++j){

        double radiusJ;
        double scaledRadiusJ;
     
        radiusJ = atomicRadiiParameters[gbviMolecule._radiiIndices[j]];

        if(  radiusJ <= 0.0 ){
            (void) fprintf( stderr, "computeScaledRadii: Warning atom %d has atomic radius=%.3f.\n", j, radiusJ );
            errors++;
        }

        if(  bonded12[j].size() == 0 && numberOfParticles > 1 ){
            (void) fprintf( stderr, "Warning GBVIForceImpl::findScaledRadii atom %d has no covalent bonds; using atomic radius=%.3f.\n", j, radiusJ );
            scaledRadiusJ = radiusJ;
//             errors++;
        } else {

            double rJ2    = radiusJ*radiusJ;
    
            // loop over bonded neighbors of atom j, applying Eq. 5 in Labute

            scaledRadiusJ = 0.0;
            for (int i = 0; i < (int) bonded12[j].size(); ++i){
    
               int index            = bonded12[j][i];
               int particle1, particle2;
               RealOpenMM bondLength;
               gbviBonds[index].getGbviBond( particle1, particle2, bondLength );
               int bondedAtomIndex  = (j == particle1) ? particle2 : particle1;
              
               double radiusI       = atomicRadiiParameters[gbviMolecule._radiiIndices[bondedAtomIndex]];
               if(  radiusJ <= 0.0 ){
                   (void) fprintf( stderr, "computeScaledRadii: Warning atom %d has atomic radius=%.3f.\n", bondedAtomIndex, radiusI );
                   errors++;
               }

               double rI2           = radiusI*radiusI;
    
               double a_ij          = (radiusI - bondLength);
                      a_ij         *= a_ij;
                      a_ij          = (rJ2 - a_ij)/(2.0*bondLength);
    
               double a_ji          = radiusJ - bondLength;
                      a_ji         *= a_ji;
                      a_ji          = (rI2 - a_ji)/(2.0*bondLength);
    
               scaledRadiusJ       += a_ij*a_ij*(3.0*radiusI - a_ij) + a_ji*a_ji*( 3.0*radiusJ - a_ji );
            }
    
            scaledRadiusJ           = (radiusJ*radiusJ*radiusJ) - 0.125*scaledRadiusJ; 
            if( scaledRadiusJ > 0.0 ){
                scaledRadiusJ  = 0.95*pow( scaledRadiusJ, (1.0/3.0) );
            } else {
//                scaledRadiusJ  = 0.0;
            }
        }
        gbviMolecule._scaledRadii[j] = scaledRadiusJ;

    }

    // abort if errors

    if( errors ){
        fprintf( stderr, "computeScaledRadii: errors -- aborting");
        fflush( stderr );
        exit(-1);
    }

#if GBVIDebug
    (void) fprintf( stderr, "                  R              q          scaled radii no. bnds\n" );
    double totalQ = 0.0;
    for( unsigned int ii = 0; ii < gbviMolecule._scaledRadii.size(); ii++ ){

        double charge       = gbviMolecule._charges[ii];
        double radiusI      = atomicRadiiParameters[gbviMolecule._radiiIndices[ii]];
        totalQ             += charge;
        double gamma        = atomicRadiiParameters[gbviMolecule._gammaIndices[ii]];
        const char* isOk    = gbviMolecule._scaledRadii[ii] > 0.1 && fabs(gbviMolecule._scaledRadii[ii] -  radiusI) < 1.5 ? "" : "XXX";
        (void) fprintf( stderr, "%4d %3s %14.5e %14.5e %14.5e %14.5e %u %s\n", ii, gbviMolecule._atomTypes[ii].c_str(), radiusI, charge, gamma, gbviMolecule._scaledRadii[ii], bonded12[ii].size(), isOk );
    }
    (void) fprintf( stderr, "Total charge=%e\n", totalQ );
    (void) fflush( stderr );
#endif
//#undef GBVIDebug

}

std::vector<GbviMolecule> globalMolecules;

/**---------------------------------------------------------------------------------------

    Read molecules from file (swigged)

    @param inputParameterFileName   input parameter file name

    --------------------------------------------------------------------------------------- */

void addMolecules( char* inputParameterFileName ){

    FILE* log                    = stderr;
    (void) fflush( NULL );
    (void) fprintf( stderr, "\naddMolecules: Opening file %s\n", inputParameterFileName );
    (void) fflush( NULL );

    MapStringString inputArgumentMap;
    globalMolecules.resize(0);
    readParameterFile( inputParameterFileName, inputArgumentMap, globalGbviParameters, globalMolecules, log );

    return;
}

/**---------------------------------------------------------------------------------------

    Compute GB/VI energy givena molecule index (swigged)

    @param moleculeIndex            index of molecule the energy is to be computed for
    @param inputParameterFileName   input parameter file name; contains parameters for
                                    for each atom type

    @return GB/VI energy

    --------------------------------------------------------------------------------------- */

double getGBVIEnergyFromFile( int moleculeIndex, char* inputParameterFile ){

    FILE* log                    = stderr;
    (void) fprintf( stderr, "Opening file %s\n", inputParameterFile );
    (void) fflush( NULL );

    MapStringString inputArgumentMap;
    readParameterFile( inputParameterFile, inputArgumentMap, globalGbviParameters, globalMolecules, log );

    RealOpenMM soluteDielectric;
    RealOpenMM solventDielectric;
    RealOpenMM quinticLowerLimitFactor;
    RealOpenMM quinticUpperBornRadiusLimit;
    globalGbviParameters.getGbviParameters( soluteDielectric, solventDielectric, quinticLowerLimitFactor, quinticUpperBornRadiusLimit );

    RealOpenMM tau = (1.0/soluteDielectric) - (1.0/solventDielectric);
    computeScaledRadii( globalMolecules[moleculeIndex], globalGbviParameters.getGbviRadiusVector() );

    globalMolecules[moleculeIndex].loadParameters( globalGbviParameters.getGbviRadiusVector(), globalGbviParameters.getGbviGammaVector() );

    int nansDetected = computeBornRadii( globalMolecules[moleculeIndex]._atomCoordinates, 
                                         globalMolecules[moleculeIndex]._atomicRadii,
                                         globalMolecules[moleculeIndex]._scaledRadii,
                                         quinticLowerLimitFactor,
                                         quinticUpperBornRadiusLimit,
                                         globalMolecules[moleculeIndex]._bornRadii );

    if( nansDetected ){
        globalMolecules[moleculeIndex]._energy = 4.184e+06;
        return 1.0e+06;
    } else {
        globalMolecules[moleculeIndex]._energy = computeBornEnergy( globalMolecules[moleculeIndex]._bornRadii,
                                                                globalMolecules[moleculeIndex]._atomCoordinates, 
                                                                globalMolecules[moleculeIndex]._atomicRadii, tau,
                                                                globalMolecules[moleculeIndex]._gamma,
                                                                globalMolecules[moleculeIndex]._charges );
    }

    return (globalMolecules[moleculeIndex]._energy/4.184);
}

/**---------------------------------------------------------------------------------------

    Print info for molecule -- used for diagnostics

    @param moleculeIndex            index of molecule info is to printed about
    @param soluteDielectric         solute dielectric 
    @param solventDielectric        solvent dielectric 
    @param radiiVector              atomic radii for each atom type
    @param gammaVector              gamma parameters for each atom type
    @param idString                 id string -- track who is calling

    --------------------------------------------------------------------------------------- */

void printInfo( int moleculeIndex, RealOpenMM soluteDielectric, RealOpenMM solventDielectric,
                const std::vector<double> radiiVector, const std::vector<double> gammaVector, std::string& idString ){

    (void) fflush( NULL );
    (void) fprintf( stderr, "nan detected: %s  molecule=%d dielectrics: %8.3f %8.3f\n", 
                    idString.c_str(), moleculeIndex, soluteDielectric, solventDielectric );
    globalMolecules[moleculeIndex].printMolecule( stderr, globalGbviParameters, "getGBVIEnergy");
    for( unsigned int ii = 0; ii < radiiVector.size(); ii++ ){
        (void) fprintf( stderr, "Input: %2u r=%14.7e g=%14.7e\n", ii, radiiVector[ii], gammaVector[ii] );
    }
    
    for( unsigned int ii = 0; ii < globalMolecules[moleculeIndex]._atomicRadii.size(); ii++ ){
        (void) fprintf( stderr, "Actual: %2u r=[%d %14.7e]   g=[%d %14.7e]  scaled=%14.7e bR=%15.7e\n", ii,
                        globalMolecules[moleculeIndex]._radiiIndices[ii],
                        globalMolecules[moleculeIndex]._atomicRadii[ii], 
                        globalMolecules[moleculeIndex]._gammaIndices[ii],
                        globalMolecules[moleculeIndex]._gamma[ii],
                        globalMolecules[moleculeIndex]._scaledRadii[ii],
                        globalMolecules[moleculeIndex]._bornRadii[ii]);
    }
    (void) fflush( stderr );

    return;
}

/**---------------------------------------------------------------------------------------

    Compute GB/VI energy given molecule index (swigged)

    @param moleculeIndex            index of molecule info is to printed about
    @param inputSoluteDielectric    solute dielectric 
    @param radiiVector              atomic radii for each atom type
    @param gammaVector              gamma parameters for each atom type

    @return GB/VI energy

    --------------------------------------------------------------------------------------- */

double getGBVIEnergy( int moleculeIndex, double inputSoluteDielectric, const std::vector<double> radiiVector, const std::vector<double> gammaVector ){

    FILE* log  = stderr;
    //(void) fprintf( stderr, "In getGBVIEnergy molecule=%d %u %u maxMolIndex=%u\n", moleculeIndex, radiiVector.size(), gammaVector.size(), globalMolecules.size() );
    //(void) fflush( NULL );
    (void) fflush( NULL );

    if( moleculeIndex < 0 || moleculeIndex >= static_cast<int>(globalMolecules.size()) ){
        (void) fprintf( stderr, "getGBVIEnergy invalid index: %d max=%u\n", moleculeIndex, globalMolecules.size() );
        (void) fflush( NULL );
        return 0.0;
    }

    RealOpenMM soluteDielectric;
    RealOpenMM solventDielectric;
    RealOpenMM quinticLowerLimitFactor;
    RealOpenMM quinticUpperBornRadiusLimit;
    globalGbviParameters.getGbviParameters( soluteDielectric, solventDielectric, quinticLowerLimitFactor, quinticUpperBornRadiusLimit );

    soluteDielectric = inputSoluteDielectric;

    RealOpenMM tau   = (1.0/soluteDielectric) - (1.0/solventDielectric);

    if( 0 && fabs( soluteDielectric - 1.0 ) > 1.0e-03 ){
        (void) fprintf( stderr, "getGBVIEnergy soluteDielectric=%12.3f tau=%12.3f\n", soluteDielectric, tau );
        (void) fflush( NULL );
    }

    computeScaledRadii( globalMolecules[moleculeIndex], radiiVector );
    globalMolecules[moleculeIndex].loadParameters( radiiVector, gammaVector );

    int nansDetected = computeBornRadii( globalMolecules[moleculeIndex]._atomCoordinates, 
                                         globalMolecules[moleculeIndex]._atomicRadii,
                                         globalMolecules[moleculeIndex]._scaledRadii,
                                         quinticLowerLimitFactor,
                                         quinticUpperBornRadiusLimit,
                                         globalMolecules[moleculeIndex]._bornRadii );

    //globalMolecules[moleculeIndex].printMolecule( log, globalGbviParameters);
    if( nansDetected ){
        globalMolecules[moleculeIndex]._energy = 4.184e+06;
        std::string idString                   = "Bad Born radii";
        //printInfo( moleculeIndex, soluteDielectric, solventDielectric, radiiVector, gammaVector, idString );
        return 1.0e+06;
    } else {
        globalMolecules[moleculeIndex]._energy = computeBornEnergy( globalMolecules[moleculeIndex]._bornRadii,
                                                                    globalMolecules[moleculeIndex]._atomCoordinates, 
                                                                    globalMolecules[moleculeIndex]._atomicRadii, tau,
                                                                    globalMolecules[moleculeIndex]._gamma,
                                                                    globalMolecules[moleculeIndex]._charges );
    }

    if( nansDetected || isNan( globalMolecules[moleculeIndex]._energy ) ){
        std::string idString                   = "Bad energy";
        //printInfo( moleculeIndex, soluteDielectric, solventDielectric, radiiVector, gammaVector, idString );
    }

    return (globalMolecules[moleculeIndex]._energy/4.184);
}

#ifndef SWIG

/**---------------------------------------------------------------------------------------

    gbvi -parameterFile gbviParameterFile.txt -deltaParameterFile deltaParameterFile.txt

    parameterFile      mol file of molecules for which GB/VI energy is to be computed
    deltaParameterFile contains parameters and atomic radii and gamma values

    results are printed to stderr

    Sample parameterFile (mol file):

    --------------------------------------------------------------------------------------

            molecules        10
            molecule 0
            name methane
            atoms  5
                0   C  -1.0867772e-01    4    4    -2.9504299e-07  -1.0309815e-05   2.8741360e-06
                1   H   2.7365677e-02    7    8     5.5406272e-02   7.9960823e-02   4.9648654e-02
                2   H   2.7326990e-02    7    8     6.8328631e-02  -8.1335044e-02  -2.5374085e-02
                3   H   2.7000926e-02    7    8    -7.7822578e-02  -3.7350243e-02   6.6930377e-02
                4   H   2.6984129e-02    7    8    -4.5912036e-02   3.8734773e-02  -9.1207826e-02
            bonds 4
                0      0      1   1.0922442e-01
                1      0      2   1.0920872e-01
                2      0      3   1.0922394e-01
                3      0      4   1.0921750e-01
            molecule 1
            name ethane
            atoms  8
                0   C  -9.3621597e-02    4    4     8.1511319e-02  -5.3833634e-02   4.9281269e-02
                1   C  -9.3972504e-02    4    4     2.1547971e-01  -5.1630899e-03  -1.1660838e-03
                2   H   3.1720728e-02    7    8     1.3566017e-05   7.3850155e-05   5.8844686e-05
                3   H   3.1139171e-02    7    8     6.9313282e-02  -1.6066608e-01   2.9061654e-02
                4   H   3.1125726e-02    7    8     7.3028535e-02  -3.8001561e-02   1.5720731e-01
                5   H   3.1340610e-02    7    8     2.9697950e-01  -5.9062803e-02   4.8048842e-02
                6   H   3.1168176e-02    7    8     2.2767420e-01   1.0165613e-01   1.9080730e-02
                7   H   3.1099686e-02    7    8     2.2395961e-01  -2.0965552e-02  -1.0909455e-01
            bonds 7
                0      0      1   1.5119949e-01
                1      0      2   1.0941091e-01
                2      0      3   1.0941114e-01
                3      0      4   1.0941043e-01
                4      1      5   1.0940523e-01
                5      1      6   1.0940285e-01
                6      1      7   1.0940832e-01
            
                .................................

            
    --------------------------------------------------------------------------------------

    Sample deltaParameterFile:

    --------------------------------------------------------------------------------------

            solventDielectric    1.000
            solventDielectric   78.300
            gamma  10  
              I   0   3.6358960e-01
              N   1  -1.6921351e+01
             Cl   2   5.4810400e-02
             Br   3  -4.8735232e+00
              C   4  -1.1978792e+00
              P   5  -1.7230130e+01
              O   6   2.8698056e+00
              F   7   8.0788856e+00
              H   8   1.0196408e+00
              S   9  -4.2739560e+00
            radius  10  
              P   0   2.1500000e-01
             Cl   1   1.8000000e-01
              O   2   1.3500000e-01
              N   3   1.6500000e-01
              C   4   1.8000000e-01
              I   5   2.6000000e-01
              S   6   1.9500000e-01
              H   7   1.2500000e-01
             Br   8   2.4000000e-01
              F   9   1.5000000e-01
            

    --------------------------------------------------------------------------------------- */

int main( int numberOfArguments, char* argv[] ) {

    FILE* log                    = stderr;
    std::vector<GbviMolecule> molecules;
    GbviParameters gbviParameters;

    // get any input arguments

    MapStringString inputArgumentMap;
    if( numberOfArguments > 0 ){ 
        appendInputArgumentsToArgumentMap( numberOfArguments, argv, inputArgumentMap);
    }

    // echo input arguments

    (void) fprintf( log, "Input arguments %u:\n", inputArgumentMap.size() ); fflush( log );
    for( MapStringStringCI ii = inputArgumentMap.begin(); ii != inputArgumentMap.end(); ii++ ){
        std::string key   = ii->first;
        std::string value = ii->second;
        (void) fprintf( log, "      %30s %40s\n", key.c_str(), value.c_str() ); fflush( log );
    }
    (void) fflush( log );

    // read input files

    std::string inputParameterFile = "NA";
    setStringFromMap( inputArgumentMap, "parameterFile", inputParameterFile );
    readParameterFile( inputParameterFile, inputArgumentMap, gbviParameters, molecules, log );

    std::string deltaParameterFile = "NA";
    setStringFromMap( inputArgumentMap, "deltaParameterFile", deltaParameterFile);
    readParameterFile( deltaParameterFile, inputArgumentMap, gbviParameters, molecules, log );

    // compute energies

    RealOpenMM soluteDielectric;
    RealOpenMM solventDielectric;
    RealOpenMM quinticLowerLimitFactor;
    RealOpenMM quinticUpperBornRadiusLimit;
    gbviParameters.getGbviParameters( soluteDielectric, solventDielectric, quinticLowerLimitFactor, quinticUpperBornRadiusLimit );

    RealOpenMM tau = (1.0/soluteDielectric) - (1.0/solventDielectric);

    //unsigned int targetMolecule = 0;

    for( unsigned int ii = 0; ii < molecules.size(); ii++ ){
        computeScaledRadii( molecules[ii], gbviParameters.getGbviRadiusVector() );
        molecules[ii].loadParameters( gbviParameters.getGbviRadiusVector(), gbviParameters.getGbviGammaVector() );
        computeBornRadii( molecules[ii]._atomCoordinates, 
                          molecules[ii]._atomicRadii,
                          molecules[ii]._scaledRadii,
                          quinticLowerLimitFactor,
                          quinticUpperBornRadiusLimit,
                          molecules[ii]._bornRadii );

        molecules[ii]._energy = computeBornEnergy( molecules[ii]._bornRadii, molecules[ii]._atomCoordinates, 
                                                   molecules[ii]._atomicRadii, tau, molecules[ii]._gamma,
                                                   molecules[ii]._charges );
    }

    // show input and energies

    for( unsigned int ii = 0; ii < molecules.size(); ii++ ){
         molecules[ii].printMolecule( log, gbviParameters, "Main:" );
    }

    return 0;
}

#endif
