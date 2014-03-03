#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
parameterize-gbvi_2.py

Parameterize the GBVI model on hydration free energies of small molecules using Bayesian inference
via Markov chain Monte Carlo (MCMC).

AUTHORS

John Chodera <jchodera@berkeley.edu>, University of California, Berkeley

The AtomTyper class is based on 'patty' by Pat Walters, Vertex Pharmaceuticals.

"""
#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import sys
import string
import shutil

from optparse import OptionParser # For parsing of command line arguments

import os
import math
import numpy

import simtk.openmm as openmm
import simtk.unit as units

import openeye.oechem
import openeye.oeomega
import openeye.oequacpac

from openeye.oechem import *
from openeye.oequacpac import *
from openeye.oeiupac import *

import time
import pymc
import gbvi
import random

#=============================================================================================
# Load OpenMM plugins.
#=============================================================================================

print "Loading OpenMM plugins..."

openmm.Platform.loadPluginsFromDirectory(os.path.join(os.environ['OPENMM_INSTALL_DIR'], 'lib'))
openmm.Platform.loadPluginsFromDirectory(os.path.join(os.environ['OPENMM_INSTALL_DIR'], 'lib', 'plugins'))

#=============================================================================================
# Generate random partition of molecules 
#=============================================================================================

def partitionMoleculeList( moleculeListSize, train_fraction ):

    """
    Generate random partition of molecules

    ARGUMENTS

    moleculeListSize (int) - number of molecules
    train_fraction (float) - training vs test fraction

    RETURNS

    partition(list)        - list of size moleculeListSize with approximately train_fraction of entries set to 1 
                             and the rest to 0
    
    """

    partition = [0]*moleculeListSize
    for ii in range( moleculeListSize ):
        if( random.random() < train_fraction ):
            partition[ii] = 1

    return partition
    
#=============================================================================================
# Write particle positions to file. File is used in runs based on serialized parameters
#=============================================================================================

def writePositionsToFile( positions, fileName = "" ):

    """
    Write particle positions to file. File is used in runs based on serialized parameters

    ARGUMENTS

    positions (2D array)   - positions[atom][0-3] = coordinate
    fileName  (string)     - output file name

            Format:
        
              24  Positions
               0    2.2449734211e-01   2.5509264469e-01  -6.6783547401e-01
               1    2.5149414539e-01   3.5013582706e-01  -5.6938042641e-01
               2    1.1764405966e-01   1.6681354046e-01  -6.5212764740e-01
               ....
        
    
    """

    verbose = 0
    try:
        if( fileName ):
            positionsFile  = open( fileName, 'w')
            positionString = "%8d  Positions\n" % (len( positions) )
            positionsFile.write( positionString )
            for ii in range( len( positions) ):
                positionString = "%8d  %18.10e %18.10e %18.10e\n" % ( ii, positions[ii][0], positions[ii][1], positions[ii][2] )
                positionsFile.write( positionString )
            positionsFile.close()

        if( verbose ):
            print( "Positions -- output to file=%s\n" ) % (fileName)
 
        return

    except:
        print "Positions output exception <", sys.exc_info()[0], ">" 

#=============================================================================================
# Serialize system and output to file
#=============================================================================================

def serializeSystemToFile( system, fileName = "" ):

    """
    Write serialized OpenMM system to file

    ARGUMENTS

    system (OpenMM system)   - system to be serialized
    fileName  (string)       - output file name

    """

    verbose = 0
    try:
        serializationString = openmm.XmlSerializer.serializeSystem( system )

        if( fileName ):
            serializationFile = open( fileName, 'w')
            serializationFile.write( serializationString )
            serializationFile.close()

        if( verbose ):
            print( "Serialized system -- output to file=%s\n" ) % (fileName)
            # print( "Serialized system %s\n" ) % serializationString
 
        return serializationString

    except:
        print "Serialization Exception <", sys.exc_info()[0], ">" 
        print "Serialization Exception <", sys.exc_value,  ">" 

#=============================================================================================
# Write serialized system and particle positions to files
#=============================================================================================

def serialize( system, positions, serializeDirectory, serializeFileName ):

    """
    Write serialized OpenMM system and particles positions to files

    File path names are serializeDirectory/serializeFileName.xml and 
                        serializeDirectory/serializeFileName.txt

    ARGUMENTS

    system (OpenMM system)        - system to be serialized
    positions (2D array)          - particle positions[atom][0-3] = coordinate
    serializeDirectory (string)   - directory path to output files
    serializeFileName (string)    - output base file name

    """

    verbose = 1
    if verbose: 
        print "Serialize: %s" % (serializeFileName)
        if system is not None:
            print "         Particles=%d NumForces=%d " % (system.getNumParticles(), system.getNumForces() )

    # serialize system

    xmlSerializeFileName  = serializeFileName + '.xml'
    fullSerializeFileName = os.path.join( serializeDirectory, xmlSerializeFileName )
    if( not os.path.isfile( fullSerializeFileName ) ):
        try:
            serializeSystemToFile( system, fullSerializeFileName )
            if verbose:
                print "Serialized system to file %s" % (fullSerializeFileName)
                sys.stdout.flush()
        except:
            print "serialization problem for xml file of %s" % (serializeFileName)
            sys.stdout.flush()
            raise
    else:
        if verbose:
            print "Serialized file %s exists -- present system not serialized." % (fullSerializeFileName)

    # write positions

    if( positions is not None ):
        positionFileName       = serializeFileName + '.txt'
        fullPositionFileName   = os.path.join( serializeDirectory, positionFileName )
        if( not os.path.isfile( fullPositionFileName ) ):
            writePositionsToFile(  positions, fullPositionFileName )
            if verbose:
                print "Wrote positions to %s" % fullPositionFileName
                sys.stdout.flush()
        else:
            print "Positions file %s exists -- not overwritten." % fullPositionFileName
            sys.stdout.flush()

#=============================================================================================
# Deserialize system from file
#=============================================================================================

def deserializeSystemFromFile(fileName):

    """
    Deserialized OpenMM system from file

    ARGUMENTS

    fileName (string)             - xml file name of serialized system

    """

    verbose = 1
    try:

        if( not os.path.exists( fileName ) ):
            print( "Serialized system file=%s is not available.\n" ) % (fileName)
            return

        serializationFile    = open( fileName, 'r')

        serializationString  = serializationFile.read( )
        serializationFile.close()
        if( verbose ):
            print( "Deserializing system from file=%s\n" ) % (fileName)
            #print( "Deserializing system from file=%s\n" ) % (serializationString)

        system = openmm.XmlSerializer.deserializeSystem( serializationString )

        if( verbose ):
            print( "Deserializing system from file=%s completed.\n" ) % (fileName)
            sys.stdout.flush()

        return system

    except:
        print "Deserialization Exception <", sys.exc_info()[0], ">" 
        print "Deserialization Exception <", sys.exc_value,  ">" 

#=============================================================================================
# Atom Typer
#=============================================================================================

class AtomTyper(object):
    """
    Atom typer

    Based on 'Patty', by Pat Walters.

    """
    
    class TypingException(Exception):
        """
        Atom typing exception.

        """
        def __init__(self, molecule, atom):
            self.molecule = molecule
            self.atom = atom

        def __str__(self):
            return "Atom not assigned: %6d %8s" % (self.atom.GetIdx(), OEGetAtomicSymbol(self.atom.GetAtomicNum()))

    def __init__(self, infileName, tagname):
        self.pattyTag = OEGetTag(tagname) 
        self.smartsList = []
        ifs = open(infileName)
        lines = ifs.readlines()
        for line in lines:
            # Strip trailing comments
            index = line.find('%')
            if index != -1:
                line = line[0:index]
            # Split into tokens.
            toks = string.split(line)
            if len(toks) == 2:
                smarts,type = toks
                pat = OESubSearch()
                pat.Init(smarts)
                pat.SetMaxMatches(0)
                self.smartsList.append([pat,type,smarts])

    def dump(self):
        for pat,type,smarts in self.smartsList:
            print pat,type,smarts

    def assignTypes(self,mol):
        # Assign null types.
        for atom in mol.GetAtoms():
            atom.SetStringData(self.pattyTag, "")        

        # Assign atom types using rules.
        OEAssignAromaticFlags(mol)
        for pat,type,smarts in self.smartsList:
            for matchbase in pat.Match(mol):
                for matchpair in matchbase.GetAtoms():
                    matchpair.target.SetStringData(self.pattyTag,type)

        # Check if any atoms remain unassigned.
        for atom in mol.GetAtoms():
            if atom.GetStringData(self.pattyTag)=="":
                raise AtomTyper.TypingException(mol, atom)

    def debugTypes(self,mol):
        for atom in mol.GetAtoms():
            print "%6d %8s %8s" % (atom.GetIdx(),OEGetAtomicSymbol(atom.GetAtomicNum()),atom.GetStringData(self.pattyTag))

    def getTypeList(self,mol):
        typeList = []
        for atom in mol.GetAtoms():
            typeList.append(atom.GetStringData(self.pattyTag))
        return typeList

#=============================================================================================
# Filter molecules based on atom types
#=============================================================================================

def filterMolecules( filterAtomType, molecules, moleculePartition ):

    """
    Filter molecules based on atom type
    For example, if atomType['F'] = 0, then any molecule containing a fluorine
    will have their moleculePartition entry set to -1. This allows these molecules
    to removed from the training/test sets.

    Number of molecules filtered out for each atom type w/ filterAtomType[] = 0,
    is reported, if verbose is set to true

    ARGUMENTS

    filterAtomType (dict)       - filterAtomType['type'] = 0 (remove) / nonzero (keep)
    molecules (list of OEMols)  - list of OpenEye molecule objects
    moleculePartition (list)    - int list of size equal to number of molecules

    """

    notFilteredMolecules = 0
    filteredMolecules    = 0
    filteredCount        = dict()

    verbose              = 1

    for (i, molecule) in enumerate(molecules):
        moleculeFiltered = 0
        for atom in molecule.GetAtoms():            
            atomtype = atom.GetStringData("gbvi_type") # GBVI atomtype
            if( atomtype in filterAtomType ):
                if( filterAtomType[atomtype] == 0 ):
                    moleculeFiltered = 1
                    if( atomtype in filteredCount ):
                        filteredCount[atomtype] += 1 
                    else:
                        filteredCount[atomtype]  = 1 
            else:
                print "Atom type %s not recognized\n" % ( atomtype )

        if( moleculeFiltered == 0 ):    
            notFilteredMolecules += 1
        else:
            moleculePartition[i] = -1
            filteredMolecules    += 1

    if( verbose ):
        print "Non filtered molecules: %d   filtered=%d" % (notFilteredMolecules, filteredMolecules)
        for atomtype in sorted( filteredCount.keys( )): 
            print "Filtered molecules: %s %d" % (atomtype, filteredCount[atomtype])

#=============================================================================================
# Load molecules from sdf file
#=============================================================================================

def loadAndTypeMolecules( fileName, allMolecules, mcmcDbName, atom_typer ):

    """
    Load molecules from sdf file

    ARGUMENTS

    fileName (string)                - file name of sdf file containing molecules
    allMolecules (int)               - if allMolecules <= 0, then all molecules in file included 
                                       in return list
                                       if allMolecules > 0,  then only first 'allMolecules' molecules included 
                                       in return list
    mcmcDbName (string)              - database name -- required so that concurrent runs
                                       perform regularization in separate directories
    atom_typer (class AtomTyper)     - int list of size equal to number of molecules

    RETURNS

    list of OEMol molecules

    """

    # Load and type all molecules in the specified dataset.
    print "Loading and typing all molecules in dataset..."
    start_time         = time.time()
    molecules          = list()
    input_molstream    = oemolistream(fileName)
    molecule           = OECreateOEGraphMol()
    while OEReadMolecule(input_molstream, molecule):
        # Get molecule name.
        name = OEGetSDData(molecule, 'name').strip()
        molecule.SetTitle(name)
        # Append to list.
        molecule_copy = OEMol(molecule)

        if( allMolecules <= 0 or len(molecules) < allMolecules ):
            molecules.append(molecule_copy)

    input_molstream.close()
    end_time     = time.time()
    elapsed_time = end_time - start_time

    print "%d molecules read in %.3f s" % (len(molecules), elapsed_time)

    # Add explicit hydrogens.

    import openeye.oeomega
    for molecule in molecules:
        openeye.oechem.OEAddExplicitHydrogens(molecule)    

    # Build a conformation for all molecules with Omega.

    omega = openeye.oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetFromCT(True)
    for molecule in molecules:
        #omega.SetFixMol(molecule)
        omega(molecule)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "Built conformations for all molecules in %.3f s ...Regularizing molecules..." % elapsed_time

    # Regularize all molecules through writing as mol2.

    regularize = 0
    if( regularize ):
        ligand_mol2_dirname  = os.path.dirname(mcmcDbName) + '/mol2'
        if( not os.path.exists( ligand_mol2_dirname ) ):
            os.makedirs(ligand_mol2_dirname)
        ligand_mol2_filename = ligand_mol2_dirname + '/temp' + os.path.basename(mcmcDbName) + '.mol2' 
        start_time = time.time()    
        omolstream = openeye.oechem.oemolostream(ligand_mol2_filename)    
        for molecule in molecules:
            # Write molecule as mol2, changing molecule through normalization.    
            openeye.oechem.OEWriteMolecule(omolstream, molecule)
        omolstream.close()
    
        elapsed_time = time.time() - start_time
        print "Regularized molecules in %.3f s elapsed" % elapsed_time
    else:
        print "Molecules not regularized"
    
    # Assign AM1-BCC charges.

    print "Assigning AM1-BCC charges..."

    start_time = time.time()
    for molecule in molecules:
        # Assign AM1-BCC charges.
        if molecule.NumAtoms() == 1:
            # Use formal charges for ions.
            OEFormalPartialCharges(molecule)         
        else:
            # Assign AM1-BCC charges for multiatom molecules.
            OEAssignPartialCharges(molecule, OECharges_AM1BCC, False) # use explicit hydrogens
        # Check to make sure we ended up with partial charges.
        if OEHasPartialCharges(molecule) == False:
            print "No charges on molecule: '%s'" % molecule.GetTitle()
            print "IUPAC name: %s" % OECreateIUPACName(molecule)
            # TODO: Write molecule out
            # Delete themolecule.
            molecules.remove(molecule)
            
    elapsed_time = time.time() - start_time
    print "Assigned AM1-BCC charges in %.3f s; %d molecules remaining" % (elapsed_time,len(molecules))
    
    # Type all molecules with GBVI parameters.

    start_time        = time.time()
    typed_molecules   = list()
    untyped_molecules = list()
    for molecule in molecules:
        # Assign GBVI types according to SMARTS rules.
        try:
            atom_typer.assignTypes(molecule)
            typed_molecules.append(OEGraphMol(molecule))
            #atom_typer.debugTypes(molecule)
            #if( len(typed_molecules) > 10 ):
            #    sys.exit(-1)
        except AtomTyper.TypingException as exception:
            name = OEGetSDData(molecule, 'name').strip()
            print name        
            print exception
            untyped_molecules.append(OEGraphMol(molecule))        
            if( len(untyped_molecules) > 10 ):
               sys.exit(-1)

    elapsed_time = time.time() - start_time

    showChargeMap = 0
    if( showChargeMap ):

        chargeDict = dict()
        for molecule in molecules:
            for atom in molecule.GetAtoms():            
                atomtype = atom.GetStringData("gbvi_type") # GBVI atomtype
                charge   = atom.GetPartialCharge() * units.elementary_charge
                name     = OEGetSDData(molecule, 'name').strip()
                if( atomtype not in chargeDict ):
                    chargeDict[atomtype]     = dict()
                chargeNoUnit  =  atom.GetPartialCharge()
                chargeNoUnit *= 1000.0
                chargeNoUnitI = int( chargeNoUnit + 0.5 )
                chargeNoUnit  = float( chargeNoUnitI )/1000.0
                charges       = chargeDict[atomtype].keys()
                minDiff       = 1.0e+30
                closest       = 1.0e+30
                for chargeK in charges:
                    if( abs( chargeK - chargeNoUnit ) < minDiff ):
                        closest = chargeK
                        minDiff = abs( chargeK - chargeNoUnit )
                        
                if( minDiff <  0.01 ):
                    chargeNoUnit = closest
    
                if( chargeNoUnit not in chargeDict[atomtype] ):
                    chargeDict[atomtype][chargeNoUnit]  = dict()
    
                if( name not in chargeDict[atomtype][chargeNoUnit] ):
                    chargeDict[atomtype][chargeNoUnit][name] = 0
    
                chargeDict[atomtype][chargeNoUnit][name] += 1
            
        for (key, value) in chargeDict.iteritems():
           outputString = "\nCharge map for %s "    % (key)
           chargeList   = value.keys()
           chargeList.sort()
           for charge in chargeList:
               names         = value[charge]
               outputString += "\n%14.5f ["    % (charge)
               count  = 0
               for name in names.keys():
                   outputString += " %s (%d)"    % (name,names[name])
                   if( count == 1000 ):
                       outputString += "\n"
                       count = 0
               outputString += "]"
           print "%s\n" % outputString

    print "%d molecules correctly typed"    % (len(typed_molecules))
    print "%d molecules missing some types" % (len(untyped_molecules))
    print "%.3f s elapsed" % elapsed_time
    #sys.exit(0)

    return typed_molecules

#=============================================================================================
# Utility routines
#=============================================================================================

def read_gbvi_parameters(filename):
        """
        Read a GB/VI parameter set from a file.

        ARGUMENTS

        filename (string) - the filename to read parameters from

        RETURNS

        parameters (dict) - parameters[(atomtype_parameterName)] contains the parameter value 

                            parameters['C_radius'] = 1.8
                            parameters['C_gamma']  = 1.2

        """

        parameters = dict()
        
        infile = open(filename, 'r')
        for line in infile:
            # Strip trailing comments
            index = line.find('%')
            if index != -1:
                line = line[0:index]            

            # Parse parameters
            elements = line.split()
            if len(elements) == 3:
                [atomtype, radius, gamma] = elements
                parameters['%s_%s' % (atomtype,'radius')] = float(radius) 
                parameters['%s_%s' % (atomtype,'gamma')]  = float(gamma)

        return parameters                

#=============================================================================================
# Computation of hydration free energies
#=============================================================================================

def compute_hydration_energies(molecules, parameters):
    """
    Compute solvation energies of a set of molecules given a GBVI parameter set.

    ARGUMENTS

    molecules (list of OEMol) - molecules with assigned atom types in type field
    parameters (dict)         - dictionary of GBVI parameters keyed on GBVI atom types

    RETURNS

    energies (dict) - energies[molecule] is the computed solvation energy of given molecule

    """

    energies = dict() # energies[index] is the computed solvation energy of molecules[index]

    platform = openmm.Platform.getPlatformByName("Reference")

    moleculeIndex = -1
    for molecule in molecules:
        moleculeIndex += 1
        # Create OpenMM System.
        system = openmm.System()
        for atom in molecule.GetAtoms():
            mass = OEGetDefaultMass(atom.GetAtomicNum())
            system.addParticle(mass * units.amu)

        # Add nonbonded term.
        #   nonbonded_force = openmm.NonbondedSoftcoreForce()
        #   nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        #   for atom in molecule.GetAtoms():
        #      charge = 0.0 * units.elementary_charge
        #      sigma = 1.0 * units.angstrom
        #      epsilon = 0.0 * units.kilocalories_per_mole
        #      nonbonded_force.addParticle(charge, sigma, epsilon)
        #   system.addForce(nonbonded_force)

        # Add GBVI term
        # gbvi_force = openmm.GBVISoftcoreForce()
        gbvi_force = openmm.GBVIForce()   
        gbvi_force.setNonbondedMethod(openmm.GBVIForce.NoCutoff) # set no cutoff
        gbvi_force.setSoluteDielectric(1)
        gbvi_force.setSolventDielectric(78.3)

        # Use scaling method.
        # gbvi_force.setBornRadiusScalingMethod(openmm.GBVISoftcoreForce.QuinticSpline)
        # gbvi_force.setQuinticLowerLimitFactor(0.75)
        # gbvi_force.setQuinticUpperBornRadiusLimit(50.0*units.nanometers)

        # Build indexable list of atoms.
        atoms = [atom for atom in molecule.GetAtoms()]   
   
        # Assign GB/VI parameters.
        for atom in molecule.GetAtoms():            
            atomtype = atom.GetStringData("gbvi_type") # GBVI atomtype
            charge = atom.GetPartialCharge() * units.elementary_charge
            radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
            gamma = parameters['%s_%s' % (atomtype, 'gamma')] * units.kilocalories_per_mole            
            # gamma *= -1.0 # DEBUG
            lambda_ = 1.0 # fully interacting
            # gbvi_force.addParticle(charge, radius, gamma, lambda_) # for GBVISoftcoreForce
            gbvi_force.addParticle(charge, radius, gamma) # for GBVIForce

        # Add bonds.
        for bond in molecule.GetBonds():
            # Get atom indices.
            iatom = bond.GetBgnIdx()
            jatom = bond.GetEndIdx()
            # Get bond length.
            (xi, yi, zi) = molecule.GetCoords(atoms[iatom])
            (xj, yj, zj) = molecule.GetCoords(atoms[jatom])
            distance = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2) * units.angstroms
            # Identify bonded atoms to GBVI.
            gbvi_force.addBond(iatom, jatom, distance)

        # Add the force to the system.
        system.addForce(gbvi_force)
        
        # Build coordinate array.
        natoms = len(atoms)
        coordinates = units.Quantity(numpy.zeros([natoms, 3]), units.angstroms)
        positions   = []
        for (index,atom) in enumerate(atoms):
            (x,y,z) = molecule.GetCoords(atom)
            coordinates[index,:] = units.Quantity(numpy.array([x,y,z]),units.angstroms)   
            positions.append( [x*0.1,y*0.1,z*0.1] )
            #positions.append( [x,y,z] )
            
        # Create OpenMM Context.
        timestep = 1.0 * units.femtosecond # arbitrary
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(system, integrator, platform)

        # Set the coordinates.
        context.setPositions(coordinates)
        serializeDirectory = '/home/friedrim/source/gbff/examples/gbvi/serialization'
        serializeFileName  = 'mol_' + ("%d" % (moleculeIndex))
        serialize( system, positions, serializeDirectory, serializeFileName )
        
        # Get the energy
        state = context.getState(getEnergy=True)
        energies[molecule] = state.getPotentialEnergy()

    return energies

#=============================================================================================
# Generate molecule info block for parameter file output
#=============================================================================================

def setup_hydration_energy(molecule, radiusParameterMap, gammaParameterMap):

    """
    Generate molecule info block for parameter file output

    ARGUMENTS

    molecule (OEMol)           - molecule whose parameters are to be output 
    radiusParameterMap (dict)  - radiusParameterMap[type_radius] = radius index
                                 where type is 'C', 'H', ... and the radius index
                                 is the index for that atom type in the radius list 
    gammaParameterMap (dict)   - gammaParameterMap[type_gamma] = gamma index
                                 where type is 'C', 'H', ... and the gamma index
                                 is the index for that atom type in the gamma list 

    RETURNS

    string for output to parameter file

    sample string for methane:

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


    """

    # Build indexable list of atoms.
    atoms             = [atom for atom in molecule.GetAtoms()]   
    natoms            = len(atoms)
    
    outputString      = "atoms " + (" %d\n" % natoms)

    atomIndex         = 0
    for atom in molecule.GetAtoms():            
        atomtype = atom.GetStringData("gbvi_type") # GBVI atomtype
        charge   = atom.GetPartialCharge()
        (x,y,z)  = molecule.GetCoords(atom)

        try:
            radiusKey   = '%s_%s' % (atomtype, 'radius')
            gammaKey    = '%s_%s' % (atomtype, 'gamma')
            if( radiusKey in radiusParameterMap ):
                radiusIndex = radiusParameterMap[radiusKey]
            else:
                radiusIndex = -1
            if( gammaKey in gammaParameterMap ):
                gammaIndex  = gammaParameterMap[gammaKey]
            else:
                gammaIndex  = -1
        except Exception, exception:
            print "Cannot find parameters for atomtype '%s' in molecule '%s'" % (atomtype, molecule.GetTitle())
            raise exception
        outputString     += "%5d %3s %15.7e %4d %4d   %15.7e %15.7e %15.7e\n" % (atomIndex, atomtype, charge, radiusIndex, gammaIndex, 0.1*x, 0.1*y, 0.1*z)
        atomIndex        += 1
        
    # Add bonds.
    bondCount = 0
    for bond in molecule.GetBonds():
        bondCount += 1

    outputString     += "bonds" + (" %d\n" % bondCount)
    bondCount         = 0
    for bond in molecule.GetBonds():
        # Get atom indices.
        iatom = bond.GetBgnIdx()
        jatom = bond.GetEndIdx()
        # Get bond length.
        (xi, yi, zi) = molecule.GetCoords(atoms[iatom])
        (xj, yj, zj) = molecule.GetCoords(atoms[jatom])
        distance = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
        # Identify bonded atoms to GBVI.
        distance         *= 0.1
        outputString     += "%5d %6d %6d %15.7e\n" % (bondCount, iatom, jatom, distance)
        bondCount        += 1

    return outputString

#=============================================================================================
# Calculate GB/VI energy for molecule given radius/gamma/solute dielectric parameters
#=============================================================================================

def compute_gbvi_energy(moleculeIndex, radiusParameterMap, gammaParameterMap, parameters):

    """
    Calculate GB/VI energy 

    ARGUMENTS

    moleculeIndex (int)        - index of molecule whose energy is to be calculated 
    radiusParameterMap (dict)  - radiusParameterMap[type_radius] = radius index
                                 where type is 'C', 'H', ... and the radius index
                                 is the index for that atom type in the radius list 
                                 input units in A
    gammaParameterMap (dict)   - gammaParameterMap[type_gamma] = gamma index
                                 where type is 'C', 'H', ... and the gamma index
                                 is the index for that atom type in the gamma list 
                                 input units in kcal/mol
    parameters (dict)          - dict of radius/gamma parameter and optionally solue dielectric values
                                 to use in calculating energy

    RETURNS

    GB/VI energy for molecule with index=moleculeIndex in kcal/mol

    """

    maxParameters     = len( radiusParameterMap )
    radii             = [0.0]*maxParameters
    gamma             = [0.0]*maxParameters
    soluteDielectric  = 1.0

    for (key, value) in parameters.iteritems():
         
        if( key.find( 'radius' ) > -1 ):
            if( key in radiusParameterMap ):
                index          = radiusParameterMap[key]
                key_int        = int(index)
                radii[key_int] = 0.1*float(value)
        elif( key.find( 'gamma' ) > -1 ):
            if( key in gammaParameterMap ):
                index          = gammaParameterMap[key]
                key_int        = int(index)
                gamma[key_int] = 4.184*float(value)
        elif( key.find( 'soluteDielectric' ) > -1 ):
                soluteDielectric = float(value)
        else: 
            print "compute_gbvi_energy: key=%s (value=%s) not recognized." % (key, value)
 
    return gbvi.getGBVIEnergy( moleculeIndex, soluteDielectric, radii, gamma )

#=============================================================================================
# Calculate GB/VI energy for molecule given radius/gamma/solute dielectric parameters
#=============================================================================================

def compute_hydration_energy(molecule, parameters, platform_name="Reference"):
    """
    Compute hydration energy of a single molecule given a GBVI parameter set.

    ARGUMENTS

    molecule (OEMol)  - molecule with GBVI atom types
    parameters (dict) - parameters for GBVI atom types

    RETURNS

    energy (float) - hydration energy in kcal/mol

    """

    platform = openmm.Platform.getPlatformByName(platform_name)

    # Create OpenMM System.
    system = openmm.System()
    for atom in molecule.GetAtoms():
        mass = OEGetDefaultMass(atom.GetAtomicNum())
        system.addParticle(mass * units.amu)

    # Add GBVI term
    gbvi_force = openmm.GBVIForce()   
    gbvi_force.setNonbondedMethod(openmm.GBVIForce.NoCutoff) # set no cutoff

    if( 'soluteDielectric' in parameters ):
        value = float(parameters['soluteDielectric'])
        gbvi_force.setSoluteDielectric(value)
    else:
        gbvi_force.setSoluteDielectric(1)
    gbvi_force.setSolventDielectric(78.3)
    
    # Use scaling method.
    
    # Build indexable list of atoms.
    atoms = [atom for atom in molecule.GetAtoms()]   
    
    # Assign GB/VI parameters.
    for atom in molecule.GetAtoms():            
        atomtype = atom.GetStringData("gbvi_type") # GBVI atomtype
        charge = atom.GetPartialCharge() * units.elementary_charge
        try:
            radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
            gamma = parameters['%s_%s' % (atomtype, 'gamma')] * units.kilocalories_per_mole
        except Exception, exception:
            print "Cannot find parameters for atomtype '%s' in molecule '%s'" % (atomtype, molecule.GetTitle())
            print parameters.keys()
            raise exception
        
        # gamma *= -1.0 # DEBUG
        lambda_ = 1.0 # fully interacting
        # gbvi_force.addParticle(charge, radius, gamma, lambda_) # for GBVISoftcoreForce
        gbvi_force.addParticle(charge, radius, gamma) # for GBVIForce
        
    # Add bonds.
    for bond in molecule.GetBonds():
        # Get atom indices.
        iatom = bond.GetBgnIdx()
        jatom = bond.GetEndIdx()
        # Get bond length.
        (xi, yi, zi) = molecule.GetCoords(atoms[iatom])
        (xj, yj, zj) = molecule.GetCoords(atoms[jatom])
        distance = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2) * units.angstroms
        # Identify bonded atoms to GBVI.
        gbvi_force.addBond(iatom, jatom, distance)

    # Add the force to the system.
    system.addForce(gbvi_force)
    
    # Build coordinate array.
    natoms = len(atoms)
    coordinates = units.Quantity(numpy.zeros([natoms, 3]), units.angstroms)
    for (index,atom) in enumerate(atoms):
        (x,y,z) = molecule.GetCoords(atom)
        coordinates[index,:] = units.Quantity(numpy.array([x,y,z]),units.angstroms)   
        
    # Create OpenMM Context.
    timestep = 1.0 * units.femtosecond # arbitrary
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator, platform)

    # Set the coordinates.
    context.setPositions(coordinates)
        
    # Get the energy
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() / units.kilocalories_per_mole
    if numpy.isnan(energy):
        energy = +1e6;

    return energy

#=============================================================================================
#  Create method to compute energy given parameter maps using OpenMM libs
#=============================================================================================

def hydration_energy_factory_OpenMM(molecule):
    def hydration_energy(**parameters):
        return compute_hydration_energy(molecule, parameters, platform_name="Reference")
    return hydration_energy

#=============================================================================================
# Create method to compute energy given parameter maps using Swigged GB/VI library
#=============================================================================================

def hydration_energy_factory_swig(moleculeIndex, radiusParameterMap, gammaParameterMap):
    """

    ARGUMENTS

    moleculeIndex (int)        - index of molecule whose energy is to be calculated 
    radiusParameterMap (dict)  - radiusParameterMap[type_radius] = radius index
                                 where type is 'C', 'H', ... and the radius index
                                 is the index for that atom type in the radius list 
                                 input units in A
    gammaParameterMap (dict)   - gammaParameterMap[type_gamma] = gamma index
                                 where type is 'C', 'H', ... and the gamma index
                                 is the index for that atom type in the gamma list 
                                 input units in kcal/mol

    RETURNS

    method to compute GB/VI energy using swigged GB/VI library methods

    """
    def hydration_energy(**parameters):
        return compute_gbvi_energy(moleculeIndex, radiusParameterMap, gammaParameterMap, parameters)
    return hydration_energy

#=============================================================================================
# Create method to compare GB/VI energies computed using Swigged GB/VI library and OpenMM libs
#=============================================================================================

def hydration_energy_factory_test( molecule, moleculeIndex, radiusParameterMap, gammaParameterMap):
    """

    ARGUMENTS

    moleculeIndex (int)        - index of molecule whose energy is to be calculated 
    radiusParameterMap (dict)  - radiusParameterMap[type_radius] = radius index
                                 where type is 'C', 'H', ... and the radius index
                                 is the index for that atom type in the radius list 
                                 input units in A
    gammaParameterMap (dict)   - gammaParameterMap[type_gamma] = gamma index
                                 where type is 'C', 'H', ... and the gamma index
                                 is the index for that atom type in the gamma list 
                                 input units in kcal/mol

    RETURNS

    method to compute GB/VI energy using swigged GB/VI library methods

    """
    def hydration_energy(**parameters):
        eSwig   = compute_gbvi_energy(moleculeIndex, radiusParameterMap, gammaParameterMap, parameters)
        eOpenMM = compute_hydration_energy(molecule, parameters, platform_name="Reference")
        eDiff   = abs(eSwig - eOpenMM)
        if( eDiff > 1.0e-05 ):
            print "%15.7e  eSwig=%15.7e eOpenMM=%15.7e" %( eDiff, eSwig, eOpenMM)
        return eOpenMM
    return hydration_energy

#=============================================================================================
# Create parameter map
#=============================================================================================

def createParameterMap(initial_parameters, fixedParameters, filterAtomType ):

    """

    ARGUMENTS

    initial_parameters (dict)  - initial_parameters[[C_radius]       = 1.7, ...
    fixedParameters    (dict)  - fixedParameters[solventDielectric]  = 78,3
                                 other recognzied fields: includeSoluteDielectricAsParameter, soluteDielectric, 
                                                          energyCalculations (swig or openmm)
    filterAtomType     (dict)  - filterAtomType[C] = nozero value if included; 0 if C is to be excluded

    RETURNS

    return parameterMap (dict) w/ keys/value:

        parameterMap['model']                        = model;
        parameterMap['stochastic']                   = parameters;
        parameterMap['radiusParameterMap']           = radiusParameterMap;
        parameterMap['radiusParameterReverseMap']    = radiusParameterReverseMap;
        parameterMap['gammaParameterMap']            = gammaParameterMap;
        parameterMap['gammaParameterReverseMap']     = gammaParameterReverseMap;

        where

            model and parameters are dictionaries with keys
            equal to parameters (e.g. C_radius or H_gamma) and values set to prior (pymc.Uniform) 

            radiusParameterMap[C_radius]                          = C_radius parameter index
            radiusParameterReverseMap[C_radius parameter index]   = C_radius

            gammaParameterMap[C_gamma]                            = C_gamma parameter index
            gammaParameterReverseMap[C_gamma parameter index]     = C_gamma

    """

    if( 'includeSoluteDielectricAsParameter' in fixedParameters ):
        includeSoluteDielectricAsParameter =  fixedParameters['includeSoluteDielectricAsParameter']
    else:
        includeSoluteDielectricAsParameter = 0
    
    if( 'soluteDielectric' in fixedParameters ):
        soluteDielectric =  fixedParameters['soluteDielectric']
    else:
        soluteDielectric = 1.0
    
    if( 'solventDielectric' in fixedParameters ):
        solventDielectric =  fixedParameters['solventDielectric']
    else:
        solventDielectric = 78.3
    
    if( 'energyCalculations' in fixedParameters ):
        energyCalculations =  fixedParameters['energyCalculations']
    else:
        energyCalculations = 'Swig'

    print "energyCalculations                : %12s"   % (energyCalculations)
    print "soluteDielectric                  : %12.3f" % (soluteDielectric)
    print "solventDielectric                 : %12.3f" % (solventDielectric)
    print "includeSoluteDielectricAsParameter: %12d"   % (includeSoluteDielectricAsParameter)
  
    # Define priors for parameters.

    model                     = dict()

    parameters                = dict() # just the parameters

    radiusParameterMap        = dict() 
    radiusParameterReverseMap = dict() 
    radiusParameterIndex      = 0
    radiusParameters          = []

    gammaParameterMap         = dict() # just the parameters
    gammaParameterReverseMap  = dict() # just the parameters
    gammaParameterIndex       = 0
    gammaParameters           = []

    for (key, value) in initial_parameters.iteritems():

        (atomtype, parameter_name)  = key.split('_')

        if( atomtype in filterAtomType and filterAtomType[atomtype] ):

            if parameter_name == 'gamma':
    
                stochastic                                       = pymc.Uniform(key, value=value, lower=-10.0, upper=+10.0)
                gammaParameterMap[key]                           = gammaParameterIndex
                gammaParameterReverseMap[gammaParameterIndex]    = key
                gammaParameters.append( 4.184*value )
                gammaParameterIndex                             += 1
    
            elif parameter_name == 'radius':
    
                stochastic                                       = pymc.Uniform(key, value=value, lower=0.8, upper=3.0)
     
                radiusParameterMap[key]                          = radiusParameterIndex
                radiusParameterReverseMap[radiusParameterIndex]  = key
                radiusParameterIndex                            += 1
                radiusParameters.append( value*0.1 )
    
            else:
                raise Exception("Unrecognized parameter name: %s" % parameter_name)
    
            model[key]      = stochastic
            parameters[key] = stochastic

    if( includeSoluteDielectricAsParameter ):
        stochastic                     = pymc.Uniform('soluteDielectric', value=1.0, lower=0.5, upper=5.0)
        model['soluteDielectric']      = stochastic
        parameters['soluteDielectric'] = stochastic

    parameterMap                                 = dict()
    parameterMap['model']                        = model;
    parameterMap['stochastic']                   = parameters;
    parameterMap['radiusParameterMap']           = radiusParameterMap;
    parameterMap['radiusParameterReverseMap']    = radiusParameterReverseMap;
    parameterMap['gammaParameterMap']            = gammaParameterMap;
    parameterMap['gammaParameterReverseMap']     = gammaParameterReverseMap;

    print "\nradiusParameterMap "
    for (key, value) in radiusParameterMap.iteritems():
        print "   %12s %12s " % (key, value)

    print "\nradiusParameterReverseMap"
    for (key, value) in radiusParameterReverseMap.iteritems():
        print "   %12s %12s " % (key, value)

    print "\nradiusParameters"
    for value in radiusParameters:
        print "   %15.7e" % (value)

    print "\ngammaParameterMap"
    for (key, value) in gammaParameterMap.iteritems():
        print "   %12s %12s " % (key, value)

    print "\ngammaParameterReverseMap"
    for (key, value) in gammaParameterReverseMap.iteritems():
        print "   %12s %12s" % (key, value)

    print "\ngammaParameters"
    for value in gammaParameters:
        print "   %15.7e" % (value)
    print "\n"

    return parameterMap

#=============================================================================================
# Write GB/VI parameter file and call swigged routine gbvi.addMolecules( fileName ) to
# setup for parameterization run
#=============================================================================================

def writeGBVIParameterFile( molecules, parameterMap, gbviParameterFileName ):

    """

    ARGUMENTS

    molecules (list of OE molecules)  - list of OE molecules
    parameterMap (disc)               - output dict from createParameterMap()
    gbviParameterFileName (string)    - file name

    Output file has the following layout:

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
               
    RETURNS


    """

    # write molecules/parameter to file

    gbviParameterFile     = open( gbviParameterFileName, 'w')
    print "Opened %s" % (gbviParameterFileName)

    outputString      = "molecules "        + (" %8d\n" % len(molecules)) 
    gbviParameterFile.write( outputString )

    for (molecule_index, molecule) in enumerate(molecules):
        name              = OEGetSDData(molecule, 'name').strip()
        outputString      = "molecule %d\n" % (molecule_index)
        outputString     += "name %s\n" % (name)
        outputString     += setup_hydration_energy( molecule, parameterMap['radiusParameterMap'], parameterMap['gammaParameterMap'] )
        #print "%d %s" % (molecule_index,outputString)
        gbviParameterFile.write( outputString )

    gbviParameterFile.close()

    # add molecules to list of molecules used in Swig library gbvi
 
    gbvi.addMolecules( gbviParameterFileName )
    
#=============================================================================================
# Create PyMC model
#=============================================================================================

def create_model( fullMoleculeList, moleculePartition, parameterMap, includeSoluteDielectricAsParameter ):

    """

    ARGUMENTS

    molecules (list of OE molecules)         - list of OE molecules
    moleculePartition (int list)             - entries w/ 1 are used in the model
    parameterMap (dict)                      - output dict from createParameterMap()
    includeSoluteDielectricAsParameter (int) - if nonzero, include solute dielectric as a parameter

    RETURNS


    """

    # Define deterministic functions for hydration free energies.

    model              = parameterMap['model']
    parameters         = parameterMap['stochastic']
    radiusParameterMap = parameterMap['radiusParameterMap']
    gammaParameterMap  = parameterMap['gammaParameterMap']

    for (molecule_index, molecule) in enumerate(fullMoleculeList):

        if( moleculePartition[molecule_index] == 1 ):
    
            molecule_name = molecule.GetTitle()
            variable_name = "dg_gbvi_%08d" % molecule_index
    
            # Determine which parameters are involved in this molecule to limit number of parents for caching.
    
            parents       = dict()
            for atom in molecule.GetAtoms():
                atomtype = atom.GetStringData("gbvi_type") # GBVI atomtype
                for parameter_name in ['gamma', 'radius']:
                    stochastic_name          = '%s_%s' % (atomtype,parameter_name)
                    if( stochastic_name in parameters ):
                        parents[stochastic_name] = parameters[stochastic_name]
                    else:
                        print "create_model Warning: parameter=%s missing for %40s" % (stochastic_name, molecule_name )
    
            if( includeSoluteDielectricAsParameter ):
                parents['soluteDielectric'] = parameters['soluteDielectric']
    
            print "create_model %40s: %s" % (molecule_name, parents.keys() )
    
            # Create deterministic variable for computed hydration free energy.
    
            #if( energyCalculations == 'Swig' ):
            function = hydration_energy_factory_swig(molecule_index, radiusParameterMap, gammaParameterMap)
    
    #        if( energyCalculations == 'OpenMM' ):
    #            function = hydration_energy_factory_OpenMM(molecule)
    #
    #        if( energyCalculations == 'Test' ):
    #            function = hydration_energy_factory_test( molecule, molecule_index, radiusParameterMap, gammaParameterMap)
    
            model[variable_name] = pymc.Deterministic(eval=function,
                                                      name=variable_name,
                                                      parents=parents,
                                                      doc=molecule_name,
                                                      trace=True,
                                                      verbose=1,
                                                      dtype=float,
                                                      plot=False,
                                                      cache_depth=2)
    
    # Define error model
    log_sigma_min              = math.log(0.01) # kcal/mol
    log_sigma_max              = math.log(10.0) # kcal/mol
    log_sigma_guess            = math.log(1.0) # kcal/mol
    model['log_sigma']         = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
    model['sigma']             = pymc.Lambda('sigma', lambda log_sigma=model['log_sigma'] : math.exp(log_sigma) )    
    model['tau']               = pymc.Lambda('tau', lambda sigma=model['sigma'] : sigma**(-2) )

    for (molecule_index, molecule) in enumerate(fullMoleculeList):

        if( moleculePartition[molecule_index] == 1 ):
            molecule_name          = molecule.GetTitle()
            variable_name          = "dg_exp_%08d" % molecule_index
            dg_exp                 = float(OEGetSDData(molecule, 'dG(exp)')) # observed hydration free energy in kcal/mol
            print "Mol=%4d dG=%15.7e  %s " % (molecule_index, dg_exp, molecule_name )
            sys.stdout.flush()
            model[variable_name]   = pymc.Normal(mu=model['dg_gbvi_%08d' % molecule_index], tau=model['tau'], value=dg_exp, observed=True)        

    return

#=============================================================================================
# Read parameter sets from files (N_gamma.txt, N_radius.txt, ...) in
# directory mcmcDbName + '.txt/Chain_0'
#=============================================================================================

def readParameterSet( mcmcDbName, parameters ):

    """

    ARGUMENTS

    mcmcDbName  (string)                     - directory location for
                                               mcmc output
    parameters (dict)                        - keys of dict used to get parameter names
                                               (N_gamma.txt, N_radius, ...)

    RETURNS

    2d-list: parameter_sets[iterationIndex][key] = parameter value
    where iterationIndex is the MCMC iteration
          key is the paramter name (e.g., N_gamma) and
          parameter value is the value of the parameter at that
          MCMC iteration

    """

    # Load updated parameter sets.

    parameter_sets  = list()

    for key in parameters.keys():

        # Read parameters.
        filename = mcmcDbName + '.txt/Chain_0/%s.txt' % key 

        print "Reading trace for %20s from file %s" %( key, filename ) 

        infile = open(filename, 'r')
        lines  = infile.readlines()
        infile.close()

        # Discard header
        lines = lines[3:]

        # Insert parameter.

        for (index, line) in enumerate(lines):
            elements  = line.split()
            parameter = float(elements[0])
            
            while( len( parameter_sets ) <= index ):
                parameter_sets.append( dict() )
            parameter_sets[index][key] = parameter

#    if( includeSoluteDielectricAsParameter ):
#
#        key = 'soluteDielectric'
#
#        # Read parameters.
#        filename = mcmcDbName + '.txt/Chain_0/%s.txt' % key
#        print "Parameter %s from file %s" %( key, filename )
#        infile = open(filename, 'r')
#        lines = infile.readlines()
#        infile.close()
#        # Discard header
#        lines = lines[3:]
#        # Insert parameter.
#        for (index, line) in enumerate(lines):
#            elements = line.split()
#            parameter = float(elements[0])
#
#            try:
#                parameter_sets[index][key] = parameter
#            except Exception:
#                parameter_sets[index].append( dict() )
#                parameter_sets[key1][key2] = parameter

    print "Done reading parameters from file %s" %(mcmcDbName )
    sys.stdout.flush()

    return parameter_sets

#=============================================================================================
# Read parameter sets from files (N_gamma.txt, N_radius.txt, ...) in
# directory mcmcDbName + '.txt/Chain_0'
#=============================================================================================

def readParameterSetPerParameter( mcmcDbName, parameters ):

    """

    ARGUMENTS

    mcmcDbName  (string)                     - directory location for
                                               mcmc output
    parameters (dict)                        - keys of dict used to get parameter names
                                               (N_gamma.txt, N_radius, ...)

    RETURNS

    2d-list: parameter_sets[key][iterationIndex] = parameter value
    where iterationIndex is the MCMC iteration
          key is the paramter name (e.g., N_gamma) and
          parameter value is the value of the parameter at that
          MCMC iteration

    """

    # Load updated parameter sets.

    parameter_sets  = dict()

    for key in parameters.keys():

        # Read parameters.

        filename = mcmcDbName + '.txt/Chain_0/%s.txt' % key 

        print "Reading trace for %20s from file %s" %( key, filename ) 

        if( os.path.exists( filename ) ):
            parameter_sets[key] = list()
            infile              = open(filename, 'r')
            lines               = infile.readlines()
            infile.close()
    
            # Discard header
            lines = lines[3:]
    
            # Insert parameter.
    
            for (index, line) in enumerate(lines):
    
                elements  = line.split()
                parameter = float(elements[0])
                
                parameter_sets[key].append( parameter )

    if( includeSoluteDielectricAsParameter ):
        key                 = 'soluteDielectric'
        parameter_sets[key] = list()
        filename            = mcmcDbName + '.txt/Chain_0/%s.txt' % key
        print "Reading trace for %s from file %s" %( key, filename )
        infile              = open(filename, 'r')
        lines               = infile.readlines()
        infile.close()
        # Discard header
        lines = lines[3:]
        # Insert parameter.
        for (index, line) in enumerate(lines):
            elements = line.split()
            parameter = float(elements[0])

            parameter_sets[key].append( parameter )

    print "Done reading parameters from file %s" %(mcmcDbName )
    sys.stdout.flush()

    return parameter_sets

#=============================================================================================
# Print parameters to a file 
#=============================================================================================

def printParametersToFile( deltaParameterFileName, parameters, parameterMap ):

    """

    ARGUMENTS

    deltaParameterFileName (string)          - destination file name
    parameters (dict)                        - keys of dict used to get parameter names
                                               (N_gamma.txt, N_radius, ...)
    parameterMap (dict)

    Sample format:

            soluteDielectric    1.000
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
            
    RETURNS

    """

    gammaParameterMap           = parameterMap['gammaParameterMap']
    gammaParameterReverseMap    = parameterMap['gammaParameterReverseMap']

    radiusParameterMap          = parameterMap['radiusParameterMap']
    radiusParameterReverseMap   = parameterMap['radiusParameterReverseMap']

    deltaParameterFile          = open( deltaParameterFileName, 'w') 
    #print "Opened %s" % (deltaParameterFileName)

    if( 'soluteDielectric' in parameters ):
        soluteDielectric = parameters['soluteDielectric']
    else:
        soluteDielectric = 1.0

    outputString      = "soluteDielectric"  + (" %8.3f\n" % soluteDielectric ) 

    outputString     += "gamma " + (" %d\n" % len( gammaParameterMap )) 
    for index in range( len( gammaParameterMap ) ): 
        gammaKey                      = gammaParameterReverseMap[index]
        gamma                         = parameters[gammaKey]
        (atomType, parameter_name )   = gammaKey.split('_')
        outputString                 += "%3s %3d %15.7e\n" % (atomType, index, gamma*4.184) 
     

    outputString     += "radius " + (" %d\n" % len( radiusParameterMap )) 
    for index in range( len( radiusParameterMap ) ): 
        radiusKey                     = radiusParameterReverseMap[index]
        radius                        = parameters[radiusKey]
        (atomType, parameter_name )   = radiusKey.split('_')
        outputString                 += "%3s %3d %15.7e\n" % (atomType, index, 0.1*radius) 
     
    deltaParameterFile.write( outputString )
    deltaParameterFile.close()

#=============================================================================================
# Get energies for input parameter sets for all molecules 
# parameter sets that give Nans are printed to file
#=============================================================================================

def getEnergiesForParameterSets( parameter_sets, moleculeList, moleculePartition, experimentalEnergies, parameterMap, fileSuffix, resultsPath ):

    """

    ARGUMENTS

    parameter_sets (dict)                    - input parameter sets
    moleculeList (dict)                      - input dict of molecules
    moleculePartition (list)                 - input list -- same size as moleculeList:
                                                           moleculePartition[i] = -1 exclude molecule i
                                                                                   0 molecule i is in test set
                                                                                  >0 molecule i is in training set

    experimentalEnergies (float array)       - input experimental energies: experimentalEnergies[i] is value for molecule i 
    parameterMap                             - input parameterMap
    fileSuffix (string)                      - input file suffix string; molecules/parameter sets that give Nans for the
                                               energy are printed to file  resultsPath + '/gbviParameters/parameters' + fileSuffix + '_' + str(index) + '.txt
    resultsPath (string)                     -- instupu path (see fileSuffix entry above)

    RETURNS

    """

    parameters = parameterMap['stochastic']
    iterations = -1

    # get number of iterations

    iterations           = len( parameter_sets )
    printIndex           = int(iterations*0.01)
    if( printIndex < 1 ):
        printIndex = 1

    count                = 0

    # outer loop over parameter sets
    # inner loop over molecules
    #    calcuate energy for each parameter set/molecule and track difference
    #    w/ experimental value; report rms differences for training/test sets and
    #    all molecules 

    for (index, parameter_set) in enumerate([parameters] + parameter_sets): 
        if( count == printIndex ):
            count = 0
            
            # Compute energies with all molecules.
    
            trainErrors       = list()
            testErrors        = list()
            signed_errors     = list()
            printedParameters = 0 
            nanMolecules      = 0 
            trainRms          = 0
            testRms           = 0
            rms               = 0
    
            for (moleculeIndex, molecule) in enumerate(moleculeList): 
                if( moleculePartition[moleculeIndex] >= 0 ):
                    energy    = compute_gbvi_energy(moleculeIndex, parameterMap['radiusParameterMap'], parameterMap['gammaParameterMap'], parameter_set)
                    if( energy > 1.0e+05 ):
                        nanMolecules += 1
                        name          = OEGetSDData(molecule, 'name').strip()
                        print "Nan detected for molecule %4d parameter set=%6d %15.7e %s" % (moleculeIndex, index, energy, name)
                        if( printedParameters == 0 ):
                            parameterFileName  = resultsPath + '/gbviParameters/parameters' + fileSuffix + '_' + str(index) + '.txt'
                            printedParameters += 1
                            printParametersToFile( parameterFileName, parameter_set, parameterMap )
                    else:
                        delta     = energy - experimentalEnergies[moleculeIndex]
                        delta2    = delta*delta
                        rms      += delta2
                        signed_errors.append( delta )
                        if( moleculePartition[moleculeIndex] == 1 ):
                            trainErrors.append( delta )
                            trainRms += delta2
                        elif( moleculePartition[moleculeIndex] == 0 ):
                            testErrors.append( delta )
                            testRms  += delta2
     
            signed_errorsArray  = numpy.array(signed_errors)
            trainErrorsArray    = numpy.array(trainErrors)
            testErrorsArray     = numpy.array(testErrors)
    
            if( len( signed_errors)  > 0 ):
                rms                 = math.sqrt( rms/len(signed_errors) ) 
    
            if( len(trainErrors) > 0 ):
                trainRms            = math.sqrt( trainRms/len(trainErrors) ) 
    
            if( len(testErrors) > 0 ):
                testRms             = math.sqrt( testRms/len(testErrors) ) 
    
            print "%8d: mu[%12.3f  %12.3f %12.3f] std[%12.3f  %12.3f %12.3f] rms[%12.3f  %12.3f %12.3f] nans=%4d %d %d %d" % (index, signed_errorsArray.mean(), trainErrorsArray.mean(), testErrorsArray.mean(), signed_errorsArray.std(), trainErrorsArray.std(), testErrorsArray.std(), rms, trainRms, testRms, nanMolecules, len(signed_errors ), len(trainErrors), len(testErrors))
        count += 1

#=============================================================================================
# Check if dbPath exists; exit if missing
#=============================================================================================

def checkDbPath( dbPath, idString ):
    if( not os.path.exists(dbPath) ):
        print "Directory %s is missing -- %s" % (dbPath, idString)
        sys.exit(-1)
    else:
        print "Reading parameters from %s -- %s." % (dbPath, idString)

    return

#=============================================================================================
# Output parameters to a file
#=============================================================================================

def outputParameterSets( burnIn, parameter_sets, parameterFileName, skipArray ):

    """
    Output parameters sets to file, starting at 'burnIn' iteration w/ a stride == values
    in skipArray
    
    ARGUMENTS

    burnIn (int)                             - skip first 'burnIn' parameter sets
    parameter_sets (dict)                    - input parameter sets
    parameterFileName (string)               - destination  file name
    skipArray (int array)                    - for each value in skipArray write every skipArray[i]th parameter set
                                               for example, if burnIn = 50 and skipArray[0] = 10 and total number of
                                               parameter_sets is 100, then print parameter sets 50, 60, 70, 80, 90, 100
    """

    iterations = -1

    # get number of iterations

    for parameter in parameter_sets.keys( ): 
        parameterArray       = parameter_sets[parameter]
        iterations           = len( parameterArray ) + 1
        break

    # loop over iterations with stride = skipValue
    # start after burn in length

    print "%s iter=%d" % ( parameterFileName, iterations)
    for skipValue in skipArray:
        index  = burnIn
        while( index < iterations ):

            # gather parameters

            radiusOutputParameters = dict()
            gammaOutputParameters  = dict()
            soluteDielectric       = 1.0

            for parameter in sorted( parameter_sets.keys( )): 
                parameterValue               = parameter_sets[parameter][index-1]
                if( parameter == 'soluteDielectric' ):
                    soluteDielectric  = parameterValue
                elif( parameter.find('_') > -1 ):
                    (atomtype, parameter_name)   = parameter.split('_')

                    if( parameter_name == 'gamma'):
                        gammaOutputParameters[atomtype]  = parameterValue
                    elif( parameter_name == 'radius' ):
                        radiusOutputParameters[atomtype] = parameterValue
                    else:
                        print "parameter %s not recognized." % (parameter)
                        sys.exit(-1)
                else:
                    print "parameter %s not recognized." % (parameter)
                    sys.exit(-1)

            #output to file

            outputString        = "soluteDielectric %15.7f\n" % (soluteDielectric)
            for atomtype in sorted( radiusOutputParameters.keys() ):
                radius          = radiusOutputParameters[atomtype]
                gamma           = gammaOutputParameters[atomtype]
                outputString   += "%4s %12.3e %12.3e\n" % (atomtype, radius, gamma)

            outputParameterFileName = parameterFileName + '_' + str(index) + '.txt'
            parameterFile           = open( outputParameterFileName, 'w')
            parameterFile.write( outputString )
            print "Wrote parameters to file %s" % (outputParameterFileName)
            parameterFile.close()

            index += skipValue

    return
      
#=============================================================================================
# Get statistics on parameters 
#=============================================================================================

def analyzeParameterSets( burnIn, parameter_sets, skipArray ):

    """
    ARGUMENTS

    burnIn (int)                             - skip first 'burnIn' parameter sets
    parameter_sets (dict)                    - input parameter sets
    parameter_sets (dict)                    - input parameter sets
    skipArray (int array)                    - for each value in skipArray collect statistics for every skipArray[i]th parameter set
                                               for example, if burnIn = 50 and skipArray[0] = 10 and total number of
                                               parameter_sets is 100, then calculate the average, ... for
                                               parameter sets 50, 60, 70, 80, 90, 100

    RETURN
    
    skipDict[parameter][skipValue] = statList where
    parameter is C_gamma, C_radius, ...
    skipValue is entry from skipArray
    statList is list w/
        statList[0] average parameter values
        statList[1] std of parameter values
        statList[2] min parameter value
        statList[3] max parameter value
        statList[4] number of parameter values included in stats
        statList[5] burnIn value

    """

    skipDict = dict()
    for parameter in sorted( parameter_sets.keys( )): 
        parameterArray       = parameter_sets[parameter]
        skipDict[parameter]  = dict()
        for skipValue in skipArray:
            skipList             = list()
            statList             = list()
            index                = burnIn
            while( index < len( parameterArray ) ):
                skipList.append( parameterArray[index] )
                index += skipValue

            if( len( skipList ) > 2 ):
                statList.append( numpy.average( skipList ) ) 
                statList.append( numpy.std( skipList ) ) 
                statList.append( numpy.amin( skipList ) ) 
                statList.append( numpy.amax( skipList ) ) 
                statList.append( len( skipList ) ) 
                statList.append( skipValue ) 
                statList.append( burnIn ) 

                skipDict[parameter][skipValue] = statList

    return skipDict
      
#=============================================================================================
# MAIN
#=============================================================================================

if __name__=="__main__":

    # Create command-line argument options.

    usage_string = """\
    usage: %prog --types typefile --parameters paramfile --molecules molfile --burnIn burnIn_iterations --iterations MCMC_iterations --mcmcDb MCMC_db_name --includeSoluteDielectricAsParameter 0/1  --train_fraction (0-1) --seed random_number_generator_seed --performFit 0/1 --allMolecules (<=0, then all; else if allMolecules=x, then first x molecules (usefor testing) --analyzeParameters 0/1 
    
    example: %prog --types parameters/gbvi.types --parameters parameters/gbvi-am1bcc.parameters --molecules datasets/solvation.sdf
                   --iterations 150 --mcmcDb MCMC.txt

    If performFit == 0 and analyzeParameters is set, then just do analysis on earlier runs (specfied below) and exit
    If performFit == 0 and analyzeParameters is not set, then do analysis on current parameter sets
    If performFit != 0, then do MCMC fit and run analysis on resulting parameter sets; typical use case
    
    """
    version_string = "%prog %__version__"
    parser = OptionParser(usage=usage_string, version=version_string)

    parser.add_option("-t", "--types", metavar='TYPES',
                      action="store", type="string", dest='atomtypes_filename', default='',
                      help="Filename defining atomtypes as SMARTS atom matches.")

    parser.add_option("-p", "--parameters", metavar='PARAMETERS',
                      action="store", type="string", dest='parameters_filename', default='',
                      help="File containing initial parameter set.")

    parser.add_option("-m", "--molecules", metavar='MOLECULES',
                      action="store", type="string", dest='molecules_filename', default='',
                      help="Small molecule set (in any OpenEye compatible file format) containing 'dG(exp)' fields with experimental hydration free energies.")

    parser.add_option("-b", "--burnIn", metavar='BURN_IN',
                      action="store", type="int", dest='burnIn', default=10,
                      help="Burn-in iterations.")
    
    parser.add_option("-i", "--iterations", metavar='ITERATIONS',
                      action="store", type="int", dest='iterations', default=150,
                      help="MCMC iterations.")
    
    parser.add_option("-d", "--mcmcDb", metavar='MCMC_Db',
                      action="store", type="string", dest='mcmcDb', default='MCMC.txt',
                      help="MCMC db name.")
    
    parser.add_option("-e", "--includeSoluteDielectricAsParameter", metavar='SoluteDielectric',
                      action="store", type="int", dest='includeSoluteDielectricAsParameter', default=0,
                      help="Include inner dielectric as parameter.")

    parser.add_option("-f", "--train_fraction", metavar='TRAIN_FRACTION',
                      action="store", type="float", dest='training_fraction', default='0.9',
                      help="Fraction of original set of molecules to be used for training.")

    parser.add_option("-s", "--seed", metavar='RNG_SEED',
                      action="store", type="int", dest='seed', default='1',
                      help="RNG seed.")
 
    parser.add_option("-r", "--performFit", metavar='PERFORM_MCMC',
                      action="store", type="int", dest='performFit', default='1',
                      help="Perform MCMC fit.")
 
    parser.add_option("-l", "--allMolecules", metavar='allMolecules',
                      action="store", type="int", dest='allMolecules', default='0',
                      help="Number of molecules (<=0 == all).")
 
    parser.add_option("-a", "--analyzeParameters", metavar='analyzeParameters',
                      action="store", type="int", dest='analyzeParameters', default='0',
                      help="Analyze parameters (0/1).")
 
    # Parse command-line arguments.

    (options,args) = parser.parse_args()
    
    # Ensure all required options have been specified.

    if options.atomtypes_filename=='' or options.parameters_filename=='' or options.molecules_filename=='':
        parser.print_help()
        parser.error("All input files must be specified.")

    burnIn                             = options.burnIn
    mcmcIterations                     = options.iterations
    mcmcDbName                         = os.path.abspath(options.mcmcDb)
    dbPath                             = mcmcDbName + '.txt'
    includeSoluteDielectricAsParameter = options.includeSoluteDielectricAsParameter

    # echo arguments

    printString  = "Starting "                                 + sys.argv[0] + "\n"
    printString += '    atom types=<'                          + options.atomtypes_filename + ">\n"
    printString += '    parameters=<'                          + options.parameters_filename + ">\n"
    printString += '    trainFraction='                        + str(options.training_fraction) + ">\n"
    printString += '    molecule=<'                            + options.molecules_filename + ">\n"
    printString += '    burnIn= '                              + str(burnIn) + "\n"
    printString += '    iterations= '                          + str(mcmcIterations) + "\n"
    printString += '    mcmcDB=<'                              + mcmcDbName + ">\n"
    printString += '    seed '                                 + str(options.seed) + "\n"
    printString += '    performFit '                           + str(options.performFit) + "\n"
    printString += '    analyzeParameters '                    + str(options.analyzeParameters) + "\n"
    printString += '    allMolecules '                         + str(options.allMolecules) + "\n"
    printString += '    includeSoluteDielectricAsParameter=<'  + str(includeSoluteDielectricAsParameter) + ">\n"
    sys.stderr.write( printString )
 
    performFit                                                 = options.performFit
    allMolecules                                               = options.allMolecules
    train_fraction                                             = options.training_fraction
    random.seed( options.seed )
    fixedParameters                                            = dict()
    fixedParameters['includeSoluteDielectricAsParameter']      = includeSoluteDielectricAsParameter

    if( not includeSoluteDielectricAsParameter ):
        fixedParameters['soluteDielectric']                    = 1.0

    fixedParameters['solventDielectric']                       = 78.3
    analyzeParameters                                          = options.analyzeParameters

    # Construct atom typer

    atom_typer = AtomTyper(options.atomtypes_filename, "gbvi_type")
    #atom_typer.dump()
    
    # Read GB/VI parameters and create parameter maps

    initialParameters   = read_gbvi_parameters(options.parameters_filename)
    if( performFit == 0 and analyzeParameters ):
        dbDir            = '/home/friedrim/source/gbff/examples/gbvi/results/'
        paramDir         = '/home/friedrim/source/gbff/examples/gbvi/gbviParameters/'
        #dbArray          = [ 'MCMCReduced_0.80_9', 'MCMCReduced_0.80_10', 'MCMCReduced_0.80_11', 'MCMCReduced_0.80_12' ]
        #dbArray          = [ 'MCMCReduced_0.80_13', 'MCMCReduced_0.80_14', 'MCMCReduced_0.80_15', 'MCMCReduced_0.80_16' ]
        #dbArray          = [ 'MCMCReduced_0.80_17', 'MCMCReduced_0.80_18', 'MCMCReduced_0.80_19', 'MCMCReduced_0.80_20' ] # 80K runs/no filter/all/more tuning
        dbArray          = [ 'MCMCReduced_0.80_21', 'MCMCReduced_0.80_22','MCMCReduced_0.80_23','MCMCReduced_0.80_24' ]
        dbParameterHash  = dict()
        skipArray        = [ 20, 50, 100, 500, 1000 ]
        printSkipArray   = [ 10 ]
        for db in dbArray:
            mcmcDbName                = dbDir + db
            parameter_sets            = readParameterSetPerParameter( mcmcDbName, initialParameters )
            dbParameterHash[db]       = analyzeParameterSets( burnIn, parameter_sets, skipArray )
            parameterOutputFileName   = paramDir + db
            print "Output params for %s" % (db)
            outputParameterSets( burnIn, parameter_sets, parameterOutputFileName, printSkipArray )

        # get active keys

        keyDict          = dict()
        for key in sorted( initialParameters.keys() ):
            for skip in skipArray:
                for db in sorted( dbParameterHash.keys() ):
                    skipDict = dbParameterHash[db]
                    if( key in skipDict and skip in skipDict[key] ):
                        keyDict[key] = 1

        if( includeSoluteDielectricAsParameter ):
            keyDict['soluteDielectric'] = 1

        for key in sorted( keyDict.keys() ):
            print "\n%12s" % (key)
            for skip in skipArray:
                outputString = "%4d " % (skip)
                for db in sorted( dbParameterHash.keys() ):
                    skipDict = dbParameterHash[db]
                    if( key in skipDict and skip in skipDict[key] ):
                        outputString += "[%8.3f %8.3f] " % ( skipDict[key][skip][0], skipDict[key][skip][1] )
                print "   %s" % (outputString)
                
        
        sys.exit(0)

    # exclude/include parameters
    # used to test if removing some atom types (e.g. halogens), would imnprove fits
    
    excludeMarker        = 1
    includeMarker        = 1
    filterAtomType       = dict()
    filterAtomType['Br'] = excludeMarker
    filterAtomType['C']  = includeMarker
    filterAtomType['Cl'] = excludeMarker
    filterAtomType['F']  = excludeMarker
    filterAtomType['H']  = includeMarker
    filterAtomType['I']  = excludeMarker
    filterAtomType['N']  = includeMarker
    filterAtomType['O']  = includeMarker
    filterAtomType['P']  = excludeMarker
    filterAtomType['S']  = includeMarker

    # create parameter map, load molecules
    parameterMap         = createParameterMap( initialParameters, fixedParameters, filterAtomType )
    fullMoleculeList     = loadAndTypeMolecules( options.molecules_filename, allMolecules, mcmcDbName, atom_typer )

    # partition molecules into training/test sets

    if( train_fraction < 1.0 ):
        moleculePartition = partitionMoleculeList( len(fullMoleculeList), train_fraction )
        filterMolecules( filterAtomType, fullMoleculeList, moleculePartition )
    else:
        moleculePartition = [1]*len(fullMoleculeList)

    resultsPath               = os.path.dirname(mcmcDbName)
    fileSuffix                = str(options.training_fraction) + '_' + str(options.seed)
    gbviParameterFileName     = resultsPath + '/gbviParameterFile_' + fileSuffix + '.txt'
    writeGBVIParameterFile( fullMoleculeList, parameterMap, gbviParameterFileName )
    gbvi.addMolecules( gbviParameterFileName )

    # Compute energies for all molecules.

    start_time                = time.time()
    energies                  = list()
    experimentalEnergies      = list()

    signed_errors             = numpy.zeros([len(fullMoleculeList)], numpy.float64)
    trainErrors               = list()
    testErrors                = list()

    trainingMolecules         = list()
    testMolecules             = list()

    for (i, molecule) in enumerate(fullMoleculeList):
        name                        = OEGetSDData(molecule, 'name').strip()
        dg_exp                      = float(OEGetSDData(molecule, 'dG(exp)'))
        experimentalEnergies.append( dg_exp )
        #print "%3d %2d Name %s " % ( i, moleculePartition[i], name )
        if( moleculePartition[i] >= 0 ):
            energy                      = compute_gbvi_energy(i, parameterMap['radiusParameterMap'], parameterMap['gammaParameterMap'], initialParameters)
            energies.append( energy )
            delta                       = energy - dg_exp
            signed_errors[i]            = delta
            outstring                   = "%4d %48s %12.3f  %12.3f %12.3f" % (i,name, delta, dg_exp, energy)
            if( moleculePartition[i]    == 1 ):
                trainErrors.append( delta )
                trainingMolecules.append( molecule )
            elif( moleculePartition[i] == 0 ):
                testErrors.append( delta )
                outstring              += " *"
                testMolecules.append( molecule )
            elif( moleculePartition[i] == -1 ):
                outstring              += " X"
    
            print outstring

    end_time     = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time
    
    trainErrorsArray = numpy.array(trainErrors)
    testErrorsArray  = numpy.array(testErrors)
    print "Initial          RMS error %12.3f kcal/mol %4d molecules" % (signed_errors.std(), len(signed_errors))
    print "Initial training RMS error %12.3f kcal/mol %4d molecules" % (trainErrorsArray.std(),len(trainErrors))
    print "Initial test     RMS error %12.3f kcal/mol %4d molecules" % (testErrorsArray.std(), len(testErrors))
    sys.stdout.flush()

    # Create MCMC model.

    create_model( fullMoleculeList, moleculePartition, parameterMap, includeSoluteDielectricAsParameter )
    model      = parameterMap['model']
    parameters = parameterMap['stochastic']

    #dbPath     = os.path.dirname(mcmcDbName)
    if( performFit ):

        # MCMC complains if the db directory exists

        if( os.path.exists(dbPath) ):
            print "Removing directory %s" % (dbPath)
            shutil.rmtree(dbPath)
    
        # Sample models.

        from pymc import MCMC
        sampler = MCMC( model, db='txt', name=mcmcDbName, verbose=0)

        start_time   = time.time()
        localtime    = time.strftime("%b %d %Y %H:%M:%S", time.localtime(time.time()))
        print "Local start time :", localtime

        #sampler.sample(iter=mcmcIterations, burn=1000, save_interval=1, verbose=False)
        sampler.sample(iter=(mcmcIterations+burnIn), burn=burnIn, tune_interval=500, tune_throughout=True, save_interval=100, verbose=0)

        elapsed_time = time.time()- start_time
        localtime    = time.strftime("%b %d %Y %H:%M:%S", time.localtime(time.time()))
        print "Local end time : %s elapsed=%.3f for %d (%d %d) total iterations" % (localtime, elapsed_time, (burnIn+mcmcIterations), burnIn, mcmcIterations)
        sys.stdout.flush()

    else:
        if( not os.path.exists(dbPath) ):
            print "Directory %s is missing." % (dbPath)
            sys.exit(-1)
        else:
            print "Reading parameters from %s." % (dbPath)

    parameter_sets = readParameterSet( mcmcDbName, parameters )
    getEnergiesForParameterSets( parameter_sets, fullMoleculeList, moleculePartition, experimentalEnergies, parameterMap, fileSuffix, resultsPath )
