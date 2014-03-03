#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
evaluate-gbvi.py

Evaluate the GBVI model on hydration free energies of small molecules for multiple iterations of the Markov chain.

"""
#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import sys,string
from openeye.oechem import *
from optparse import OptionParser # For parsing of command line arguments

import os
import math
import numpy
import simtk.openmm as openmm
import simtk.unit as units

import openeye.oechem
import openeye.oequacpac
import openeye.oeiupac

from openeye.oechem import *
from openeye.oequacpac import *
from openeye.oeszybki import *
from openeye.oeiupac import *

import time
import pymc 

#=============================================================================================
# Load OpenMM plugins.
#=============================================================================================

print "Loading OpenMM plugins..."

openmm.Platform.loadPluginsFromDirectory(os.path.join(os.environ['OPENMM_INSTALL_DIR'], 'lib'))
openmm.Platform.loadPluginsFromDirectory(os.path.join(os.environ['OPENMM_INSTALL_DIR'], 'lib', 'plugins'))

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
# Utility routines
#=============================================================================================

def read_gbvi_parameters(filename):
        """
        Read a GBVI parameter set from a file.

        ARGUMENTS

        filename (string) - the filename to read parameters from

        RETURNS

        parameters (dict) - parameters[(atomtype,parameter_name)] contains the dimensionless parameter 
        
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
                parameters['%s_%s' % (atomtype,'gamma')] = float(gamma)

        return parameters                

#=============================================================================================
# Computation of hydration free energies
#=============================================================================================

def function(x):
    (molecule, parameters) = x
    return compute_hydration_energy(molecule, parameters)    

def compute_hydration_energies_parallel(molecules, parameters):
    import multiprocessing

    # Create processor pool.
    nprocs = 8
    pool = multiprocessing.Pool(processes=nprocs)

    x = list()
    for molecule in molecules:
        x.append( (molecule, parameters) )

    # Distribute calculation.
    results = pool.map(function, x)

    return results

def compute_hydration_energies(molecules, parameters):
    """
    Compute solvation energies of all specified molecules using given parameter set.

    ARGUMENTS

    molecules (list of OEMol) - molecules with atom types
    parameters (dict) - parameters for atom types

    RETURNS

    energies (dict) - energies[molecule] is the computed solvation energy of given molecule

    """

    energies = dict() # energies[index] is the computed solvation energy of molecules[index]

    platform = openmm.Platform.getPlatformByName("Reference")

    for molecule in molecules:
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
        gbvi_force.setSolventDielectric(78)

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
        energies[molecule] = state.getPotentialEnergy()

    return energies

def compute_hydration_energy(molecule, parameters, platform_name="Reference"):
    """
    Compute hydration energy of a specified molecule given the specified GBVI parameter set.

    ARGUMENTS

    molecule (OEMol) - molecule with GBVI atom types
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
    # gbvi_force = openmm.GBVISoftcoreForce()
    gbvi_force = openmm.GBVIForce()   
    gbvi_force.setNonbondedMethod(openmm.GBVIForce.NoCutoff) # set no cutoff
    gbvi_force.setSoluteDielectric(1)
    gbvi_force.setSolventDielectric(78)
    
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

def hydration_energy_factory(molecule):
    def hydration_energy(**parameters):
        return compute_hydration_energy(molecule, parameters, platform_name="Reference")
    return hydration_energy

#=============================================================================================
# PyMC model
#=============================================================================================

def testfun(molecule_index, *x):
    print molecule_index
    return molecule_index

def create_model(molecules, initial_parameters):

    # Define priors for parameters.
    model = dict()
    parameters = dict() # just the parameters
    for (key, value) in initial_parameters.iteritems():
        (atomtype, parameter_name) = key.split('_')
        if parameter_name == 'gamma':
            stochastic = pymc.Uniform(key, value=value, lower=-10.0, upper=+10.0)
        elif parameter_name == 'radius':
            stochastic = pymc.Uniform(key, value=value, lower=1.0, upper=3.0)
        else:
            raise Exception("Unrecognized parameter name: %s" % parameter_name)
        model[key] = stochastic
        parameters[key] = stochastic

    # Define deterministic functions for hydration free energies.
    for (molecule_index, molecule) in enumerate(molecules):
        molecule_name = molecule.GetTitle()
        variable_name = "dg_gbvi_%08d" % molecule_index
        # Determine which parameters are involved in this molecule to limit number of parents for caching.
        parents = dict()
        for atom in molecule.GetAtoms():
            atomtype = atom.GetStringData("gbvi_type") # GBVI atomtype
            for parameter_name in ['gamma', 'radius']:
                stochastic_name = '%s_%s' % (atomtype,parameter_name)
                parents[stochastic_name] = parameters[stochastic_name]
        print "%s : " % molecule_name,
        print parents.keys()
        # Create deterministic variable for computed hydration free energy.
        function = hydration_energy_factory(molecule)
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
    log_sigma_min = math.log(0.01) # kcal/mol
    log_sigma_max = math.log(10.0) # kcal/mol
    log_sigma_guess = math.log(0.2) # kcal/mol
    model['log_sigma'] = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
    model['sigma'] = pymc.Lambda('sigma', lambda log_sigma=model['log_sigma'] : math.exp(log_sigma) )    
    model['tau'] = pymc.Lambda('tau', lambda sigma=model['sigma'] : sigma**(-2) )
    for (molecule_index, molecule) in enumerate(molecules):
        molecule_name = molecule.GetTitle()
        variable_name = "dg_exp_%08d" % molecule_index
        dg_exp = float(OEGetSDData(molecule, 'dG(exp)')) # observed hydration free energy in kcal/mol
        model[variable_name] = pymc.Normal(mu=model['dg_gbvi_%08d' % molecule_index], tau=model['tau'], value=dg_exp, observed=True)        

    return model

#=============================================================================================
# MAIN
#=============================================================================================

if __name__=="__main__":

    # Create command-line argument options.
    usage_string = """\
    usage: %prog --types typefile --parameters paramfile --molecules molfile
    
    example: %prog --types parameters/gbvi.types --parameters parameters/gbvi-am1bcc.parameters --molecules datasets/solvation.sdf  --mcmcDb MCMC_db_name
    
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

    parser.add_option("-d", "--mcmcDb", metavar='MCMC_Db',
                      action="store", type="string", dest='mcmcDb', default='',
                      help="MCMC db name.")

    
    # Parse command-line arguments.
    (options,args) = parser.parse_args()
    
    # Ensure all required options have been specified.
    if options.atomtypes_filename=='' or options.parameters_filename=='' or options.molecules_filename=='' or options.mcmcDb == '':
        parser.print_help()
        parser.error("All input files must be specified.")

    # Read GBVI parameters.
    parameters = read_gbvi_parameters(options.parameters_filename)

    mcmcDbName     = options.mcmcDb
    printString  = "Starting " + sys.argv[0] + "\n"
    printString += '    atom types=<'   + options.atomtypes_filename + ">\n"
    printString += '    parameters=<'   + options.parameters_filename + ">\n"
    printString += '    molecule=<'     + options.molecules_filename + ">\n"
    printString += '    mcmcDB=<'       + options.mcmcDb + ">\n"
    sys.stderr.write( printString )
    sys.stdout.write( printString )

        
    # Construct atom typer.
    atom_typer = AtomTyper(options.atomtypes_filename, "gbvi_type")
    
    # Load and type all molecules in the specified dataset.
    print "Loading and typing all molecules in dataset..."
    start_time = time.time()
    molecules = list()
    input_molstream = oemolistream(options.molecules_filename)
    molecule = OECreateOEGraphMol()
    while OEReadMolecule(input_molstream, molecule):
        # Get molecule name.
        name = OEGetSDData(molecule, 'name').strip()
        molecule.SetTitle(name)
        # Append to list.
        molecule_copy = OEMol(molecule)
        molecules.append(molecule_copy)
    input_molstream.close()
    print "%d molecules read" % len(molecules)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time

    # Add explicit hydrogens.
    for molecule in molecules:
        openeye.oechem.OEAddExplicitHydrogens(molecule)    

    # Build a conformation for all molecules with Omega.
    print "Building conformations for all molecules..."    
    import openeye.oeomega
    omega = openeye.oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetFromCT(True)
    for molecule in molecules:
        #omega.SetFixMol(molecule)
        omega(molecule)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time

    # Regularize all molecules through writing as mol2.
    print "Regularizing all molecules..."
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time
    
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
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time
    print "%d molecules remaining" % len(molecules)
    
    # Type all molecules with GAFF parameters.
    start_time = time.time()
    typed_molecules = list()
    untyped_molecules = list()
    for molecule in molecules:
        # Assign GBVI types according to SMARTS rules.
        try:
            atom_typer.assignTypes(molecule)
            typed_molecules.append(OEGraphMol(molecule))
            #atom_typer.debugTypes(molecule)
        except AtomTyper.TypingException as exception:
            print name        
            print exception
            untyped_molecules.append(OEGraphMol(molecule))        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%d molecules correctly typed" % (len(typed_molecules))
    print "%d molecules missing some types" % (len(untyped_molecules))
    print "%.3f s elapsed" % elapsed_time

    # Load updated parameter sets.
    parameter_sets  = list()
    for key in parameters.keys():
        # Read parameters.
        filename = mcmcDbName + '.txt/Chain_0/%s.txt' % key
        print "Parameter %s from file %s" %( key, filename ) 
        infile = open(filename, 'r')
        lines = infile.readlines()
        infile.close()
        # Discard header
        lines = lines[3:]
        # Insert parameter.
        for (index, line) in enumerate(lines):
            elements = line.split()
            parameter = float(elements[0])
            try:
                parameter_sets[index][key] = parameter
            except Exception:
                parameter_sets.append( dict() )
                parameter_sets[index][key] = parameter

    for (index, parameter_set) in enumerate([parameters] + parameter_sets): # skip some
    #for (index, parameter_set) in enumerate([parameters] + parameter_sets[::10]): # skip some
        
        # Compute energies with all molecules.
        print "Computing all energies..."
        start_time = time.time()
        energies = compute_hydration_energies(typed_molecules, parameter_set)
        #energies = compute_hydration_energies_parallel(typed_molecules, parameter_set)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print "%.3f s elapsed" % elapsed_time

#        # Print comparison.
#        for molecule in typed_molecules:
#            # Get metadata.
#            name = OEGetSDData(molecule, 'name').strip()
#            dg_exp = float(OEGetSDData(molecule, 'dG(exp)')) * units.kilocalories_per_mole            
#            # Form output.
#            outstring = "%48s %8.3f %8.3f" % (name, dg_exp / units.kilocalories_per_mole, energies[molecule] / units.kilocalories_per_mole)            
#            print outstring

        # Print summary statistics.
        signed_errors = numpy.zeros([len(typed_molecules)], numpy.float64)
        for (i, molecule) in enumerate(typed_molecules):
            # Get metadata.
            name = OEGetSDData(molecule, 'name').strip()
            energy = energies[molecule] / units.kilocalories_per_mole
            if( math.isnan(energy) ):
                   print "%5d dG: nan %8.3f %s" % (i, dg_exp / units.kilocalories_per_mole, name)
            else:
                try:
                   dg_exp = float(OEGetSDData(molecule, 'dG(exp)')) * units.kilocalories_per_mole
                   signed_errors[i] = energies[molecule] / units.kilocalories_per_mole - dg_exp / units.kilocalories_per_mole
                except:
                    print "Problem getting dG(exp) for molecule %d %s" % (i, name)

        print "iteration %8d : RMS error %8.3f kcal/mol" % (index, signed_errors.std())

