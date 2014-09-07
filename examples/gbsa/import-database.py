#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
parameterize-gbsa.py

Parameterize the GBSA model on hydration free energies of small molecules using Bayesian inference
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
import os
import os.path
import time
import math
import copy
import tempfile

from optparse import OptionParser # For parsing of command line arguments

import numpy

import simtk.openmm as openmm
import simtk.unit as units
import simtk.openmm.app as app

import openeye.oechem
import openeye.oeomega
import openeye.oequacpac

# OpenEye toolkit
from openeye import oechem
from openeye import oequacpac
from openeye import oeiupac
from openeye import oeomega

import pymc

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
            return "Atom not assigned: %6d %8s" % (self.atom.GetIdx(), oechem.OEGetAtomicSymbol(self.atom.GetAtomicNum()))

    def __init__(self, infileName, tagname):
        self.pattyTag = oechem.OEGetTag(tagname)
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
                pat = oechem.OESubSearch()
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
        oechem.OEAssignAromaticFlags(mol)
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
            print "%6d %8s %8s" % (atom.GetIdx(),oechem.OEGetAtomicSymbol(atom.GetAtomicNum()),atom.GetStringData(self.pattyTag))

    def getTypeList(self,mol):
        typeList = []
        for atom in mol.GetAtoms():
            typeList.append(atom.GetStringData(self.pattyTag))
        return typeList

#=============================================================================================
# Utility routines
#=============================================================================================

def read_gbsa_parameters(filename):
        """
        Read a GBSA parameter set from a file.

        ARGUMENTS

        filename (string) - the filename to read parameters from

        RETURNS

        parameters (dict) - parameters[(atomtype,parameter_name)] contains the dimensionless parameter

        TODO

        * Replace this with a standard format?

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
                [atomtype, radius, scalingFactor] = elements
                parameters['%s_%s' % (atomtype,'radius')] = float(radius)
                parameters['%s_%s' % (atomtype,'scalingFactor')] = float(scalingFactor)

        return parameters

#=============================================================================================
# Computation of hydration free energies
#=============================================================================================

def compute_hydration_energies(database, parameters):
    """
    Compute solvation energies of a set of molecules given a GBSA parameter set.

    ARGUMENTS

    molecules (list of OEMol) - molecules with GBSA assigned atom types in type field
    parameters (dict) - dictionary of GBSA parameters keyed on GBSA atom types

    RETURNS

    energies (dict) - energies[molecule] is the computed solvation energy of given molecule

    """

    energies = dict() # energies[index] is the computed solvation energy of molecules[index]

    platform = openmm.Platform.getPlatformByName("Reference")

    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']

        # Retrieve OpenMM System.
        # Make deep copy.
        system = copy.deepcopy(entry['system'])

        # Get nonbonded force.
        forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
        nonbonded_force = forces['NonbondedForce']

        # Add GBSA term
        gbsa_force = openmm.GBSAOBCForce()
        gbsa_force.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff) # set no cutoff
        gbsa_force.setSoluteDielectric(1)
        gbsa_force.setSolventDielectric(78)

        # Build indexable list of atoms.
        atoms = [atom for atom in molecule.GetAtoms()]

        # Assign GBSA parameters.
        for (atom_index, atom) in enumerate(atoms):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
            atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
            radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
            scalingFactor = parameters['%s_%s' % (atomtype, 'scalingFactor')] * units.kilocalories_per_mole
            gbsa_force.addParticle(charge, radius, scalingFactor)

        # Add the force to the system.
        system.addForce(gbsa_force)

        timestep = 1.0 * units.femtosecond # arbitrary
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(system, integrator, platform)

        # Set the coordinates.
        positions = entry['positions']
        context.setPositions(positions)

        # Get the energy with GB.
        state = context.getState(getEnergy=True)
        energies[molecule] = state.getPotentialEnergy()

        # Clean up.
        del context, integrator

        # Compute the energy without GB.
        system = entry['system']
        timestep = 1.0 * units.femtosecond # arbitrary
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(system, integrator, platform)
        positions = entry['positions']
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        energies[molecule] -= state.getPotentialEnergy()
        del context, integrator

    return energies

def compute_hydration_energy(entry, parameters, platform_name="Reference"):
    """
    Compute hydration energy of a single molecule given a GBSA parameter set.

    ARGUMENTS

    molecule (OEMol) - molecule with GBSA atom types
    parameters (dict) - parameters for GBSA atom types

    RETURNS

    energy (float) - hydration energy in kcal/mol

    """

    platform = openmm.Platform.getPlatformByName(platform_name)

    # Retrieve molecule.
    molecule = entry['molecule']

    # Retrieve OpenMM System.
    # Make deep copy.
    system = copy.deepcopy(entry['system'])

    # Get nonbonded force.
    forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
    nonbonded_force = forces['NonbondedForce']

    # Add GBSA term
    gbsa_force = openmm.GBSAOBCForce()
    gbsa_force.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff) # set no cutoff
    gbsa_force.setSoluteDielectric(1)
    gbsa_force.setSolventDielectric(78)

    # Build indexable list of atoms.
    atoms = [atom for atom in molecule.GetAtoms()]

    # Assign GBSA parameters.
    for (atom_index, atom) in enumerate(atoms):
        [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
        atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
        try:
            radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
            scalingFactor = parameters['%s_%s' % (atomtype, 'scalingFactor')] * units.kilocalories_per_mole
        except Exception, exception:
            print "Cannot find parameters for atomtype '%s' in molecule '%s'" % (atomtype, molecule.GetTitle())
            print parameters.keys()
            raise exception

        gbsa_force.addParticle(charge, radius, scalingFactor) #

    # Add the force to the system.
    system.addForce(gbsa_force)

    # Create OpenMM Context.
    timestep = 1.0 * units.femtosecond # arbitrary
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator, platform)

    # Set the coordinates.
    positions = entry['positions']
    context.setPositions(positions)

    # Get the energy
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() / units.kilocalories_per_mole
    if numpy.isnan(energy):
        energy = +1e6;

    # Clean up.
    del context, integrator

    # Compute the energy without GB.
    system = entry['system']
    timestep = 1.0 * units.femtosecond # arbitrary
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator, platform)
    positions = entry['positions']
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    energy -= state.getPotentialEnergy() / units.kilocalories_per_mole
    del context, integrator

    return energy

def hydration_energy_factory(entry):
    def hydration_energy(**parameters):
        return compute_hydration_energy(entry, parameters, platform_name="CPU")
    return hydration_energy

#=============================================================================================
# PyMC model
#=============================================================================================

def testfun(molecule_index, *x):
    print molecule_index
    return molecule_index

def create_model(database, initial_parameters):

    # Define priors for parameters.
    model = dict()
    parameters = dict() # just the parameters
    for (key, value) in initial_parameters.iteritems():
        (atomtype, parameter_name) = key.split('_')
        if parameter_name == 'scalingFactor':
            stochastic = pymc.Uniform(key, value=value, lower=+0.01, upper=+1.0)
        elif parameter_name == 'radius':
            stochastic = pymc.Uniform(key, value=value, lower=0.5, upper=3.5)
        else:
            raise Exception("Unrecognized parameter name: %s" % parameter_name)
        model[key] = stochastic
        parameters[key] = stochastic

    # Define deterministic functions for hydration free energies.
    cid_list = database.keys()
    for (molecule_index, cid) in enumerate(cid_list):
        entry = database[cid]
        molecule = entry['molecule']

        molecule_name = molecule.GetTitle()
        variable_name = "dg_gbsa_%s" % cid
        # Determine which parameters are involved in this molecule to limit number of parents for caching.
        parents = dict()
        for atom in molecule.GetAtoms():
            atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
            for parameter_name in ['scalingFactor', 'radius']:
                stochastic_name = '%s_%s' % (atomtype,parameter_name)
                parents[stochastic_name] = parameters[stochastic_name]
        print "%s : " % molecule_name,
        print parents.keys()
        # Create deterministic variable for computed hydration free energy.
        function = hydration_energy_factory(entry)
        model[variable_name] = pymc.Deterministic(eval=function,
                                                  name=variable_name,
                                                  parents=parents,
                                                  doc=cid,
                                                  trace=True,
                                                  verbose=1,
                                                  dtype=float,
                                                  plot=False,
                                                  cache_depth=2)

    # Define error model
    log_sigma_min              = math.log(0.01) # kcal/mol
    log_sigma_max              = math.log(10.0) # kcal/mol
    log_sigma_guess            = math.log(0.2) # kcal/mol
    model['log_sigma']         = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
    model['sigma']             = pymc.Lambda('sigma', lambda log_sigma=model['log_sigma'] : math.exp(log_sigma) )
    model['tau']               = pymc.Lambda('tau', lambda sigma=model['sigma'] : sigma**(-2) )
    cid_list = database.keys()
    for (molecule_index, cid) in enumerate(cid_list):
        entry = database[cid]
        molecule = entry['molecule']

        molecule_name          = molecule.GetTitle()
        variable_name          = "dg_exp_%s" % cid
        dg_exp                 = float(molecule.GetData('expt')) # observed hydration free energy in kcal/mol
        ddg_exp                 = float(molecule.GetData('d_expt')) # observed hydration free energy uncertainty in kcal/mol
        #model[variable_name]   = pymc.Normal(variable_name, mu=model['dg_gbsa_%08d' % molecule_index], tau=model['tau'], value=expt, observed=True)
        model['tau_%s' % cid] = pymc.Lambda('tau_%s' % cid, lambda sigma=model['sigma'] : 1.0 / (sigma**2 + ddg_exp**2) )
        model[variable_name]   = pymc.Normal(variable_name, mu=model['dg_gbsa_%s' % cid], tau=model['tau_%s' % cid], value=dg_exp, observed=True)

    return model

def print_file(filename):
    infile = open(filename, 'r')
    print infile.read()
    infile.close()

#=============================================================================================
# MAIN
#=============================================================================================

if __name__=="__main__":

    # Create command-line argument options.
    usage_string = """\
    usage: %prog --types typefile --parameters paramfile --database database --iterations MCMC_iterations --mcmcout MCMC_db_name

    example: %prog --types parameters/gbsa.types --parameters parameters/gbsa-am1bcc.parameters --database datasets/FreeSolv/v0.3/database.pickle --iterations 150 --mcmcout MCMC --verbose

    """
    version_string = "%prog %__version__"
    parser = OptionParser(usage=usage_string, version=version_string)

    parser.add_option("-t", "--types", metavar='TYPES',
                      action="store", type="string", dest='atomtypes_filename', default='',
                      help="Filename defining atomtypes as SMARTS atom matches.")

    parser.add_option("-p", "--parameters", metavar='PARAMETERS',
                      action="store", type="string", dest='parameters_filename', default='',
                      help="File containing initial parameter set.")

    parser.add_option("-d", "--database", metavar='DATABASE',
                      action="store", type="string", dest='database_filename', default='',
                      help="Python pickle file of database with molecule names, SMILES strings, hydration free energies, and experimental uncertainties (FreeSolv format).")

    parser.add_option("-i", "--iterations", metavar='ITERATIONS',
                      action="store", type="int", dest='iterations', default=150,
                      help="MCMC iterations.")

    parser.add_option("-o", "--mcmcout", metavar='MCMCOUT',
                      action="store", type="string", dest='mcmcout', default='MCMC',
                      help="MCMC output database name.")

    parser.add_option("-v", "--verbose", metavar='VERBOSE',
                      action="store_true", dest='verbose', default=False,
                      help="Verbosity flag.")

    # Parse command-line arguments.
    (options,args) = parser.parse_args()

    # Ensure all required options have been specified.
    if options.atomtypes_filename=='' or options.parameters_filename=='' or options.database_filename=='':
        parser.print_help()
        parser.error("All input files must be specified.")

    # Set verbosity.
    verbose = options.verbose

    # Read GBSA parameters.
    parameters = read_gbsa_parameters(options.parameters_filename)
    print parameters

    mcmcIterations = options.iterations
    mcmcDbName     = os.path.abspath(options.mcmcout)

    # Construct atom typer.
    atom_typer = AtomTyper(options.atomtypes_filename, "gbsa_type")
    #atom_typer.dump()

    # Open database.
    import pickle
    database = pickle.load(open(options.database_filename, 'r'))

    # DEBUG: Create a small subset.
    subset_size = 20
    cid_list = database.keys()
    database = dict((k, database[k]) for k in cid_list[0:subset_size])

    # Process all molecules in the dataset.
    start_time = time.time()
    for cid in database.keys():
        # Get database entry.
        entry = database[cid]

        # Read relevant entry data.
        smiles = entry['smiles']
        iupac_name = entry['iupac']
        experimental_DeltaG = entry['expt'] * units.kilocalories_per_mole
        experimental_dDeltaG = entry['d_expt'] * units.kilocalories_per_mole

        # Create OpenEye molecule representation.
        molecule = openeye.oechem.OEGraphMol()
        openeye.oechem.OEParseSmiles(molecule, smiles)

        # Set properties.
        molecule.SetTitle(iupac_name)
        molecule.SetData('smiles', smiles)
        molecule.SetData('cid', cid)
        molecule.SetData('expt', experimental_DeltaG / units.kilocalories_per_mole)
        molecule.SetData('d_expt', experimental_dDeltaG / units.kilocalories_per_mole)

        # Add explicit hydrogens.
        oechem.OEAddExplicitHydrogens(molecule)

        # Store molecule.
        entry['molecule'] = oechem.OEMol(molecule)

    print "%d molecules read" % len(database.keys())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time

    # Build a conformation for all molecules with Omega.
    print "Building conformations for all molecules..."
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetFromCT(True)
    # Add explicit hydrogens.
    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']
        if verbose: print "  " + molecule.GetTitle()
        omega(molecule)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time

    # Create OpenMM System objects.
    import gaff2xml
    print "Running Antechamber..."
    original_directory = os.getcwd()
    working_directory = tempfile.mkdtemp()
    os.chdir(working_directory)
    start_time = time.time()
    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']

        if verbose: print "  " + molecule.GetTitle()

        # Write molecule in Tripos format.
        tripos_mol2_filename = 'molecule.tripos.mol2'
        omolstream = oechem.oemolostream(tripos_mol2_filename)
        oechem.OEWriteMolecule(omolstream, molecule)
        omolstream.close()

        # Parameterize for AMBER.
        molecule_name = 'molecule'
        [gaff_mol2_filename, frcmod_filename] = gaff2xml.utils.run_antechamber(molecule_name, tripos_mol2_filename, charge_method="bcc", net_charge=0)
        #print_file(gaff_mol2_filename)
        [prmtop_filename, inpcrd_filename] = gaff2xml.utils.run_tleap(molecule_name, gaff_mol2_filename, frcmod_filename)
        #print_file(prmtop_filename)

        # Create OpenMM System object for molecule in vacuum.
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        inpcrd = app.AmberInpcrdFile(inpcrd_filename)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds, implicitSolvent=None, removeCMMotion=False)
        positions = inpcrd.getPositions()

        # Store system and positions.
        entry['system'] = system
        entry['positions'] = positions

        # Unlink files.
        for filename in os.listdir(working_directory):
            os.unlink(filename)

    os.chdir(original_directory)
    # TODO: Remove temporary directory and contents.

    # Type all molecules with GBSA parameters.
    start_time = time.time()
    typed_molecules = list()
    untyped_molecules = list()
    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']

        if verbose: print "  " + molecule.GetTitle()

        # Assign GBSA types according to SMARTS rules.
        try:
            atom_typer.assignTypes(molecule)
            typed_molecules.append(oechem.OEGraphMol(molecule))
            #atom_typer.debugTypes(molecule)
            #if( len(typed_molecules) > 10 ):
            #    sys.exit(-1)
        except AtomTyper.TypingException as exception:
            name = molecule.GetTitle()
            print name
            print exception
            untyped_molecules.append(oechem.OEGraphMol(molecule))
            if( len(untyped_molecules) > 10 ):
               sys.exit(-1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%d molecules correctly typed" % (len(typed_molecules))
    print "%d molecules missing some types" % (len(untyped_molecules))
    print "%.3f s elapsed" % elapsed_time

    # Compute energies with all molecules.
    print "Computing all energies..."
    start_time = time.time()
    energies = compute_hydration_energies(database, parameters)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time

    # Print comparison.
    signed_errors = numpy.zeros([len(database.keys())], numpy.float64)
    for (i, cid) in enumerate(database.keys()):
        # Get metadata.
        entry = database[cid]
        molecule = entry['molecule']
        name = molecule.GetTitle()
        dg_exp           = float(molecule.GetData('expt')) * units.kilocalories_per_mole
        ddg_exp         = float(molecule.GetData('d_expt')) * units.kilocalories_per_mole
        signed_errors[i] = energies[molecule] / units.kilocalories_per_mole - dg_exp / units.kilocalories_per_mole

        # Form output.
        outstring = "%48s %8.3f %8.3f %8.3f" % (name, dg_exp / units.kilocalories_per_mole, ddg_exp / units.kilocalories_per_mole, energies[molecule] / units.kilocalories_per_mole)

        print outstring

    print "Initial RMS error %8.3f kcal/mol" % (signed_errors.std())

    # Create MCMC model.
    model = create_model(database, parameters)

    # Sample models.
    from pymc import MCMC
    sampler = MCMC(model, db='txt', name=mcmcDbName)
    sampler.isample(iter=mcmcIterations, burn=0, save_interval=1, verbose=True)
    #sampler.sample(iter=mcmcIterations, burn=0, save_interval=1, verbose=True, progress_bar=True)
    sampler.db.close()
