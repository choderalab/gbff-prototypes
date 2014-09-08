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

import pymbar

#=============================================================================================
# PARAMETERS
#=============================================================================================

temperature = 300.0 * units.kelvin # TODO: This should be part of the data specification

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
# Generate simulation data.
#=============================================================================================

def generate_simulation_data(database, parameters):
    """
    Regenerate simulation data for given parameters.

    ARGUMENTS

    database (dict) - database of molecules
    parameters (dict) - dictionary of GBSA parameters keyed on GBSA atom types

    """

    platform = openmm.Platform.getPlatformByName("Reference")

    from pymbar import timeseries

    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']
        iupac_name = entry['iupac']

        # Retrieve OpenMM System.
        solvent_system = copy.deepcopy(entry['system'])

        # Get nonbonded force.
        forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
        nonbonded_force = forces['NonbondedForce']

        # Add GBSA term
        gbsa_force = openmm.GBSAOBCForce()
        gbsa_force.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff) # set no cutoff
        gbsa_force.setSoluteDielectric(1)
        gbsa_force.setSolventDielectric(78)

        # Build indexable list of atoms.
        atoms = [atom for atom in molecule.GetAtoms()]
        natoms = len(atoms)

        # Assign GBSA parameters.
        for (atom_index, atom) in enumerate(atoms):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
            atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
            radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
            scalingFactor = parameters['%s_%s' % (atomtype, 'scalingFactor')] * units.kilocalories_per_mole
            gbsa_force.addParticle(charge, radius, scalingFactor)

        # Add the force to the system.
        solvent_system.addForce(gbsa_force)

        # Create context for solvent system.
        timestep = 2.0 * units.femtosecond
        collision_rate = 5.0 / units.picoseconds
        solvent_integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        solvent_context = openmm.Context(solvent_system, solvent_integrator, platform)

        # Set the coordinates.
        positions = entry['positions']
        solvent_context.setPositions(positions)

        # Minimize.
        openmm.LocalEnergyMinimizer.minimize(solvent_context)

        # Simulate, saving periodic snapshots of configurations.
        kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant
        kT = kB * temperature
        beta = 1.0 / kT

        initial_time = time.time()
        nsteps_per_iteration = 500
        niterations = 100
        x_n = numpy.zeros([niterations,natoms,3], numpy.float32) # positions, in nm
        u_n = numpy.zeros([niterations], numpy.float64) # energy differences, in kT
        for iteration in range(niterations):
            solvent_integrator.step(nsteps_per_iteration)
            solvent_state = solvent_context.getState(getEnergy=True, getPositions=True)
            x_n[iteration,:,:] = solvent_state.getPositions(asNumpy=True) / units.nanometers
            u_n[iteration] = beta * solvent_state.getPotentialEnergy()

        if numpy.any(numpy.isnan(u_n)):
            raise Exception("Encountered NaN for molecule %s | %s" % (cid, iupac_name))

        final_time = time.time()
        elapsed_time = final_time - initial_time

        # Clean up.
        del solvent_context, solvent_integrator

        # Discard initial transient to equilibration.
        [t0, g, Neff_max] = timeseries.detectEquilibration(u_n)
        x_n = x_n[t0:,:,:]
        u_n = u_n[t0:]

        # Subsample to remove correlation.
        indices = timeseries.subsampleCorrelatedData(u_n, g=g)
        x_n = x_n[indices,:,:]
        u_n = u_n[indices]

        # Store data.
        entry['x_n'] = x_n
        entry['u_n'] = u_n

        print "%48s | %48s | simulation %12.3f s | %5d samples discarded | %5d independent samples remain" % (cid, iupac_name, elapsed_time, t0, len(indices))

    return

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

    from pymbar import MBAR

    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']
        iupac_name = entry['iupac']

        # Retrieve OpenMM System.
        vacuum_system = entry['system']
        solvent_system = copy.deepcopy(entry['system'])

        # Get nonbonded force.
        forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
        nonbonded_force = forces['NonbondedForce']

        # Add GBSA term
        gbsa_force = openmm.GBSAOBCForce()
        gbsa_force.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff) # set no cutoff
        gbsa_force.setSoluteDielectric(1)
        gbsa_force.setSolventDielectric(78)

        # Build indexable list of atoms.
        atoms = [atom for atom in molecule.GetAtoms()]
        natoms = len(atoms)

        # Assign GBSA parameters.
        for (atom_index, atom) in enumerate(atoms):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
            atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
            radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
            scalingFactor = parameters['%s_%s' % (atomtype, 'scalingFactor')] * units.kilocalories_per_mole
            gbsa_force.addParticle(charge, radius, scalingFactor)

        # Add the force to the system.
        solvent_system.addForce(gbsa_force)

        # Create context for solvent system.
        timestep = 2.0 * units.femtosecond
        solvent_integrator = openmm.VerletIntegrator(timestep)
        solvent_context = openmm.Context(solvent_system, solvent_integrator, platform)

        # Create context for vacuum system.
        vacuum_integrator = openmm.VerletIntegrator(timestep)
        vacuum_context = openmm.Context(vacuum_system, vacuum_integrator, platform)

        # Compute energy differences.
        kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant
        kT = kB * temperature
        beta = 1.0 / kT

        initial_time = time.time()
        nsteps_per_iteration = 500
        niterations = 100
        x_n = entry['x_n']
        u_n = entry['u_n']
        nsamples = len(u_n)
        nstates = 3 # number of thermodynamic states
        u_kln = numpy.zeros([3,3,nsamples], numpy.float64)
        for sample in range(nsamples):
            positions = units.Quantity(x_n[sample,:,:], units.nanometers)

            u_kln[0,0,sample] = u_n[sample]

            vacuum_context.setPositions(positions)
            vacuum_state = vacuum_context.getState(getEnergy=True)
            u_kln[0,1,sample] = beta * vacuum_state.getPotentialEnergy()

            solvent_context.setPositions(positions)
            solvent_state = solvent_context.getState(getEnergy=True)
            u_kln[0,2,sample] = beta * solvent_state.getPotentialEnergy()

        N_k = numpy.zeros([nstates], numpy.int32)
        N_k[0] = nsamples

        mbar = MBAR(u_kln, N_k)
        [df_ij, ddf_ij] = mbar.getFreeEnergyDifferences()

        DeltaG_in_kT = df_ij[1,2]
        dDeltaG_in_kT = ddf_ij[1,2]

        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "%48s | %48s | reweighting took %.3f s" % (cid, iupac_name, elapsed_time)

        # Clean up.
        del solvent_context, solvent_integrator
        del vacuum_context, vacuum_integrator

        energies[molecule] = kT * DeltaG_in_kT

        print "%48s | %48s | DeltaG = %.3f +- %.3f kT" % (cid, iupac_name, DeltaG_in_kT, dDeltaG_in_kT)
        print ""

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

    from pymbar import MBAR

    molecule = entry['molecule']
    iupac_name = entry['iupac']
    cid = molecule.GetData('cid')

    # Retrieve OpenMM System.
    vacuum_system = entry['system']
    solvent_system = copy.deepcopy(entry['system'])

    # Get nonbonded force.
    forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
    nonbonded_force = forces['NonbondedForce']

    # Add GBSA term
    gbsa_force = openmm.GBSAOBCForce()
    gbsa_force.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff) # set no cutoff
    gbsa_force.setSoluteDielectric(1)
    gbsa_force.setSolventDielectric(78)

    # Build indexable list of atoms.
    atoms = [atom for atom in molecule.GetAtoms()]
    natoms = len(atoms)

    # Assign GBSA parameters.
    for (atom_index, atom) in enumerate(atoms):
        [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
        atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
        radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
        scalingFactor = parameters['%s_%s' % (atomtype, 'scalingFactor')] * units.kilocalories_per_mole
        gbsa_force.addParticle(charge, radius, scalingFactor)

    # Add the force to the system.
    solvent_system.addForce(gbsa_force)

    # Create context for solvent system.
    timestep = 2.0 * units.femtosecond
    solvent_integrator = openmm.VerletIntegrator(timestep)
    solvent_context = openmm.Context(solvent_system, solvent_integrator, platform)

    # Create context for vacuum system.
    vacuum_integrator = openmm.VerletIntegrator(timestep)
    vacuum_context = openmm.Context(vacuum_system, vacuum_integrator, platform)

    # Compute energy differences.
    kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant
    kT = kB * temperature
    beta = 1.0 / kT

    initial_time = time.time()
    nsteps_per_iteration = 500
    niterations = 100
    x_n = entry['x_n']
    u_n = entry['u_n']
    nsamples = len(u_n)
    nstates = 3 # number of thermodynamic states
    u_kln = numpy.zeros([3,3,nsamples], numpy.float64)
    for sample in range(nsamples):
        positions = units.Quantity(x_n[sample,:,:], units.nanometers)

        u_kln[0,0,sample] = u_n[sample]

        vacuum_context.setPositions(positions)
        vacuum_state = vacuum_context.getState(getEnergy=True)
        u_kln[0,1,sample] = beta * vacuum_state.getPotentialEnergy()

        solvent_context.setPositions(positions)
        solvent_state = solvent_context.getState(getEnergy=True)
        u_kln[0,2,sample] = beta * solvent_state.getPotentialEnergy()

    N_k = numpy.zeros([nstates], numpy.int32)
    N_k[0] = nsamples

    mbar = MBAR(u_kln, N_k)
    [df_ij, ddf_ij] = mbar.getFreeEnergyDifferences()

    DeltaG_in_kT = df_ij[1,2]
    dDeltaG_in_kT = ddf_ij[1,2]

    final_time = time.time()
    elapsed_time = final_time - initial_time
    print "%48s | %48s | reweighting took %.3f s" % (cid, iupac_name, elapsed_time)

    # Clean up.
    del solvent_context, solvent_integrator
    del vacuum_context, vacuum_integrator

    energy = kT * DeltaG_in_kT

    print "%48s | %48s | DeltaG = %.3f +- %.3f kT" % (cid, iupac_name, DeltaG_in_kT, dDeltaG_in_kT)
    print ""

    return energy / units.kilocalories_per_mole

def hydration_energy_factory(entry):
    def hydration_energy(**parameters):
        return compute_hydration_energy(entry, parameters, platform_name="Reference")
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

    example: %prog --types parameters/gbsa.types --parameters parameters/gbsa-am1bcc.parameters --database datasets/FreeSolv/v0.3/database.pickle --iterations 150 --mcmcout MCMC --verbose [--subset 10] [--mol2 datasets/FreeSolv/v0.3/mol2files_sybyl]

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

    parser.add_option("-m", "--mol2", metavar='MOL2',
                      action="store", type="string", dest='mol2_directory', default='',
                      help="Directory containing charged mol2 files (optional).")

    parser.add_option("-i", "--iterations", metavar='ITERATIONS',
                      action="store", type="int", dest='iterations', default=150,
                      help="MCMC iterations.")

    parser.add_option("-o", "--mcmcout", metavar='MCMCOUT',
                      action="store", type="string", dest='mcmcout', default='MCMC',
                      help="MCMC output database name.")

    parser.add_option("-s", "--subset", metavar='SUBSET',
                      action="store", type="int", dest='subset_size', default=None,
                      help="Size of subset to consider (for testing).")

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
    if options.subset_size:
        subset_size = options.subset_size
        cid_list = database.keys()
        database = dict((k, database[k]) for k in cid_list[0:subset_size])

    # Create omega instance in case we have to generate geometry.
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetFromCT(True)

    # Process all molecules in the dataset.
    start_time = time.time()
    for cid in database.keys():
        # Get database entry.
        entry = database[cid]

        # Extract relevant entry data from database.
        smiles = entry['smiles']
        iupac_name = entry['iupac']
        experimental_DeltaG = entry['expt'] * units.kilocalories_per_mole
        experimental_dDeltaG = entry['d_expt'] * units.kilocalories_per_mole

        # Read molecule.
        molecule = openeye.oechem.OEGraphMol()
        if options.mol2_directory:
            # Load the mol2 file.
            tripos_mol2_filename = os.path.join(options.mol2_directory, cid + '.mol2')
            omolstream = oechem.oemolistream(tripos_mol2_filename)
            oechem.OEReadMolecule(omolstream, molecule)
            omolstream.close()
        else:
            # Create OpenEye molecule from SMILES representation.
            openeye.oechem.OEParseSmiles(molecule, smiles)
            omega(molecule)

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

    # Create OpenMM System objects.
    import gaff2xml
    print "Running Antechamber..."
    original_directory = os.getcwd()
    working_directory = tempfile.mkdtemp()
    os.chdir(working_directory)
    start_time = time.time()
    problematic_cids = list() # list of cid entries that must be removed
    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']

        if verbose: print "  " + molecule.GetTitle()

        # Write molecule in Tripos format.
        tripos_mol2_filename = 'molecule.tripos.mol2'
        omolstream = oechem.oemolostream(tripos_mol2_filename)
        oechem.OEWriteMolecule(omolstream, molecule)
        omolstream.close()

        try:
            # Parameterize for AMBER.
            molecule_name = 'molecule'
            [gaff_mol2_filename, frcmod_filename] = gaff2xml.utils.run_antechamber(molecule_name, tripos_mol2_filename, charge_method=None)
            [prmtop_filename, inpcrd_filename] = gaff2xml.utils.run_tleap(molecule_name, gaff_mol2_filename, frcmod_filename)

            # Create OpenMM System object for molecule in vacuum.
            prmtop = app.AmberPrmtopFile(prmtop_filename)
            inpcrd = app.AmberInpcrdFile(inpcrd_filename)
            system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds, implicitSolvent=None, removeCMMotion=False)
            positions = inpcrd.getPositions()

            # TODO: Ensure that atom charges from molecule match those from System generated via prmtop file (as a check of atom ordering).

            # Store system and positions.
            entry['system'] = system
            entry['positions'] = positions
        except Exception as e:
            print e
            problematic_cids.append(cid)

        # Unlink files.
        for filename in os.listdir(working_directory):
            os.unlink(filename)

    os.chdir(original_directory)
    # TODO: Remove temporary directory and contents.

    # Remove problematic molecules
    print "Problematic molecules: %s" % str(problematic_cids)
    outfile = open('removed-molecules.txt', 'w')
    for cid in problematic_cids:
        iupac = database[cid]['iupac']
        outfile.write('%s %s\n' % (cid, iupac))
        del database[cid]
    outfile.close()

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

    # Generate simulation data.
    print "Generating simulation data for all molecules..."
    start_time = time.time()
    generate_simulation_data(database, parameters)
    end_time = time.time()
    elapsed_time = end_time - start_time
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
        ddg_exp          = float(molecule.GetData('d_expt')) * units.kilocalories_per_mole
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

