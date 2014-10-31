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

import utils

#=============================================================================================
# Model assessment.
#=============================================================================================

def print_file(filename):
    infile = open(filename, 'r')
    print infile.read()
    infile.close()

def evaluate_performance(phase, database, initial_parameters, parameter_sets):

    import os, os.path

    os.makedirs(phase)
    outfile = open(os.path.join(phase, 'evaluate.txt'), 'w')

    for (iteration_index, parameters) in enumerate([initial_parameters] + parameter_sets):

        iteration_outfile = open(os.path.join(phase, 'iteration-%05d.out' % iteration_index), 'w')

        # Compute energies with all molecules.
        print "Computing all energies..."
        start_time = time.time()
        energies = utils.compute_hydration_energies(database, parameters)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print "%.3f s elapsed" % elapsed_time

        # Print summary statistics.
        nmolecules = len(database)
        signed_errors = numpy.zeros([nmolecules], numpy.float64)
        for (i, cid) in enumerate(database.keys()):
            # Get metadata.
            entry = database[cid]
            molecule = entry['molecule']
            name = molecule.GetTitle()
            dg_exp           = float(molecule.GetData('expt')) * units.kilocalories_per_mole
            ddg_exp          = float(molecule.GetData('d_expt')) * units.kilocalories_per_mole
            signed_errors[i] = energies[molecule] / units.kilocalories_per_mole - dg_exp / units.kilocalories_per_mole

            # Form output.
            outstring = "%64s %8.3f %8.3f %8.3f" % (name, dg_exp / units.kilocalories_per_mole, ddg_exp / units.kilocalories_per_mole, energies[molecule] / units.kilocalories_per_mole)
            print outstring

            # Write comprarison of computed and experimental.
            iteration_outfile.write('%16.8f %16.8f %16.8f %16.8f\n' % (energies[molecule] / units.kilocalories_per_mole, 0.0, dg_exp / units.kilocalories_per_mole, ddg_exp / units.kilocalories_per_mole))

        # Close iteration file.
        iteration_outfile.close()

        print "============================================================="
        print "iteration %8d : RMS error %8.3f kcal/mol" % (iteration_index, signed_errors.std())
        print "============================================================="

        outfile.write('%8d %12.6f ' % (iteration_index, signed_errors.std()))
        for i in range(signed_errors.size):
            outfile.write('%12.6f ' % signed_errors[i])
        outfile.write('\n')
        outfile.flush()
    outfile.close()

    return

#=============================================================================================
# MAIN
#=============================================================================================

if __name__=="__main__":

    # Create command-line argument options.
    usage_string = """\
    usage: %prog --types typefile --parameters paramfile --iterations MCMC_iterations --mcmcout MCMC_db_name [--train train.pkl] [--test test.pkl]

    example: %prog --types parameters/gbsa-amber-mbondi2.types --parameters parameters/gbsa-amber-mbondi2.parameters --mcmcout MCMC --verbose --mol2 datasets/FreeSolv/FreeSolv/tripos_mol2 --train train.pkl --test test.pkl

    """
    version_string = "%prog %__version__"
    parser = OptionParser(usage=usage_string, version=version_string)

    parser.add_option("-t", "--types", metavar='TYPES',
                      action="store", type="string", dest='atomtypes_filename', default='',
                      help="Filename defining atomtypes as SMARTS atom matches.")

    parser.add_option("-p", "--parameters", metavar='PARAMETERS',
                      action="store", type="string", dest='parameters_filename', default='',
                      help="File containing initial parameter set.")

    parser.add_option("-m", "--mol2", metavar='MOL2',
                      action="store", type="string", dest='mol2_directory', default='',
                      help="Directory containing charged mol2 files (optional).")

    parser.add_option("-i", "--iteration", metavar='ITERATION',
                      action="store", type="int", dest='iteration', default=None,
                      help="MCMC iteration to read.")

    parser.add_option("-o", "--mcmcout", metavar='MCMCOUT',
                      action="store", type="string", dest='mcmcout', default='MCMC',
                      help="MCMC output database name.")

    parser.add_option("-r", "--train", metavar='TRAIN',
                      action="store", type="string", dest='train_database_filename', default=None,
                      help="Size of subset to consider for training.")

    parser.add_option("-s", "--test", metavar='TEST',
                      action="store", type="string", dest='test_database_filename', default=None,
                      help="Size of subset to consider for testing.")

    parser.add_option("-v", "--verbose", metavar='VERBOSE',
                      action="store_true", dest='verbose', default=False,
                      help="Verbosity flag.")

    # Parse command-line arguments.
    (options,args) = parser.parse_args()

    # Ensure all required options have been specified.
    if options.atomtypes_filename=='' or options.parameters_filename=='' or options.train_database_filename=='' or options.test_database_filename=='':
        parser.print_help()
        parser.error("All input files must be specified.")

    # Read GBSA parameters from desired iteration.
    parameters = utils.read_gbsa_parameters(options.parameters_filename)

    mcmcDbName     = os.path.abspath(options.mcmcout)

    # Open database.
    import pickle

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


    # Divide database into training set and test set.
    if options.train_database_filename:
        train_database = pickle.load(open(options.train_database_filename, 'r'))
        utils.prepare_database(train_database, options.atomtypes_filename, parameters, mol2_directory=options.mol2_directory, verbose=options.verbose)
        evaluate_performance('train', train_database, parameters, parameter_sets)

    if options.test_database_filename:
        test_database = pickle.load(open(options.test_database_filename, 'r'))
        utils.prepare_database(test_database, options.atomtypes_filename, parameters, mol2_directory=options.mol2_directory, verbose=options.verbose)
        evaluate_performance('test', test_database, parameters, parameter_sets)


