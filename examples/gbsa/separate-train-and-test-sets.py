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
        function = utils.hydration_energy_factory(entry)
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
        model['tau_%s' % cid] = pymc.Lambda('tau_%s' % cid, lambda sigma=model['sigma'] : 1.0 / (sigma**2 + ddg_exp**2) ) # Include model error
        #model['tau_%s' % cid] = pymc.Lambda('tau_%s' % cid, lambda sigma=model['sigma'] : 1.0 / (ddg_exp**2) ) # Do not include model error.
        model[variable_name]   = pymc.Normal(variable_name, mu=model['dg_gbsa_%s' % cid], tau=model['tau_%s' % cid], value=dg_exp, observed=True)

    # Define convenience functions.
    parents = {'dg_gbsa_%s'%cid : model['dg_gbsa_%s' % cid] for cid in cid_list }
    def RMSE(**args):
        nmolecules = len(cid_list)
        error = numpy.zeros([nmolecules], numpy.float64)
        for (molecule_index, cid) in enumerate(cid_list):
            entry = database[cid]
            molecule = entry['molecule']
            error[molecule_index] = args['dg_gbsa_%s' % cid] - float(molecule.GetData('expt'))
        mse = numpy.mean((error - numpy.mean(error))**2)
        return numpy.sqrt(mse)

    model['RMSE'] = pymc.Deterministic(eval=RMSE,
                                       name='RMSE',
                                       parents=parents,
                                       doc='RMSE',
                                       trace=True,
                                       verbose=1,
                                       dtype=float,
                                       plot=True,
                                       cache_depth=2)

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
    usage: %prog --database database --ntest ntest --train train.pickle --test test.pickle

    example: %prog --database datasets/FreeSolv/v0.3/database.pickle --ntest 100 --train train.pkl --test test.pkl

    """
    version_string = "%prog %__version__"
    parser = OptionParser(usage=usage_string, version=version_string)

    parser.add_option("-d", "--database", metavar='DATABASE',
                      action="store", type="string", dest='database_filename', default='',
                      help="Python pickle file of database with molecule names, SMILES strings, hydration free energies, and experimental uncertainties (FreeSolv format).")

    parser.add_option("-t", "--ntest", metavar='NTEST',
                      action="store", type="int", dest='ntest', default=None,
                      help="Size of subset to consider (for testing).")

    parser.add_option("-r", "--train", metavar='TRAIN',
                      action="store", type="string", dest='train_database_filename', default='',
                      help="Python pickle file of training database.")

    parser.add_option("-e", "--test", metavar='TEST',
                      action="store", type="string", dest='test_database_filename', default='',
                      help="Python pickle file of testing database.")

    # Parse command-line arguments.
    (options,args) = parser.parse_args()

    # Ensure all required options have been specified.
    if options.ntest==None or options.train_database_filename=='' or options.test_database_filename=='':
        parser.print_help()
        parser.error("All input files must be specified.")

    # Open database.
    import pickle
    database = pickle.load(open(options.database_filename, 'r'))

    # Select training and testing subset.
    keys = database.keys()
    import numpy.random
    permute = numpy.random.permutation(keys)
    test_database = dict((k, database[k]) for k in permute[0:options.ntest])
    train_database = dict((k, database[k]) for k in permute[options.ntest:])

    # Write databases.
    train_database_file = open(options.train_database_filename, 'w')
    pickle.dump(train_database, train_database_file)
    train_database_file.close()
    print "%d molecules in training set." % len(train_database.keys())

    test_database_file = open(options.test_database_filename, 'w')
    pickle.dump(test_database, test_database_file)
    test_database_file.close()
    print "%d molecules in test set." % len(test_database.keys())
