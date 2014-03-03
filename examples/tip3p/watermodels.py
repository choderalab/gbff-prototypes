#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
watermodels.py

Construct various water models.

AUTHORS

John Chodera <jchodera@berkeley.edu>, University of California, Berkeley

"""
#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import sys
import math
import time
import string

import numpy

import simtk.unit as units
import simtk.openmm as mm

#=============================================================================================
# TIP3P (flexible)
#=============================================================================================

class TIP3P(object):

    def __init__(self):
        pass
    
    def get_default_parameters(self):
        """
        Get default parameters for TIP3P model.
        
        RETURNS
        
        parameters (dict) - dictionary of parameters

        EXAMPLE
        
        >>> parameters = TIP3P.get_default_parameters()
        
        """
        
        parameters = dict()
        
        parameters['qO'] = -0.8340 * units.elementary_charge
        parameters['qH'] = +0.4170 * units.elementary_charge
        parameters['sigma'] = 3.15061 * units.angstrom
        parameters['epsilon'] = 0.6364  * units.kilojoule_per_mole 
        parameters['rOH'] = 0.9572  * units.angstrom
        parameters['kOH'] = 553.0 * units.kilocalories_per_mole / units.angstrom**2 # from AMBER parm96
        parameters['aHOH'] = 104.52  * units.degree
        parameters['kHOH'] = 100.0 * units.kilocalories_per_mole / units.radian**2 # from AMBER parm96
        
        return parameters

    def create_system(self, parameters=None, nmolecules=512, verbose=False):
        """
        Construct a flexible TIP3P system.
        
        RETURNS
        
        system (simtk.openmm.System) - TIP3P system with given parameters
        
        EXAMPLES

        Create with default parameters.
        
        >>> system = TIP3P.create_system()

        Create with specified parameters.
        
        >>> parameters = TIP3P.get_default_parameters()
        >>> system = TIP3P.create_system(parameters)

        """
        
        initial_time = time.time()
        
        # Fixed parameters.
        massO  = 16.0 * units.amu # oxygen mass
        massH  =  1.0 * units.amu # hydrogen mass
        cutoff = None # override for nonbonded cutoff
        nonbonded_method = mm.NonbondedForce.PME # nonbonded method
        unit_sigma = 1.0 * units.angstrom # hydrogen sigma
        zero_epsilon = 0.0 * units.kilocalories_per_mole # hydrogen epsilon
        
        # Set parameters if not provided.
        if parameters is None:
            parameters = TIP3P.get_default_parameters()
            
        # Create system.
        system = mm.System()

        # Masses.
        for molecule_index in range(nmolecules):
            system.addParticle(massO)
            system.addParticle(massH)
            system.addParticle(massH)

        # Nonbonded interactions.
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(nonbonded_method)
        if cutoff is not None: nb.setCutoffDistance(cutoff)    
        for molecule_index in range(nmolecules):
            # Nonbonded parameters.
            nb.addParticle(parameters['qO'], parameters['sigma'], parameters['epsilon'])
            nb.addParticle(parameters['qH'], unit_sigma, zero_epsilon)
            nb.addParticle(parameters['qH'], unit_sigma, zero_epsilon)

            # Nonbonded exceptions.       
            nb.addException(molecule_index*3, molecule_index*3+1, 0.0, unit_sigma, zero_epsilon)
            nb.addException(molecule_index*3, molecule_index*3+2, 0.0, unit_sigma, zero_epsilon)         
            nb.addException(molecule_index*3+1, molecule_index*3+2, 0.0, unit_sigma, zero_epsilon)
        system.addForce(nb)

        # Bonds.
        bonds = mm.HarmonicBondForce()
        for molecule_index in range(nmolecules):
            bonds.addBond(3*molecule_index+0, 3*molecule_index+1, parameters['rOH'], parameters['kOH'])
            bonds.addBond(3*molecule_index+0, 3*molecule_index+2, parameters['rOH'], parameters['kOH'])       
        system.addForce(bonds)

        # Angles.
        angles = mm.HarmonicAngleForce()
        for molecule_index in range(nmolecules):
            angles.addAngle(3*molecule_index+1, 3*molecule_index+0, 3*molecule_index+2, parameters['aHOH'], parameters['kHOH'])
        system.addForce(angles)

        final_time = time.time()
        elapsed_time = final_time - initial_time
        if verbose: print "%.3f s elapsed" % elapsed_time

        return system

    def modify_parameters(self, system, parameters):
        """
        Construct a flexible TIP3P system.
        
        ARGUMENTS
        
        parameters (dict) - dictionary of parameters
        
        RETURNS
        
        system (simtk.openmm.System) - TIP3P system with given parameters
        
        """

        initial_time = time.time()

        nmolecules = system.getNumParticles() / 3 # number of molecules

        # Nonbonded interactions.
        nb = system.getForce(0)
        for molecule_index in range(nmolecules):
            nb.setParticleParameters(3*molecule_index+0, parameters['qO'], parameters['sigma'], parameters['epsilon'])

        # Bonds.
        bonds = system.getForce(1)
        for molecule_index in range(nmolecules):
            bonds.setBondParameters(2*molecule_index+0, 3*molecule_index+0, 3*molecule_index+1, parameters['rOH'], parameters['kOH'])
            bonds.setBondParameters(2*molecule_index+1, 3*molecule_index+0, 3*molecule_index+2, parameters['rOH'], parameters['kOH'])       

        # Angles.
        angles = system.getForce(2)
        for molecule_index in range(nmolecules):
            angles.setAngleParameters(molecule_index, 3*molecule_index+1, 3*molecule_index+0, 3*molecule_index+2, parameters['aHOH'], parameters['kHOH'])

        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "%.3f s elapsed" % elapsed_time

        return 
 
#=============================================================================================
# TIP3P (flexible)
#=============================================================================================

if __name__=='__main__':
    # Run doctests.
    import doctests
    doctests.testmod()


    
    
   
   
