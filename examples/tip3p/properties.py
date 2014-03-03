#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
properties.py

Module for computation of experimentally-observable properties for water.

PROPERTIES

* volume
* mass
* density

AVERAGE PROPERITES

* bulk density (II.E.3 of [1])
* enthalpy of vaporiztion (II.E.4 of [1])
* isobaric heat capacity (II.E.5 of [1])
* isothermal compressibility (II.E.6 of [1])
* thermal expansion coefficient (II.E.7 of [1])
* self-diffusion coefficient (II.E.8 of [1])
* static dielectric constant (II.E.9 of [1])
* scattering intensities (II.E.10 of [1])

REFERENCES

[1] Horn HW, Swope WC, Pitera JW, Madura JD, Dick TJ, Hura GL, Head-Gordon T. Development of
an improved four-site water model for biomolecular simulations: TIP4P-Ew. JCP 120:9665, 2004.
http://dx.doi.org/10.1063/1.1683075

[2] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple
equilibrium states. J. Chem. Phys. 129:124105 

AUTHORS

John Chodera <jchodera@berkeley.edu>, University of California, Berkeley

TODO

* Turn properties into classes where there is a per-snapshot labeling step and a reduction step.
* Create a Property base class?


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
import simtk.openmm as openmm

#=============================================================================================
# Density
#=============================================================================================

def volume(system, state):
    """
    Compute the volume of the current configuration.    
    
    RETURNS

    volume (simtk.unit.Quantity) - the volume of the system (in units of length^3), or None if no box coordinates are defined
    
    """

    # Compute volume of parallelepiped.
    [a,b,c] = state.getPeriodicBoxvectors(asNumpy=True)
    A = numpy.array([a/a.unit, b/a.unit, c/a.unit])
    volume = numpy.linalg.det(A) * a.unit**3
    return volume

def mass(system, state):
    """
    Compute the total system mass.

    RETURNS

    mass (simtk.unit.Quantity) - the total mass of the system

    """

    total_mass = 0.0 * system.getPrticleMass(0)
    nparticles = system.getNumParticles()
    for i in range(particles):
        total_mass += system.getParticleMass(i)
    return total_mass

def density(system, state):
    """
    Compute density of a single configuration.

    ARGUMENTS

    system (simtk.openmm.System) 
    state

    RETURNS

    density (simtk.unit.Quantity)

    """

    # Compute the total box volume.
    box_volume = volume(system, state)

    # Compute the total system mass.
    total_mass = mass(system, state)

    # Return density.
    density = total_mass / box_volume

    return density


    
