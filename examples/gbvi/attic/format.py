"""
Hack to format the sdf's from paul labute

WE read in the sdf
we grab the tag data. 
we add it to the OEMol
we check to see if we have coordinates
We apply AM1BCC charging. 
we write out the resultand file and and the Name:Energy

"""
from openeye.oechem import *
from openeye.oequacpac import *
from openeye.oeszybki import *
from openeye.oeomega import *

import os, sys

#Global vars
debug = True
badMolCounter = 0

#Build a Szybki Object in case we need one to optimise
#sz = OESzybki()
#sz.SetOptimizerType(OERunType_CartesiansOpt)
#sz.SetRemoveAttractiveVdWForces(True)
#sz.SetOptCriteria(100, 0.0)

#Omega Object
omega = OEOmega()
omega.SetMaxConfGen(1)
omega.SetMaxConfs(1)
omega.SetFromCT(True)

ifs = oemolistream('labute.sdf')
ofs = oemolostream('labute_3D_charged.oeb')
for mol in ifs.GetOEGraphMols():
    title = mol.GetTitle().strip()
    
    #Get the SD Data
    print title
    name = OEGetSDData(mol, 'name')
    name = name.strip()
    dg = OEGetSDData(mol, 'dG(exp)')
    dg = float(dg)

    #Check the number of hydrogens and the total atom count as a sanity check
    initialNumHydrogens = OECount(mol, OEIsHydrogen())
    initialNumAtoms = mol.NumAtoms()
    #mol dimension check
    #if mol.GetDimension() < 3:
        #print '#Warning %s has dimension < 3' % name
        #badMolCounter +=1
        #write the Fucker out
        #ofs2 = oemolostream('BadMol-%s.sdf' % badMolCounter)
        #OEWriteMolecule(ofs2, mol)
        #ofs2.close()
        #continue
            
    #we need to make a single 3D mol, and refine it.  
    tmp = OEMol(mol)
    omega.SetFixMol(mol)
    omega(tmp)
    #omega.ClearFixMol()
    #sz(tmp)
    mol = OEGraphMol(tmp)
       

    #Now we check to ensure we are not adding more protons than required or 
    #changing the total number of atoms. 
    processedNumHydrogens = OECount(mol, OEIsHydrogen())
    processedNumAtoms = mol.NumAtoms()
    
    #Delta
    print 'Initial H count %i Final H count %i DELTA %i' % (initialNumHydrogens, processedNumHydrogens, (initialNumHydrogens - processedNumHydrogens))
    print 'Initial Atom count %i Final Atom count %i DELTA %i' % (initialNumAtoms, processedNumAtoms, (initialNumAtoms - processedNumAtoms))
    
    
    #now add the properties to the Mol
    mol.SetTitle(name)
    mol.SetFloatData('dg', dg)
        
        
    if (OEAssignPartialCharges(mol,OECharges_AM1BCC,False,debug)) == False:
        print '#WARNING: failure to assign charges to %s' % (name)
    else:
        print 'charges assigned to %s' % (name)
    
        

    #Check the charges
    if OEHasPartialCharges(mol) == False:
        if mol.NumAtoms() == 1:
            #its an ion
            OEAssignFormalCharges(mol)
            print '#Formal charges assigned to %s ' % name
            OEWriteMolecule(ofs, mol)
            
        else:
            print '#WARNING: no charges on %s ' % name
            badMolCounter +=1
            #write the Fucker out
            ofs2 = oemolostream('BadMol-%i.sdf' % badMolCounter)
            OEWriteMolecule(ofs2, mol)
            ofs2.close()
        
    else:
        print 'CHARGES CORRECT on %s ' % name    
        OEWriteMolecule(ofs, mol)
    
ofs.close() 
print "%i molecules failed" % badMolCounter
    ##sd data
    
    
