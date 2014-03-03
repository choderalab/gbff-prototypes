#!/usr/bin/env python

import sys,string
from openeye.oechem import *

class Patty:
    def __init__(self,infileName):
        self.pattyTag = OEGetTag("patty") 
        self.smartsList = []
        ifs = open(infileName)
        lines = ifs.readlines()
        for line in lines:
            # Strip trailing comments
            index = line.find('%')
            if index != -1:
                line = line[0:index]
            print line            
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
        OEAssignAromaticFlags(mol)
        for pat,type,smarts in self.smartsList:
            for matchbase in pat.Match(mol):
                for matchpair in matchbase.GetAtoms():
                    matchpair.target.SetStringData(self.pattyTag,type)

    def debugTypes(self,mol):
        for atom in mol.GetAtoms():
            print "%6d %8s %8s" % (atom.GetIdx(),OEGetAtomicSymbol(atom.GetAtomicNum()),atom.GetStringData(self.pattyTag))

    def getTypeList(self,mol):
        typeList = []
        for atom in mol.GetAtoms():
            typeList.append(atom.GetStringData(self.pattyTag))
        return typeList
                

if __name__=="__main__":
    patty = Patty(sys.argv[1])
    ifs = oemolistream(sys.argv[2])
    mol = OECreateOEGraphMol()
    while OEReadMolecule(ifs,mol):
        name = OEGetSDData(mol, 'name').strip()
        print name
        
        patty.assignTypes(mol)
        patty.debugTypes(mol)
    ifs.close()
    
    
    
        
    
