#!/bin/tcsh

rm -f leap.log waterbox.{prmtop,crd,pdb}
tleap -f setup-waterbox.leap.in
cat waterbox.crd | ambpdb -p waterbox.prmtop > waterbox.pdb

