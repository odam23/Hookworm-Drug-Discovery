#! /bin/bash
#babel minim.sdf ZINC.sdf -m
for m in *.sdf ; do
babel -isdf $m -opdb `basename $m sdf`pdb ; 
done
for n in *.pdb ; do
babel -ipdb $n -opdbqt `basename $n pdb`pdbqt -h ; 
done
#mkdir -p ./pdbntfiles
mkdir -p ./pdbfiles
mv *pdb ./pdbfiles
mv *pdbqt ./pdbqtfiles
