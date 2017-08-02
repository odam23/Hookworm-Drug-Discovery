from modeller import *

env = environ()
aln = alignment(env)
mdl = model(env, file='5c8y', model_segment=('FIRST:D','LAST:D'))
aln.append_model(mdl, align_codes='5c8yD', atom_files='5c8y.pdb')
aln.append(file='W2T758.ali', align_codes='W2T758')
aln.align2d()
aln.write(file='W2T758-5c8yD.ali', alignment_format='PIR')
aln.write(file='W2T758-5c8yD.pap', alignment_format='PAP')
