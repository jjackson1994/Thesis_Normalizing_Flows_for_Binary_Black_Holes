# This script supports training with comand line argument parsing
# This imports the defualts argument values from /tools/constants.py 
# so it is always consitent with the function version

from tools.constants import defaults as D
from tools.model import train
from argparse import ArgumentParser

parser = ArgumentParser(description='NF training')
#See tools/constants.py for defaults
parser.add_argument("-sample_cols", help = "sample_cols", default =D['sample_cols'])
parser.add_argument("-pop_cols", help = "pop_cols", default = D['pop_cols'])
parser.add_argument("-N", type = int, dest="epochs", help = "number of epochs", default=D['epochs'])
parser.add_argument("-A", type = str, help = "Activation Function", default=D['A'])
parser.add_argument("-blocks", type = int, help = "num_blocks", default = D['blocks'])
parser.add_argument("-hidden", type = int, help = "num_hidden", default = D['hidden'])
parser.add_argument("-label", type = str, help = "Model label", default =  D['label'])
parser.add_argument("-early_stop", help = "early stopping bool (True / False) ", default = D['early_stop'])
parser.add_argument("-time", help = "Set max runtime in min (useful for tuning)", default = D['time'])

# Hardware Config
parser.add_argument("-device", type = str, help = "Set device", default = D['device'])
parser.add_argument("-DL",dest ='dataloader', help = "Bool for using dataloader", default = D['dataloader'])
parser.add_argument("-batch_size", type = int, help = "Set batch size", default = D['batch_size'])
parser.add_argument("-workers", type = int, help = "Set num workers", default = D['workers'])
parser.add_argument("-pin_mem", type = int, dest='PIN_MEM', help = "Memory Pinning", default = D['PIN_MEM'])

#Filepaths
parser.add_argument("-T", type = str, dest="training_file", help = "File path for training set", default =D['training_file'])
parser.add_argument("-V", type = str, dest="validation_file",help = "File path validation set", default =D['validation_file'])
parser.add_argument("-O", dest = 'outdir', help = "Folder where model is saved", default = D['outdir'])

args = parser.parse_args()
info = vars(args)

if isinstance(info['sample_cols'], str):info['sample_cols'] = info['sample_cols'].split(':')
if isinstance(info['pop_cols'],str): info['pop_cols'] = info['pop_cols'].split(':')

def bool_fix(bool_v):
    bool_out = bool_v
    if  isinstance(bool_v, str):
        bool_conv = {'True' : True, 'true' : True, 'False': False, 'false': False}
        bool_out = bool_conv[bool_v]
    return bool_out

for bl_key in ['early_stop', 'dataloader', 'PIN_MEM']:
    info[bl_key]=bool_fix(info[bl_key])

# Train the model:
train(info)