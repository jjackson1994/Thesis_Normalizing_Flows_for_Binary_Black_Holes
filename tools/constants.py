from os.path import join
from platform import platform
root_dir = '/Users/jakejackson/Normalising_Flows_Binary_Black_Holes'
data_dir = join(root_dir, 'data')
full_data_path = join(data_dir, 'BHBHm.pq')

# SEVN Input Data Hyperparameters
Zs =[0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.014, 0.017, 0.02  , 0.03]
alphas =  [0.5, 1. , 3. , 5. ]

# Default values: set here so the training runs the same with or without argparse
defaults = {# Training Parameters -----------------------------------------------------------------
            'sample_cols' : ['Mass_0', 'Mass_1'], # Sample columns 
            'pop_cols' : ['Z', 'alpha'],          # Population parameters
            'epochs' : 100,                       # Number of epochs
            'A': 'tanh',                          # Activation Function
            'blocks' : 10,                        # Number of Blocks
            'hidden': 128,                        # Number of Hidden layers
            'label' : 'default',                  # Run Label (for saving)
            'early_stop' :True,                   # Stop if learning plateaus
            'time' :False,                        # Max runtime
            
            # Hardware Config---------------------------------------------------------------------
            'device' :'cuda',                     # Torch Device :'cpu', 'cuda', 'mps' ...
            'dataloader' : True,                  # Use Dataloader True/False
            'batch_size' : 150000,                # Dataloader batch size
            'workers' : 4,                        # Dataloader Workers
            'PIN_MEM' : True,                     # Pin memory

            #File Paths---------------------------------------------------------------------------
            'training_file' : join(data_dir,'BHBHm_Tr_0.6_Va_0.2_Te_0.2_train.pq'),
            'validation_file' : join(data_dir,'BHBHm_Tr_0.6_Va_0.2_Te_0.2_val.pq'),
            'outdir' : root_dir}

#DEVICE SPECIFIC SETTINGS
OS = platform()
if 'mac' in OS: 
    defaults['workers'] = 0 
    defaults['device'] = 'mps'