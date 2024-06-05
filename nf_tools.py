import json
import os
def get_all_files(path):
    files = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files

def get_run(path):
    import json
    json_f = open(path)
    info = json.load(json_f)
    return info 

def get_all_json(path): 
    all_files= get_all_files(path)
    all_json = [f for f in all_files if f.endswith('.json')]
    return all_json
    
def get_matches(path, attributes, criteria = '=='):
    all_files= get_all_files(path)
    all_json = [f for f in all_files if f.endswith('.json')]
    matches = []
    for js in all_json:
        att_match = []
        info = get_run(js)
        for key in attributes:
            if key in info: 
                if criteria == '==':
                    if  info[key]==attributes[key]:  
                        att_match.append(js)
                if criteria == '>=':
                    if  info[key]>=attributes[key]:
                        att_match.append(js)
                if criteria == '>':
                    if  info[key]>attributes[key]: 
                        att_match.append(js) 
                if criteria == '<=':
                    if  info[key]<=attributes[key]:  
                        att_match.append(js)
                if criteria == '<':
                    if info[key]<attributes[key]:
                        att_match.append(js)
         
        if len(att_match)== len(attributes):
            matches.append(js)
    return matches

def get_best_run(path, log = False):
    all_files= get_all_files(path)
    all_json = [f for f in all_files if f.endswith('.json') and 'pq_trial' not in f]
    best_run_loss = 9999
    final_loss_key = 'best_loss'
    for f in all_json: 
        json_f = open(f)
        info = json.load(json_f)
        if log == info['log']:
            if final_loss_key not in json_f:
                final_loss_key = 'final_loss'
            best_run_loss = info[final_loss_key]
            best_run_info = info
            print('Best run loss',  best_run_loss)
    return best_run_info

def train_from_prev(path, params):
    #Systematically check its a correct match
    #we want to avoid the system matching on epochs
    
    default_vals = get_run('trained_model/default.json')
    #need to add final epoch
    N = default_vals['epochs']
    del default_vals['epochs']
    
    if 'epochs' in params:
        N = params['epochs']
        del params['epochs']
        
    match_critera =  ["training_file", "validation_file", "A", "blocks", "hidden", "log", "early_stop", "MAFconfig"]
    default_trim = {key : default_vals[key] for key in default_vals if key in match_critera}
    print(default_trim)
    param_test = default_trim.update(params)
    
    print(param_test)

    matches = get_matches(path, param_test)
    print(matches)
    prev_epoch_dist=9999
    for m in matches:
        match_dict=get_run(m)
        match_epoch = match_dict['epochs']
        if stop_epoch in match_dict:
            match_epoch = match_dict['stop_epoch']
        
        if match_epoch < N:
            epoch_dist = N - match_epoch
            if epoch_dist < prev_epoch_dist:
                closest_match = match_dict
                prev_epoch_dist = epoch_dist 
        
    if prev_epoch_dist!= 9999:
        print('closest_match ', epoch_dist, 'epochs away')
        print(match_dict)
    
    return match_dict 

def data_masking(df_, criterion):
    #criterion  = 'Mass_0>3:Mass_0<6'
    operators=['>', '<', '<=', '>=', '==','!=', '&=', '|=', '^=', '>>=', '<<=']

    masks = []
    for mask in criterion.split(':'):
        mask_info = [mask.split(o)[0]+o+mask.split(o)[1] for o in operators if o in mask] #find match
        query_string = mask_info[0]
        df_=df_.query(query_string)
    return df_


def get_search_labels(params):
    from itertools import product
    labels = []
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, p)) for p in product(*values)]
    for run in combinations:
        labels.append("_".join([k+ '_'+ str(run[k])for k in run]))
    return labels

def get_search(path, params):
    out = []
    all_js= get_all_json(path)
    print(all_js)
    all_lb = get_search_labels(params)
    print(all_lb)
    for js in all_js:
        for lb in all_lb:
            if lb in js: 
                out.append(js)
    return out

def plot_loss(json_path):
    run_info = get_run(json_path)
    train_loss = np.load(run_info['loss_path'])['train_loss.npy']
    valid_loss = np.load(run_info['loss_path'])['valid_loss.npy']

    print(np.min(train_loss), np.min(valid_loss))
    epoch= np.arange(start= 1, stop =train_loss.size+1)
    df_loss=pd.DataFrame(data=np.column_stack([epoch,train_loss, valid_loss])
                        , columns=['epoch','train_loss','valid_loss'])                   
    plt.figure(figsize = (15,5))
    sns.lineplot(data=df_loss, x='epoch', y='valid_loss', label='Validation')
    sns.lineplot(data=df_loss, x='epoch', y='train_loss', label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()     
    
def sample_run(info):
    if isinstance(info, str):
        info=get_run(info)
    import numpy as np
    import torch
    from torch import nn
    from models.flows import MAF,BatchNormFlow
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.ticker as tck
    import corner.corner as cc 
    import pandas as pd
    import os
    import seaborn 
    
    device = torch.device('cpu') # choose device 'cpu' or 'cuda' 
    model = MAF(**info['MAFconfig']).to(device) # initialize the MAF model
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4) 
    checkpoint = torch.load(info['best_model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    

    model.eval() # evaluate mode

    #test_data=np.load('BHNSm.pq_train.npz')
    # i think this should be from a another source
    #pop_para =  np.load('BHBHm.pq_test.npz')['pop_parameters'] 
    df_test = pd.read_parquet('/home/melika/jake/Thesis_Notes/code/nf-emulator/data/BHBHm_Tr_0.6_Va_0.2_Te_0.2_val.pq')
    #np.array([df[['Z', 'alpha']].values])
    pop_para = np.array(df_test[['Z','alpha']].values)
                              
                              
    # set event parameters not sure how to change this one 
    #event = np.array([0,0]).reshape(1,2)
    event = np.repeat([[0,0]], len(pop_para), axis=0)
    # sample from theoretical dist. Whhich i dont have so did test
    #s1 = np.load('BHBHm.pq_test.npz')['samples'] 
    s1= np.array(df_test[['Mass_0','Mass_1']].values)

    a = torch.from_numpy(event).float().to(device)# event what
    b = torch.from_numpy(pop_para).float().to(device)
    print(model.log_probs(a, b)) 

    # sample from NF model
    s2 = model.sample(num_samples=len(pop_para), cond_inputs=b).detach().cpu().numpy()
    df_NF=pd.DataFrame(data=np.column_stack([s2,b]), index=None, columns=['Mass_0','Mass_1','Z','alpha'])
    return df_NF



########### we need to evaluate the probi first to 'activate' the model then do the sampling




    # Initialize the model #########################################
    alphas = np.linspace(0, 5, num = 1000)
    Z = np.repeat(0.0001, alphas.size)
    pop_para_ = np.column_stack([Z, alphas])
    event =np.array([0.0001, 4])
    
    #pop_para_ = np.unique(po_para, axis = 0)
    pop_tensor = torch.from_numpy(pop_para_).float().to(device)
    event_tensor = torch.from_numpy(event).float().to(device)
    log_probs = model.log_probs(pop_tensor, event_tensor)
    ################################################################
    



    Z = 0.0001
    alphas_unique = [0.5, 1, 3, 5]
    N_samples = 500

    i = 1
    for alp in alphas_unique:
        pop_row = np.array([Z,alp]).reshape(1,2)
        pop_row_block = np.repeat(pop_row, N_samples, axis=0)

        if i == 1:  pop_stack = pop_row_block
        else: pop_stack=np.vstack([pop_stack, pop_row_block])
        i+=1
    
    pop_stack_tensor = torch.from_numpy(pop_stack).float().to(device)
    model.eval()
    samples = model.sample(num_samples=len(pop_stack_tensor), cond_inputs=pop_stack_tensor).detach().cpu().numpy()
    
    def bool_conv(bool_v):
        bool_out = bool_v
        if  isinstance(bool_v, str):
            bool_conv = {'True' : True, 'true' : True, 'False': False, 'false': False}
            bool_out = bool_conv[bool_v]
        return bool_out
        
    print('info log', info['log'])
    if bool_conv(info['log']):
        print('log')
        samples= np.power(2, samples)
    else: print('Samples not logged')
    df_NF=pd.DataFrame(data=np.column_stack([samples, pop_stack]), index=None, columns=['Mass_0','Mass_1','Z','alpha'])
    return df_NF
    
def get_test_df(info):
    test_path=info['validation_file'][:-6]+'test.pq'
    return pd.read_parquet(test_path)

def plot_test_data(info):
    fig, ax = plt.subplots(figsize=(15,7), ncols=2)
    test_path=info['validation_file'][:-6]+'test.pq'
    df = pd.read_parquet(test_path)
    df.plot.scatter(ax=ax[0], x='Mass_0',y='Mass_1', c='DarkBlue', alpha = 0.01, xlabel= 'Mass 0', ylabel= 'Mass 1')
    df['Mass_0'].plot(ax=ax[1], kind='hist',x='Mass_0',bins=100, logy=False,alpha=0.4, label = 'Mass 0')
    df['Mass_1'].plot(ax=ax[1], kind='hist',x='Mass_1',bins=100, logy=False,alpha=0.4, label = 'Mass 1',color  ='green')
    ax[1].legend()
    ax[1].set_xlabel('Mass')
    

def plot_results(df, info, kind='scatter'):
    import matplotlib.pyplot as plt
    if kind =='scatter':
        fig, ax = plt.subplots(figsize=(15,7), ncols=2)
        df.plot.scatter(ax=ax[0], x='Mass_0',y='Mass_1', c='DarkBlue', alpha = 0.01, xlabel= 'Mass 0', ylabel= 'Mass 1')
        df['Mass_0'].plot(ax=ax[1], kind='hist',x='Mass_0',bins=100, logy=True,alpha=1, label = 'Mass 0')
        df['Mass_1'].plot(ax=ax[1], kind='hist',x='Mass_1',bins=100, logy=True,alpha=0.4, label = 'Mass 1',color  ='green')
        ax[1].legend()
        ax[1].set_xlabel('Mass')
    if kind == 'hist2D':
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(15,7), ncols =2)
        sns.histplot(data = data_masking(df, 'Mass_0<100:Mass_1<100'), x= 'Mass_0', y='Mass_1', ax=ax[0], cbar=True, cmap =  sns.color_palette("mako", as_cmap=True))
        ax[0].set_xlabel('Mass 0')
        ax[0].set_ylabel('Mass 1')

        sns.histplot(data = data_masking(df, 'Mass_0<9.2:Mass_0>4.5:Mass_1<9.2:Mass_1>4.5'),ax=ax[1], x= 'Mass_0', y='Mass_1', cbar=True, cmap =  sns.color_palette("mako", as_cmap=True))
        ax[1].set_xlabel('Mass 0')
        ax[1].set_ylabel('Mass 1')
        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.savefig('/home/melika/Pictures/Thesis_OLD2NEW', bbox_inches='tight')

        print('Mass > 100 =',(len(data_masking(df, 'Mass_0>100'))+len(data_masking(df, 'Mass_1>100')))/len(df), '%')
        print('Mass < 0 =',(len(data_masking(df, 'Mass_0<0'))+len(data_masking(df, 'Mass_1<0')))/len(df), '%')

def get_SEVN_df():
    import pandas as pd
    return pd.read_parquet('BHBHm.pq')

def eval_likelihood(info, samples, pop_parameters):
    import numpy as np
    import torch
    from torch import nn
    from models.flows import MAF,BatchNormFlow
    from tqdm import tqdm
    import time
    import pandas as pd
    import argparse
    from os import mkdir
    from os.path import join as p_join
    from os.path import exists as p_exists

    device = torch.device('cpu') # choose device 'cpu' or 'cuda' 
    model = MAF(**info['MAFconfig']).to(device) # initialize the MAF model
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4) 

    checkpoint = torch.load(info['best_model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    def loss_function(ps,cp):
        # We are aiming to find the mimumium for loss function in ML
        # so we add a negative sign to the llh to be our loss function
        likelihood = model.log_probs(ps,cp)
        return -torch.mean(likelihood)

    # function to read simulation data
    def data_transform(samples, pop_parameters):
        # If we have `Nsim` set of population parameters 
        # and each simulation with 'Nevent' binary merger events
        # and each event with `Ndim` event parameters,
        # Then samples and pop_parameters should come in with shape (Nsim*Nevent,Ndim)
        s = samples
        c = pop_parameters

        Nsim, Nevent, Ndim = s.shape
        if info['log']:
            s= np.log2(s)
        
        # convert to tensor form and to the device being used ( cpu or gpu )
        torch_s = torch.from_numpy(s).float()
        torch_c = torch.from_numpy(c).float()

        if not info['dataloader']:
            torch_s = torch.to(device) 
            torch_c = torch.to(device)
        #c = np.repeat(c,s.shape[1],axis=0)
        return Nsim, Nevent, Ndim, torch_s.reshape(Nsim*Nevent, Ndim), torch_c.reshape(Nsim*Nevent, Ndim)

    Nsim_train, Nevent_train, Ndim, samples, pop_parameters = data_transform(samples, pop_parameters) 

    print('Training set: {} simulation, each with {} events and each event with {} parameters'.format(Nsim_train, Nevent_train, Ndim))

    model.eval() 
    optimizer.zero_grad()
    with torch.no_grad():
        model(samples, pop_parameters)
            
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 1
                    
    # calculate validation loss
    with torch.no_grad():
        likelihood = model.log_probs(samples,pop_parameters)

    return likelihood.cpu().detach().numpy()

def test_likelihood(info):
    m = np.linspace(0.0, 80, num=300)
    M1, M2 = np.meshgrid(m, m)
    data = np.vstack((M1.flatten(), M2.flatten())).T
    hyperparams =np.repeat([[0.0001, 0.5]], len(data), axis =0)
    m = np.linspace(0.0, 80, num=300)
    M1, M2 = np.meshgrid(m, m)
    data = np.vstack((M1.flatten(), M2.flatten())).T
    likelihood=eval_likelihood(info, np.array([data]), np.array([hyperparams]))
    return hyperparams, data[:, 0], data[:, 1], np.exp(likelihood)

def plot_likelihood(info):
    hyperparams, x,y,z =test_likelihood(info)
    plt.scatter(x, y, c=z, cmap='viridis')
    #lb = '$p( d |\lambda)$' 
    lb = 'Likelihood'
    plt.colorbar(label=lb +' for $\lambda$ = (Z='+str(hyperparams[0][0])+', alpha ='+str(hyperparams[0][1])+')')  # Add a colorbar to show the function values
    plt.xlabel('Mass 0')
    plt.ylabel('Mass 1')