# File tools for tuning hyperparameters
def run_combinations(params, outdir, time = False):
    import sys
    sys.path.append('..')
    from os.path import join
    from itertools import product
    from tools.model import train
    from numpy import empty

    params = {key :list(params[key]) for key in params}
    params['outdir'] = [outdir]
    json_time_lb=''
    if time is not False: params['time'] = [time]
    if 'time' in params: json_time_lb =f'_{time}min'

    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, p)) for p in product(*values)]
    print('There are ', len(combinations),' combinations in this search')
    print(combinations)
    run_no = 0
    best_run_loss = 9999
    results = []
    for i,  run in enumerate(combinations): #loop through unique combinations of parameters
        run_no+=1
        out_label = "_".join([f'{k}_{run[k]}' for k in run])
        run['label'] = out_label
        
        try:
            output_dict = train(run, return_info=True)
        except:
            print(f'Run {out_label} Failed')
            if i==0: best_info_path = join(params['outdir'][0], f'{out_label}{json_time_lb}.json')
            break

        results.append([run,output_dict['best_loss']])
        if output_dict['best_loss'] < best_run_loss:
            best_run_loss = output_dict['best_loss']
            best_run= run
            best_info_path = join(params['outdir'][0], f'{out_label}{json_time_lb}.json')
    
    return results, best_info_path


def get_all_files(path):
    #Find all files in a directory
    from os import walk
    from os.path import join, abspath
    files = []
    for dirpath, _, filenames in walk(path):
        for f in filenames:
            files.append(abspath(join(dirpath, f)))
    return files

def get_run(path):
    #Get the information from a json file, loads it into a dictionary
    from json import load as json_load
    json_f = open(path)
    info = json_load(json_f)
    return info 

def get_all_json(path): 
    #Get all json files in a directory
    all_files= get_all_files(path)
    all_json = [f for f in all_files if f.endswith('.json')]
    return all_json
    
# Filtering tools
def get_matches(path, attributes, criteria = '=='):
    #Get all json files that match a particular set of attributes
    #Can prevent need to retrain models

    #Usage example: find all that have 100 epochs
    #matches= get_matches(os.path.join(os.getcwd(),'tuning'), {'epochs' : 100}, criteria = '==')

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
    #Get the best run (lowest loss) from a directory
    #Loads dictionary need to run 
    from json import load as json_load

    all_files= get_all_files(path)
    all_json = [f for f in all_files if f.endswith('.json') and 'pq_trial' not in f]
    best_run_loss = 9999
    final_loss_key = 'best_loss'
    for f in all_json: 
        json_f = open(f)
        info = json_load(json_f)
        if log == info['log']:
            if final_loss_key not in json_f:
                final_loss_key = 'final_loss'
            best_run_loss = info[final_loss_key]
            best_run_info = info
            print('Best run loss',  best_run_loss)
    return best_run_info

def get_search_labels(params):
    from itertools import product
    labels = []
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, p)) for p in product(*values)]
    for run in combinations:
        labels.append("_".join([k+ '_'+ str(run[k])for k in run]))
    return labels

def get_search(path, params):
    #get all json files that match a particular set of tuning parameters

    out = []
    all_js= get_all_json(path)
    #print(all_js)
    all_lb = get_search_labels(params)
    #print(all_lb)
    for js in all_js:
        for lb in all_lb:
            if lb in js: 
                out.append(js)
    return out

# Results tools
def sf_round(num):
    return float('%.3g' % num)

def get_results_table(path, params):
    from pandas import DataFrame as pd_df
    from pandas import concat as pd_concat
    from numpy import column_stack as np_column_stack
    from os.path import exists as os_exists

    if not os_exists(path):
        print('Path does not exist: ', path)  
        return None

    matches = get_search(path, params)

    #Find the last time point all runs passed  
    t_small=999999999
    for m in matches:
        m_run = get_run(m)
        t = m_run['epoch_time'][-1]
        if t<t_small:
            t_small = t
            run_small = m_run['label']
    
    print('Shortest Run:', t_small, run_small)
    
    i=0
    for m in matches: 
        m_run = get_run(m)
        
        df_run_info=pd_df(data=np_column_stack([m_run['epochs'], m_run['A'], m_run['blocks'], 
                                                       m_run['hidden'], sf_round(m_run['best_loss']),
                                                       sf_round(m_run['d_score_rate'][-1]), sf_round(m_run['runtime']) ]),
                                                   columns=['Epochs','Activation','Blocks','Hidden','L_V','Loss Rate', 'Runtime [s]'])  
        #df_run_info['Run']=m_run['label']
        if i==0:
            df_all_info=df_run_info   
        else:
            df_all_info=pd_concat([df_all_info, df_run_info])
        i+=1
    print(df_all_info.sort_values('Loss Rate', ascending=True).to_latex(index=False))
    return df_all_info

def get_results_loss(path, params):
    from pandas import DataFrame as pd_df
    from pandas import concat as pd_concat
    from numpy import column_stack as np_column_stack
    from numpy import arange as np_arange
    from numpy import load as np_load

    matches = get_search(path, params)
    i=0
    print(matches)
    for m in matches: 
        m_run = get_run(m)
        train_loss = np_load(m_run['loss_path'])['train_loss.npy']
        valid_loss = np_load(m_run['loss_path'])['valid_loss.npy']

        epoch= np_arange(start= 1, stop =train_loss.size+1)
        df_loss=pd_df(data=np_column_stack([epoch, train_loss, valid_loss]), columns=['epoch','train_loss','valid_loss'])  
        df_loss['run']=m_run['label']
        get_run
        
        if i==0:
            df_all_loss=df_loss   
        else:
            df_all_loss=pd_concat([df_all_loss, df_loss])
        i+=1
    return df_all_loss

def data_masking(df_, criterion):
    #criterion  = 'Mass_0>3:Mass_0<6'
    #needed until we find a better way of 
    operators=['>', '<', '<=', '>=', '==','!=', '&=', '|=', '^=', '>>=', '<<=']

    masks = []
    for mask in criterion.split(':'):
        mask_info = [mask.split(o)[0]+o+mask.split(o)[1] for o in operators if o in mask] #find match
        query_string = mask_info[0]
        df_=df_.query(query_string)
    return df_

def run_combinations_CMD(params, outdir, time = False):
    #Should aim to use standard verison as calling CMD from script not the best practice
    #This can test the argparse usage
    
    from os.path import join as p_join
    from itertools import product
    import json

    params = {key :list(params[key]) for key in params}
    if 'time' in params:time = params['time']
      
    config_cmds, time_lb = '', ''
    config_cmds += f' -O {outdir}'

    results_ = []
    keys, values = zip(*params.items())
    print(keys, values)
    combinations = [dict(zip(keys, p)) for p in product(*values)]
    print('There are ', len(combinations),' combinations in this search')
    print(combinations)
    run_no = 0
    best_run_loss = 9999
    json_time_lb =''
    for run in combinations: #loop through unique combinations of parameters
        run_no+=1
        print('Run No = ', run_no, ' : ', run)
        cmd = " ".join([f'-{k} {run[k]}' for k in run])
        
        #makes a label based on unique inputs
        out_label = "_".join([f'{k}_{run[k]}' for k in run])
        if time is not False: 
            config_cmds += f' -time {time}'
            json_time_lb =f'_{time}min'
            
        #print('Run: '+ str(run_no)+', Label: ' +  out_label)
        cmd += f'{config_cmds} -label {out_label} '
        print(cmd)
        !python ../nf_train.py $cmd

        # Running the training script will generate a json that we open
        json_path = p_join(outdir, f'{out_label}{json_time_lb}.json')
        json_file = open(json_path)
        output_dict = json.load(json_file)

        results_.append([run,output_dict['best_loss']])
        if output_dict['best_loss'] < best_run_loss:
            best_run_loss = output_dict['best_loss']
            best_run= run
            best_info_path = json_path
            
    print('Best Run = ', best_run, ', Loss = ', best_run_loss) 
    return results_, best_info_path