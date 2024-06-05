import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_loss(json_path, existing_ax = False, label = False):
    if isinstance(json_path, dict):
        run_info = json_path
    else:
        from tools.tuning import get_run
        run_info = get_run(json_path)
    lb =''
    if label is not False: lb=label
    if existing_ax is not False:
        ax = existing_ax
    else:
        fig, ax = plt.subplots(figsize = (15,5))
        
    train_loss = np.load(run_info['loss_path'])['train_loss.npy']
    valid_loss = np.load(run_info['loss_path'])['valid_loss.npy']

    print(np.min(train_loss), np.min(valid_loss))
    epoch= np.arange(start= 1, stop =train_loss.size+1)
    df_loss=pd.DataFrame(data=np.column_stack([epoch,train_loss, valid_loss])
                        , columns=['epoch','train_loss','valid_loss'])                   
    sns.lineplot(data=df_loss, x='epoch', y='valid_loss', label=f'Validation {lb}', ax=ax)
    sns.lineplot(data=df_loss, x='epoch', y='train_loss', label=f'Training {lb}',ax =ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()  
    
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

def plot_likelihood(info, hyperparams = [0.0001, 0.5], existing_ax = False, save = False, m_range=(0,60)):
    from tools.model import test_likelihood
    from matplotlib.cm import ScalarMappable, get_cmap
    #from mpl_toolkits.axes_grid1 import make_axes_locatable


    if existing_ax is not False:
        ax = existing_ax
    else: fig, ax =plt.subplots(figsize=(15,7))

    hyperparams, x,y,z = test_likelihood(info, hyperparams = hyperparams, m_range=m_range)
    cm = get_cmap('viridis')
    sc =ax.scatter(x, y, c=z, cmap=cm)
    ax.tick_params(axis='both', labelsize=14)
    #divider = make_axes_locatable(ax[1])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(sc, cax=cax)
    plt.colorbar(sc,fraction=0.046, pad=0.04)
    #lb = '$p( d |\lambda)$' 
    lb = 'Likelihood'
    ax.set_xlabel('Mass 0', fontsize=15 )
    ax.set_ylabel('Mass 1', fontsize=15)
    if save is not False:
        plt.tight_layout()
        plt.savefig(save)