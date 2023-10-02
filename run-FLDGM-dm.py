from utils import *
from module import *
import torch
import numpy as np
import argparse
import warnings
import matplotlib.pyplot as plt

import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from torch.utils.tensorboard import SummaryWriter

import seaborn as sb
import time
import os
import json
import pathlib
from tqdm import tqdm

from ddpm import *

warnings.filterwarnings('ignore')

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def visualize(ori_data, fake_data, save_path, epoch):
    
    ori_data = np.asarray(ori_data)

    fake_data = np.asarray(fake_data)
    
    ori_data = ori_data[:fake_data.shape[0]]
    
    sample_size = 250
    
    idx = np.random.permutation(len(ori_data))[:sample_size]
    
    randn_num = np.random.permutation(sample_size)[:1]
    
    real_sample = ori_data[idx]

    fake_sample = fake_data[idx]
    
    real_sample_2d = real_sample
    
    fake_sample_2d = fake_sample
    
    
    mode = 'visualization'
        

        
    ### PCA
    
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_sample_2d))
                     .assign(Data='Synthetic'))
    pca_result = pca_real.append(pca_synthetic).rename(
        columns={0: '1st Component', 1: '2nd Component'})
    
    
    ### TSNE
    
    tsne_data = np.concatenate((real_sample_2d,
                            fake_sample_2d), axis=0)

    tsne = TSNE(n_components=2,
                verbose=0,
                perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)
    
    
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    
    tsne_result.loc[len(real_sample_2d):, 'Data'] = 'Synthetic'
    
    fig, axs = plt.subplots(ncols = 2, nrows=1, figsize=(10, 5))

    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                    hue='Data', style='Data', ax=axs[0])
    sb.despine()
    
    axs[0].set_title('PCA Result')


    sb.scatterplot(x='X', y='Y',
                    data=tsne_result,
                    hue='Data', 
                    style='Data', 
                    ax=axs[1])
    sb.despine()

    axs[1].set_title('t-SNE Result')

    fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', 
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    
    plt.savefig(os.path.join(f'{save_path}', f'{time.time()}-tsne-result-fairdisco-{epoch}.png'))


def main():
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = 'Adult-sex'
    
    train_data, test_data, D = load_dataset(dataset)
    S_train, S_test = train_data.S.numpy(), test_data.S.numpy()
    Y_train, Y_test = train_data.Y.numpy(), test_data.Y.numpy()
    
    loss = 'Diffusion'
    
    file_name = f'{dataset}-{loss}'
    
    folder_name = f'saved_files/{time.time():.4f}-{file_name}'
    
    
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    
    gan_fig_dir_path = f'{folder_name}/output/gan'

    pathlib.Path(gan_fig_dir_path).mkdir(parents=True, exist_ok=True) 
    
    batch_size = 2048
    epochs = 20
    verbose = 100

    lr = 1e-3
    x_dim = train_data.X.shape[1]
    s_dim = train_data.S.max().item()+1
    h_dim = 64
    z_dim = 8
    
    timesteps = 1000
    objective = 'pred_v'
    
    logs = []
    
    model = FairDisCo(x_dim, h_dim, z_dim, s_dim, D)


    model.load('saved_weights/FairDisCo_Adult-sex_7.pkl')
    model.eval()


    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 2048)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 500)
    
    data_batch = next(iter(test_loader))
    
    x, c, s, y = data_batch
    
    with torch.no_grad():

        reaL_test_data_encoded = model.encode(x, s)
        
        reaL_test_data_encoded = unnormalize_to_zero_to_one(reaL_test_data_encoded)
        
    model_dif = FairDiscoDiffusion(
        
        features = z_dim,
        latent_dim = 256    
    
    )
    
    diffusion = GaussianDiffusion1D(
        model_dif,
        timesteps = timesteps,  
        objective = objective, # pred_x0, pred_v
        loss_type = 'l2'   
    )
    
    diffusion = diffusion.to(device)

    lr = 1e-4
    
    betas = (0.9, 0.99)

    optim = torch.optim.Adam(diffusion.parameters(), lr = lr, betas = betas)

    epoch = 5000
    
    for diff_iter in tqdm(range(epoch)):

        for i, data in enumerate(train_loader):

            x, c, s, y = data

            x, c, s, y = x.to(device), c.to(device), s.to(device), y.to(device)

            batch_size = x.shape[0]

            with torch.no_grad():

                reaL_data_encoded = model.encode(x.cpu(), s.cpu())
                
                reaL_data_encoded = unnormalize_to_zero_to_one(reaL_data_encoded)

                reaL_data_encoded = reaL_data_encoded.to(device)
                
            optim.zero_grad()
            
            loss = diffusion(reaL_data_encoded)
            
            loss.backward()
            
            optim.step()


            if i%len(train_loader)==0 and diff_iter%10==0:

                print(f'Epoch: [{diff_iter+1}/{epoch}], diff_loss: {loss.item()}')    

            if i%len(train_loader)==0 and diff_iter%500==0:   

                with torch.no_grad():

                    samples = diffusion.sample(500)

                    samples = samples.cpu().numpy()
                    
                    np.save(f'{folder_name}/synth-latent-{dataset}-{diff_iter}.npy', samples)
                    
                    visualize(reaL_test_data_encoded.detach().cpu(), samples, gan_fig_dir_path, diff_iter)
                    
                    torch.save({

                        'epoch': diff_iter+1,
                        'diff_state_dict_gan': diffusion.state_dict(),
                        'diff_optim_state_dict': optim.state_dict()

                        }, os.path.join(f'{folder_name}', f'{file_name}-ep-{diff_iter+1}.pth'))

    torch.save({

        'epoch': diff_iter+1,
        'diff_state_dict_gan': diffusion.state_dict(),
        'diff_optim_state_dict': optim.state_dict()

        }, os.path.join(f'{folder_name}', f'{file_name}-final.pth'))
                    
                    
if __name__ == "__main__":
    
    main()

    
    
