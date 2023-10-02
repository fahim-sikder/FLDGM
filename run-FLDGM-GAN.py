from utils import *
from module import *
import torch
import numpy as np
import argparse
import warnings
import matplotlib.pyplot as plt

import torch
from torch import autograd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from torch.utils.tensorboard import SummaryWriter

import seaborn as sb
import time
import os
import json
import pathlib
from tqdm import tqdm

warnings.filterwarnings('ignore')

class VectorDataset2(torch.utils.data.Dataset):
    def __init__(self, X, S):
        self.X = X
        self.S = S

    def __getitem__(self, i):
        x, s = self.X[i], self.S[i]
        return x, s
    
    def __len__(self):
        return self.X.shape[0]


class Generator(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 8)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.2, inplace = True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.relu(self.linear1(self.batch_norm1(x)))
        
        x = self.relu(self.linear2(self.batch_norm2(x)))
        
        x = self.relu(self.linear3(self.batch_norm3(x)))
        
        x = self.relu(self.linear4(self.batch_norm2(x)))
        
        x = self.linear5(x)
        
        return x
    
class Discrminator(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(8)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)

        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 1)

        self.relu = nn.LeakyReLU(0.2, inplace = True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.relu(self.linear1(self.batch_norm1(x)))
        
        x = self.relu(self.linear2(self.batch_norm2(x)))
        
        x = self.relu(self.linear3(self.batch_norm3(x)))
        
        x = self.linear4(x)
        
        return x

def gradient_penalty(real, fake, critic, device):
    
    
    m = real.shape[0]
    epsilon = torch.randn(m, 1)
    if device == 'cuda':
        epsilon = epsilon.cuda()

    interpolated_img = epsilon * real + (1-epsilon) * fake
    interpolated_out = critic(interpolated_img)

    grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
                          grad_outputs=torch.ones(interpolated_out.shape).cuda() if device == 'cuda' else torch.ones(interpolated_out.shape), create_graph=True, retain_graph=True)[0]
    grads = grads.reshape([m, -1])
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() 
    return grad_penalty

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
     



def main(args):
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = 'Adult-sex'

    loss = args.loss
    
    
    train_data, test_data, D = load_dataset(dataset)
    S_train, S_test = train_data.S.numpy(), test_data.S.numpy()
    Y_train, Y_test = train_data.Y.numpy(), test_data.Y.numpy()
    
    x_train, x_test = train_data.X.numpy(), test_data.X.numpy()
    
    x_data = np.concatenate([x_train, x_test], 0)
    s_data = np.concatenate([S_train, S_test], 0)
    
    catenated_data = VectorDataset2(x_data, s_data)

    
    # loss = 'wgan-gp'
    
    file_name = f'{dataset}-{loss}'
    
    folder_name = f'saved_files/{time.time():.4f}-{file_name}'
    

    
    
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    
    gan_fig_dir_path = f'{folder_name}/output/gan'
    
    pathlib.Path(gan_fig_dir_path).mkdir(parents=True, exist_ok=True) 
    
    batch_size = 2048
    verbose = 100
    
    GRADIENT_PENALTY = 10

    lr = 1e-3
    x_dim = train_data.X.shape[1]
    s_dim = train_data.S.max().item()+1
    h_dim = 64
    z_dim = 8
    
    logs = []
    
    

    model = FairDisCo(x_dim, h_dim, z_dim, s_dim, D)

    model.load('saved_weights/FairDisCo_Adult-sex_7.pkl')
    model.eval()

            
    train_loader = torch.utils.data.DataLoader(catenated_data, batch_size = 2048, shuffle = True)
    
    data_batch = next(iter(train_loader))
    
    x, s = data_batch
    
    with torch.no_grad():

        reaL_test_data_encoded = model.encode(x, s)
        
    generator = Generator().to(device)

    discriminator = Discrminator().to(device)
    
    gen_loss = []
    disc_loss = []
    
    clipping_rate = 0.01
    n_critic = 5
    epoch = 20000

    
    
    gen_optim = torch.optim.Adam(generator.parameters(), lr = 0.0002,  betas = (0.9, 0.999), weight_decay = 0.0001)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, betas = (0.9, 0.999), weight_decay = 0.0001)
    

    
    for gan_iter in tqdm(range(epoch)):

        for i, data in enumerate(train_loader):

            x, s = data

            x, s = x.to(device), s.to(device)

            batch_size = x.shape[0]

            z = torch.randn(batch_size, 64, dtype = torch.float, device = device)

            with torch.no_grad():

                reaL_data_encoded = model.encode(x.cpu(), s.cpu())

                reaL_data_encoded = reaL_data_encoded.to(device)
                
            if loss == 'wgan-gp':
                
                fake_output_gen = generator(z)
                
                ## train critic/discriminator
                
                disc_optim.zero_grad()
                
                real_output = discriminator(reaL_data_encoded.detach())
                fake_output = discriminator(fake_output_gen.detach())
                
                x_out = torch.cat([real_output, fake_output])
                
                d_loss = -(real_output.mean() - fake_output.mean()) + gradient_penalty(reaL_data_encoded, fake_output_gen, discriminator, device) * GRADIENT_PENALTY + (x_out ** 2).mean() * 0.0001
                
                d_loss.backward()
                
                disc_optim.step()
                
                ## train generator
                
                if i%n_critic == 0:
                    
                    z_fake = torch.randn(batch_size, 64, dtype = torch.float, device = device)
                    
                    gen_optim.zero_grad()
                    
                    gen_out = generator(z_fake)
                    
                    discr_out = discriminator(gen_out)
                    
                    g_loss = - discr_out.mean()
                    
                    g_loss.backward()
                    
                    gen_optim.step()
                
                

            elif loss == 'wgan':

                for p in discriminator.parameters():
                    p.required_grad = True

                discriminator.zero_grad()

                real_output = discriminator(reaL_data_encoded).view(-1)

                fake_output = discriminator(generator(z).detach()).view(-1)

                d_loss = -(torch.mean(real_output) - torch.mean(fake_output))

                d_loss.backward()

                disc_optim.step()

                disc_loss.append(d_loss.item())

                for p in discriminator.parameters():

                    p.data.clamp_(-clipping_rate, clipping_rate)

                if i%n_critic == 0:

                    for p in discriminator.parameters():

                        p.required_grad = False

                    generator.zero_grad()

                    fake_out = discriminator(generator(z)).view(-1)

                    g_loss = -torch.mean(fake_out)

                    g_loss.backward()

                    gen_optim.step()

                    gen_loss.append(g_loss.item())

            elif loss == 'lsgan':

                discriminator.zero_grad()

                real_output = discriminator(reaL_data_encoded).view(-1)

                fake_output = discriminator(generator(z).detach()).view(-1) 

                d_loss = 0.5 * ((torch.mean(real_output) - 1)**2) + 0.5 * (torch.mean(fake_output)**2) # loss for discriminator

                d_loss.backward()

                disc_optim.step()

                disc_loss.append(d_loss.item())

                ## GEN training

                generator.zero_grad()

                fake_out = discriminator(generator(z)).view(-1)

                g_loss = 0.5 * ((torch.mean(fake_out) - 1)**2)

                g_loss.backward()

                gen_optim.step()

                gen_loss.append(g_loss.item())


            if i%len(train_loader)==0 and gan_iter%100==0:

                print(f'GAN Epoch: [{gan_iter+1}/{epoch}], g_loss: {g_loss.item():.4f}, d_loss: {d_loss.item():.4f}')    

            if i%len(train_loader)==0 and gan_iter%500==0:   


                with torch.no_grad():

                    fixed_noise = torch.randn(512, 64, dtype = torch.float, device = device)

                    out_gen = generator(fixed_noise).detach().cpu()

                    visualize(reaL_test_data_encoded.detach().cpu(), out_gen, gan_fig_dir_path, gan_iter)
                    
                    torch.save({

                        'epoch': gan_iter+1,
                        'gen_state_dict_gan': generator.state_dict(),
                        'disc_state_dict_gan': discriminator.state_dict(),
                        'gen_optim_state_dict': gen_optim.state_dict(),
                        'disc_optim_state_dict': disc_optim.state_dict()

                        }, os.path.join(f'{folder_name}', f'{file_name}-ep-{gan_iter+1}.pth'))

    torch.save({

        'epoch': gan_iter+1,
        'gen_state_dict_gan': generator.state_dict(),
        'disc_state_dict_gan': discriminator.state_dict(),
        'gen_optim_state_dict': gen_optim.state_dict(),
        'disc_optim_state_dict': disc_optim.state_dict()

        }, os.path.join(f'{folder_name}', f'{file_name}-final.pth'))
                    
                    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--loss',
        choices=['wgan-gp','lsgan'],
        default='lsgan',
        type=str)
    
    args = parser.parse_args() 
    
    main(args)

    
    
