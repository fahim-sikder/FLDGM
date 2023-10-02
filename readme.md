# Fair Latent Deep Generative Models (FLDGMs) for Syntax-agnostic and Fair Synthetic Data Generation

First, we need to install the docker container by running the shell `build.sh`, after that we can run the environment by using `run-shell.sh` shell.

```terminal
chmod +x build.sh
./build.sh
chmod +x run-shell.sh
./run-shell.sh
```

## Training

To train FLDGM with GAN architecture, run the following command inside the docker container.

For wgan-gp pass `wgan-gp` parameter and for LSGAN, pass `lsgan`

```terminal
python3 run-FLDGM-GAN.py --loss wgan-gp
```

To run the Diffusion architecture, run the following command.

```terminal
python3 run-FLDGM-dm.py
```

### Generated samples from the runtime can be found in the `saved_files` directory after running the experiments.