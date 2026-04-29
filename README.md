# Diffusion Models Project (Project_AA3)

This project implements score-based generative models using diffusion processes, developed as part of the Machine Learning course at Universidad Autónoma de Madrid.

## Authors
- Maya Ceamanos López-Sanvicente
- Sandra Yunta Martín

## Project Overview

The goal of this project is to study generative models capable of learning the underlying distribution of a dataset and generating new realistic samples from random noise.

In particular, we focus on **score-based diffusion models**, where data generation is formulated as the inversion of a stochastic diffusion process.

## Methodology

The project is based on:

- Stochastic Differential Equations (SDEs)
- Score-based generative modeling
- Denoising score matching
- Numerical methods such as Euler–Maruyama

We implemented both:

- **Variance Exploding (VE)** processes (Brownian motion)
- **Variance Preserving (VP)** processes (Ornstein–Uhlenbeck)

## Project Structure

```
├── score_model.py              # Neural network (ScoreNet)
├── diffusion_process.py        # Diffusion processes and loss
├── diffusion_utilities.py      # Visualization tools
├── notebooks/
│   ├── demo_MNIST_diffusion.ipynb
│   ├── demo_diffusion_models_generative_AI.ipynb
│   ├── project_AAIII_teamCode_Yunta_Ceamanos.ipynb
│   ├── RGB_IMAGES.ipynb
│   └── use_cases.ipynb
```


## Main Features

- Training of score-based generative models
- Simulation of forward diffusion processes
- Image generation via reverse diffusion
- Support for different noise schedules:
  - Linear
  - Cosine
  - Quadratic
- Visualization of generated samples and trajectories
- Improve using RGB also in the data

## Results

The models are capable of generating realistic MNIST digits using different diffusion configurations.

Experimental results show that:

- VP processes (Ornstein–Uhlenbeck) achieve better performance
- Noise schedule significantly affects model quality
- Euler–Maruyama provides a simple and effective sampling method

## How to Run

1. Open and run: 
demo_MNIST_diffusion.ipynb
demo_diffusion_models_generative_AI.ipynb

2. Open and run the main notebook (Modify parameters (epochs, noise schedule, etc.) to experiment):
project_AAIII_teamCode_Yunta_Ceamanos.ipynb

3. Open and run:
RGB_IMAGES.ipynb


## Requirements

- Python 3.10
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## Datasets

The project is evaluated using standard image datasets:

- **MNIST**: grayscale handwritten digit dataset (28x28)
- **CIFAR-10**: RGB image dataset used for extension experiments

The datasets are loaded using `torchvision.datasets`.

Note: datasets are not included in the repository due to size constraints.

---

## License

This project is developed for academic purposes.