# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor



def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,  
    drift_coefficient: Callable[float, float],
    diffusion_coefficient: Callable[float],
    seed: Union[int, None] = None
) -> Tensor:
    """Euler-Maruyama integrator (approximate)

     Args:
        x_0: The initial images of dimensions 
            (batch_size, n_channels, image_height, image_width)
        t_0: float,
        t_end: endpoint of the integration interval    
        n_steps: number of integration steps 
        drift_coefficient: Function of :math`(x(t), t)` that defines the drift term            
        diffusion_coefficient: Function of :math`(t)` that defines the diffusion term  
        seed: Seed for the random number generator
        
    Returns:
        x_t: Trajectories that result from the integration of the SDE.
             The shape is (*np.shape(x_0), (n_steps + 1))
            
    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t: - x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (2, 3, 5, 4)))
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient, 
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(np.shape(x_t))
        torch.Size([2, 3, 5, 4, 7])
    """
    
    # [TO DO: Include short comments to describe operation of the code]
    
    device = x_0.device  # initial condition

    # Discretización del tiempo en n_steps pasos entre t_0 y t_end
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    # Tamaño del paso temporal delta de t
    dt = times[1] - times[0]  
    
    # Inicializa el tensor para almacenar la trayectoria x_t
    # shape: (batch, canales, alto, ancho, tiempo)
    x_t = torch.tensor(
        np.empty((*np.shape(x_0), len(times))), 
        dtype=torch.float32,
        device=device,
    )

    # condicion inicial: x_t(t_0) = x_0
    x_t[..., 0] = x_0
    
    # Genera el ruido gaussiano para cada paso de tiempo y cada muestra del batch
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0 # No se introduce ruido en el último paso
    
    # iteración temporal (Euler-Maruyama)
    for n, t in enumerate(times[:-1]): 
        # expande t a un vector del mismo tamaño que el batch para evaluar los coeficientes de drift y difusión
        t = torch.ones(x_0.shape[0], device=device) * t

        # paso de integración de Euler-Maruyama:
        # x_{t+Δt} = x_t + drift*dt + diffusion*sqrt(dt)*z
        x_t[..., n + 1] = ( 
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t).view(-1, 1, 1, 1) 
              * torch.sqrt(torch.abs(dt)) 
              * z[..., n]
        )
        
    # Devuelve tiempos y trayectoria completa
    return times, x_t
    
class DiffussionProcess:

    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
     

        
class GaussianDiffussionProcess(DiffussionProcess):
    """
    
    [TO DO: Complete the docstring]
    
    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffussionProcess(
        ...     drift_coefficient=lambda x_t, t: mu,
        ...     diffusion_coefficient=lambda t: sigma,
        ...     mu_t=lambda x_0, t: x_0 + mu*t,
        ...     sigma_t=lambda t: np.sqrt(2.0 * t),
        ... )
        >>> print(bm.drift_coefficient(x_t=3.0, t=10.0))
        1.5
        >>> print(bm.diffusion_coefficient(t=10.0))
        2.0
        >>> print(bm.mu_t(x_0=3.0, t=10.0), bm.sigma_t(t=10.0))
        18.0 4.47213595499958
        

    """
    kind = "Gaussian"
    
    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
        mu_t: Callable[float, float] = lambda x_0, t: x_0,
        sigma_t: Callable[float] = lambda t: np.sqrt(t),
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t
    

    def loss_function(
        self,
        score_model, 
        x_0: torch.Tensor, 
        eps: float = 1.0e-5,
    ):
        """The loss function for training score-based generative models.

          Args:
              score_model:  A PyTorch model instance that represents a 
                            time-dependent score-based model.
          x_0: A mini-batch of training data.    
          eps: A tolerance value for numerical stability.
        """
        
        # sample t ~ Uniform(eps, 1.0)
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps  
        
        # ruido Z ~ N(0, I) con el mismo tamaño que x_0
        z = torch.randn_like(x_0)

        # media y desviación típica de la distribución de x_t condicionada a x_0
        sigma_t = self.sigma_t(t).view(-1, 1, 1, 1)
        mu_t = self.mu_t(x_0, t)

        # generar ruido gaussiano x_t ~ N(mu_t, sigma_t^2)
        x_t = mu_t + sigma_t * z

        # predecir el score con el modelo
        score = score_model(x_t, t)

        # loss
        loss = torch.mean(
            torch.sum((sigma_t * score + z) ** 2, dim=(1, 2, 3))  # mse por pixel, luego media sobre el batch
        )

        return loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()
