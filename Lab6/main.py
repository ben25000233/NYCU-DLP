import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

#device = "cuda" if torch.cuda.is_available() else "cpu"

def noise_schedule(timesteps=500, start=0.0001, end=0.02):
    #linear_schedule
    return torch.linspace(start, end, timesteps)

def forward_diffusion_step(x0, t):
    T = 200
    betas = noise_schedule(timesteps=T)
    alphas = 1-betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_oneminus_alphas_cumprod = torch.sqrt(1-alphas_cumprod)

    noise = torch.randn_like(x0) 
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_oneminus_alphas_cumprod_t = sqrt_oneminus_alphas_cumprod[t]
    
    #element-wise的運算
    return sqrt_alphas_cumprod_t*x0 + sqrt_oneminus_alphas_cumprod_t * noise, noise
    #return sqrt_alphas_cumprod_t.to(device)*x0.to(device) + sqrt_oneminus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


import os
# Simulate forward diffusion

root = os.getcwd()
print(root + "/temp.png")
im = Image.open(root + "/temp.png")
print(im.format, im.size, im.mode)
im.show()

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(200/num_images)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

for idx in range(0, 200, stepsize):
    t = idx
    plt.subplot(1, num_images+1, (idx/stepsize) + 1)
    image, noise = forward_diffusion_step(image, t)
    show_tensor_image(image)

plt.show()