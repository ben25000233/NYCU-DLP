import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.optim import AdamW
from diffusers import DDPMScheduler, UNet2DModel
from evaluator import evaluation_model
from dataloaders import get_picture
import dataloaders
import argparse
import os
from evaluator import evaluation_model
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, args):
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    training_set = get_picture("train")
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    epochs = args.epoch 

    for epoch in range(1, epochs+1):
        for step, (img, label) in enumerate(trainloader):
            img, label = img.to(args.device,dtype=torch.float32), label.to(args.device,dtype=torch.float32)
            #img has normalized in dataloader
            noise = torch.randn_like(img).to(args.device)

            timesteps = torch.randint(1, 1000, (img.shape[0],)).long().to(args.device)
            noisy_x = noise_scheduler.add_noise(img, noise, timesteps)
            
            noise_prediction = model(noisy_x.to(args.device), timesteps.to(args.device), label.to(args.device))
            noise_prediction = noise_prediction.sample
            loss = nn.MSELoss()(noise_prediction, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step %50 == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        if args.save_model:
            checkpoint_directory = os.path.join(os.getcwd(), "ckpt")
            os.makedirs(checkpoint_directory, exist_ok=True)  
            checkpoint_path = os.path.join(checkpoint_directory, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)

def sample(args, model, dataloader, mode):
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    model = model.to(args.device)
    result = os.path.join(os.getcwd(), mode)
    os.makedirs(result, exist_ok=True)  
    model.eval()
    xt = torch.randn(args.test_batch, 3, 64, 64).to(args.device)
    transform=transforms.Compose([
            transforms.Normalize((0, 0, 0), (1/0.5, 1/0.5, 1/0.5)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
        ])
    
    num_timesteps = noise_scheduler.num_train_timesteps
    timesteps_to_save = [int(i * num_timesteps / 6) for i in range(0, 6)]  # Divide timesteps into 5 equal parts
    timesteps_to_save.reverse() 

    fig, axs = plt.subplots(1, len(timesteps_to_save), figsize=(15, 6))  # Create a 1x5 subplot grid
    
    for index, label in enumerate(dataloader):
        label = label.reshape((32, 24))
        label = label.to(args.device)

        for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
            with torch.no_grad():
                residual = model(xt.to(args.device), t.to(args.device),label.to(torch.float32).to(args.device)) 
            xt = noise_scheduler.step(residual.sample.to(args.device), t.to(args.device), xt.to(args.device)).prev_sample

            if t.item() in timesteps_to_save:
                denoised_img = xt.clone().cpu()
                denoised_img = denoised_img.squeeze(0)
                denoised_img = transform(denoised_img)
                denoised_img = denoised_img[0]
                t_index = timesteps_to_save.index(t.item())
                ax = axs[t_index]  # Choose the appropriate subplot
                ax.imshow(denoised_img.permute(1, 2, 0))  # Transpose the image dimensions
                ax.set_title(f'Timestep: {t.item()}')
                ax.axis('off')

        evaluate = evaluation_model()
        acc = evaluate.eval(xt.to(args.device), label.to(args.device))
        print("Test Result : ", acc)

        plt.tight_layout()
        save_path = os.path.join(result, mode + '_subplot.png')
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to release resources
        print(f"Subplot images saved to {save_path}")

        
        img = transform(xt)
        save_images(mode, img)

def save_images(mode, images):
    grid = torchvision.utils.make_grid(images)
    save_image(grid, fp = ".\\" + mode +".png")

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4 * 0.5)
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--action',  default="test")
    args = parser.parse_args()

    model = UNet2DModel(
        sample_size = 64,
        in_channels = 3,
        out_channels = 3,
        layers_per_block = 2,
        block_out_channels = (128, 128, 256, 256, 512, 512), 
        down_block_types=(
        "DownBlock2D",  
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D", 
        "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", 
            "AttnUpBlock2D", 
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    model.class_embedding = nn.Linear(24 ,512)
    model = model.to(args.device)
    
    test_data = dataloaders.get_testing_data("test.json")
    new_test_data = dataloaders.get_testing_data("new_test.json")

    test_loader = torch.utils.data.DataLoader(test_data,shuffle=False)
    new_test_loader = torch.utils.data.DataLoader(new_test_data,shuffle=False)

    if args.action == "train":
        train(model, args)
        sample(args, model, test_loader, "test_result")
        sample(args, model, new_test_loader, "new_test_result")
    else:
        #use pretrain model 
        path = os.path.join(os.getcwd(), "ckpt")
        path = os.path.join(path, "checkpoint_epoch_20.pt")
        model.load_state_dict(torch.load(path))
        print(f'Finished loading model from {path}')
        model = model.to(args.device)
            
        sample(args, model, test_loader, "test_result")
        sample(args, model, new_test_loader, "new_test_result")


if __name__ == '__main__':
    main()
