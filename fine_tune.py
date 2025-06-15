import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
from glide_text2im.download import load_checkpoint

from celeba_dataset import CelebA_Dataset, get_celeba_dataloader

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # --- START OF THE FIX ---
    # Get the absolute path to the directory where fine_tune.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory is: {script_dir}")

    # Construct absolute paths to your data directories
    attr_file_path = "/Users/deniskrylov/Downloads/archive/list_attr_celeba.csv"
    root_dir_path = "/Users/deniskrylov/Downloads/archive/img_align_celeba/img_align_celeba"

    print(f"Expecting attributes CSV at: {attr_file_path}")
    print(f"Expecting image directory at: {root_dir_path}")
    # --- END OF THE FIX ---

    # Create and configure model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False
    options['timestep_respacing'] = '100'

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**options)

    print("Loading checkpoint...")
    model.load_state_dict(load_checkpoint('base', device))
    model.to(device)
    model.train()

    # Create dataset and dataloader USING THE NEW ABSOLUTE PATHS
    print("Creating dataset...")
    dataloader = get_celeba_dataloader(
        attr_file=attr_file_path,
        root_dir=root_dir_path,
        batch_size=64,
        num_workers=0, # Keep this at 0 for now
    )

    # Set up optimizer
    print("Setting up optimizer...")
    optimizer = Adam(model.parameters(), lr=1e-5)

    # Train
    print("Starting training...")
    train(model, diffusion, dataloader, optimizer, device, options, num_epochs=10)

    # Save model
    torch.save(model.state_dict(), "glide_celeba_finetuned.pt")
    print("Training complete and model saved!")

def train(model, diffusion, dataloader, optimizer, device, options, num_epochs=10):
    """Train GLIDE on the CelebA dataset with tqdm progress tracking"""
    
    # Outer loop for epochs with tqdm
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        total_loss = 0
        batch_count = 0
        
        # Create progress bar for batches
        batch_progress = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            leave=False, 
            position=1
        )
        
        for batch in batch_progress:
            # Get your images and prompts
            images, prompts = batch
            images = images.to(device)
            
            # Process text tokens
            tokens = [model.tokenizer.encode(prompt) for prompt in prompts]
            token_data = [model.tokenizer.padded_tokens_and_mask(t, options['text_ctx']) 
                          for t in tokens]
            
            # Unpack token data
            tokens_batch = torch.tensor([data[0] for data in token_data], device=device)
            mask_batch = torch.tensor([data[1] for data in token_data], device=device)
            
            # Forward diffusion process
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, t, noise=noise)
            
            # Model prediction
            model_output = model(noisy_images, t, tokens=tokens_batch, mask=mask_batch)
            
            # Loss calculation (predict the noise)
            loss = torch.nn.functional.mse_loss(model_output, noise)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            current_loss = loss.item()
            total_loss += current_loss
            batch_count += 1
            
            # Update progress bar description with current loss
            batch_progress.set_postfix({"loss": f"{current_loss:.6f}"})
        
        # Calculate and display epoch average loss
        avg_loss = total_loss / batch_count
        tqdm.write(f"Epoch {epoch+1} complete, Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"glide_celeba_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()