import os
import torch
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
from glide_text2im.download import load_checkpoint
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from celeba_dataset import CelebA_Dataset, get_ffhq_dataloader

# NEW: Import the PEFT library for LoRA
from peft import LoraConfig, get_peft_model

import warnings

# This will hide all FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NEW: Helper function to print the number of trainable parameters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GLIDE with LoRA on face images")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--num_epochs", type=int, default=20, 
                        help="Number of epochs to train")
    parser.add_argument("--checkpoint_freq", type=int, default=5, 
                        help="Save checkpoint every N epochs")
    parser.add_argument("--dataset", type=str, 
                        default="/home/deniskrylov/.cache/kagglehub/datasets/kushsheth/face-vae/versions/1", 
                        help="Path to root directory containing images")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="Directory to save checkpoints")
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Construct paths based on the dataset argument
    attr_file_path = os.path.join(args.dataset, "list_attr_celeba.csv")
    root_dir_path = os.path.join(args.dataset, "img_align_celeba", "img_align_celeba")

    # Create and configure model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = True
    options['timestep_respacing'] = '1000'
    options['noise_schedule'] = 'squaredcos_cap_v2'

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**options)

    print("Loading checkpoint...")
    model.load_state_dict(load_checkpoint('base', device))
    model.to(device)

    # Define the LoRA configuration
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv", "c_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    # Apply LoRA to the model
    print("\nApplying LoRA configuration...")
    model = get_peft_model(model, config)
    
    print("\nModel with LoRA adapters trainable parameters:")
    print_trainable_parameters(model)
    
    model.train()

    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataloader = get_ffhq_dataloader(
        attr_file=attr_file_path,
        root_dir=root_dir_path,
        batch_size=24,
        num_workers=0,
    )

    # Set up optimizer
    print("Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(os.path.join(args.checkpoint, "checkpoint.pt"))
        
        # Load adapter weights
        model.load_adapter(args.checkpoint, 'default')
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Resume from saved epoch
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}")

    # Train
    print(f"Starting LoRA fine-tuning from epoch {start_epoch}...")
    final_epoch, best_loss = train(
        model, 
        diffusion, 
        dataloader, 
        optimizer, 
        device, 
        options, 
        num_epochs=args.num_epochs, 
        start_epoch=start_epoch,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        best_loss=best_loss
    )

    # --- MERGE AND SAVE FULL MODEL ---
    print("\nMerging LoRA adapters into the base model...")
    merged_model = model.merge_and_unload()
    print("Model merged successfully.")

    # Save the state dictionary of the merged model
    output_path = os.path.join(args.checkpoint_dir, "glide_celeba_lora_finetuned_full.pt")
    torch.save(merged_model.state_dict(), output_path)

    print(f"\nTraining complete! Full fine-tuned model saved to '{output_path}'.")


def train(model, diffusion, dataloader, optimizer, device, options, 
          num_epochs=10, start_epoch=0, checkpoint_dir="checkpoints", 
          checkpoint_freq=5, best_loss=float('inf')):
    """Train GLIDE on the CelebA dataset with checkpoint support"""
    
    scaler = GradScaler() if options['use_fp16'] and device.type == 'cuda' else None
    
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs", position=0):
        total_loss = 0
        batch_count = 0
        
        batch_progress = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            leave=False, 
            position=1
        )
        
        for batch in batch_progress:
            images, prompts = batch
            images = resize(images, [64, 64])
            images = images.to(device)
            
            tokens = [model.tokenizer.encode(prompt) for prompt in prompts]
            token_data = [model.tokenizer.padded_tokens_and_mask(t, options['text_ctx']) 
                          for t in tokens]
            
            tokens_batch = torch.tensor([data[0] for data in token_data], device=device)
            mask_batch = torch.tensor([data[1] for data in token_data], device=device)
            
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, t, noise=noise)
            
            with autocast(enabled=options['use_fp16'] and device.type == 'cuda'):
                model_output = model(noisy_images, t, tokens=tokens_batch, mask=mask_batch)
                
                if isinstance(model_output, tuple):
                    model_output = model_output[0]

                predicted_noise = model_output[:, :3, :, :]
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            batch_progress.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        tqdm.write(f"Epoch {epoch+1} complete, Average Loss: {avg_loss:.6f}")
        
        # Update best loss
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or is_best or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save the LoRA adapter weights
            model.save_pretrained(checkpoint_path)
            
            # Save optimizer state, epoch, and loss info
            checkpoint = {
                'epoch': epoch,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict()
            }
            
            torch.save(checkpoint, os.path.join(checkpoint_path, "checkpoint.pt"))
            tqdm.write(f"Checkpoint saved to {checkpoint_path}")
            
            # If this is the best model so far, create a symlink or copy
            if is_best:
                best_path = os.path.join(checkpoint_dir, "best")
                if os.path.exists(best_path):
                    import shutil
                    shutil.rmtree(best_path)
                
                # Save as best model
                model.save_pretrained(best_path)
                torch.save(checkpoint, os.path.join(best_path, "checkpoint.pt"))
                tqdm.write(f"Best model saved (loss: {best_loss:.6f})")
    
    return num_epochs, best_loss

if __name__ == "__main__":
    main()
