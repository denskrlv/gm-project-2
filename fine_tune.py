import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
from glide_text2im.download import load_checkpoint
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from ffhq_dataset import FFHQ_Dataset, get_ffhq_dataloader

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    attr_file_path = "/home/deniskrylov/.cache/kagglehub/datasets/kushsheth/face-vae/versions/1/list_attr_celeba.csv"
    root_dir_path = "/home/deniskrylov/.cache/kagglehub/datasets/kushsheth/face-vae/versions/1/img_align_celeba/img_align_celeba"

    # Create and configure model
    options = model_and_diffusion_defaults()
    options['use_fp16'] = True
    options['timestep_respacing'] = '100'

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**options)

    print("Loading checkpoint...")
    model.load_state_dict(load_checkpoint('base', device))
    model.to(device)
    # No need to call model.train() yet, we do it after wrapping with PEFT

    # --- START OF LORA MODIFICATION ---
    print("\nOriginal model trainable parameters:")
    print_trainable_parameters(model)

    # 1. Define the LoRA configuration
    # r: The rank of the update matrices. Lower rank means fewer parameters. 8 is a good starting point.
    # lora_alpha: LoRA scaling factor. A common setting is 2*r.
    # target_modules: The names of the layers to apply LoRA to. In GLIDE, 'qkv' is the main linear
    #                 projection in the self-attention blocks. This is the most effective place to adapt.
    # lora_dropout: Dropout probability for LoRA layers.
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv", "c_proj"], # Target Query-Key-Value and output projections in attention
        lora_dropout=0.1,
        bias="none", # We only train the LoRA weights, not the bias terms
    )

    # 2. Wrap the model with PEFT
    # This freezes all original weights and injects the trainable LoRA adapters.
    print("\nApplying LoRA configuration...")
    model = get_peft_model(model, config)
    
    print("\nModel with LoRA adapters trainable parameters:")
    print_trainable_parameters(model) # You will see a massive reduction here!
    # --- END OF LORA MODIFICATION ---
    
    model.train() # Now set the model to training mode

    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataloader = get_ffhq_dataloader(
        attr_file=attr_file_path,
        root_dir=root_dir_path,
        batch_size=64, # You can adjust batch size
        num_workers=0,
    )

    # Set up optimizer
    print("Setting up optimizer...")
    # The optimizer will automatically only update the trainable LoRA parameters
    optimizer = Adam(model.parameters(), lr=1e-4) # We can often use a slightly HIGHER LR with LoRA

    # Train
    print("Starting LoRA fine-tuning...")
    train(model, diffusion, dataloader, optimizer, device, options, num_epochs=10)

    # --- NEW: MERGE AND SAVE FULL MODEL ---
    print("\nMerging LoRA adapters into the base model...")
    # The `merge_and_unload()` method combines the adapters with the original weights
    # and returns a standard PyTorch model.
    merged_model = model.merge_and_unload()
    print("Model merged successfully.")

    # Now, save the state dictionary of the merged model like a regular PyTorch model.
    output_path = "glide_celeba_lora_finetuned_full.pt"
    torch.save(merged_model.state_dict(), output_path)

    print(f"\nTraining complete! Full fine-tuned model saved to '{output_path}'.")
    print("This file contains the entire model with adapters merged and can be loaded directly.")


def train(model, diffusion, dataloader, optimizer, device, options, num_epochs=10, start_epoch=0):
    """Train GLIDE on the CelebA dataset with tqdm progress tracking"""
    
    from torch.cuda.amp import autocast, GradScaler
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
                
                # The model output might have extra channels (for variance). We only need the first 3 for noise prediction.
                # In PEFT, the output is sometimes a tuple. We take the first element.
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
        
        # --- MODIFIED CHECKPOINT SAVING ---
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = f"lora_checkpoint_epoch_{epoch+1}"
            model.save_pretrained(checkpoint_dir)
            tqdm.write(f"LoRA checkpoint saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()