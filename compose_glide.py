import math
from matplotlib import pyplot as plt
import numpy as np
import torch as th
from transformers import AutoTokenizer

from functools import partial
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from glide_text2im.unet import QKVAttention
from utils import get_gpu, prepare_prompt


class ComposeGlide:
    
    def __init__(self, model_name='base', device=None, verbose: bool=True):
        if device is None:
            self.device = get_gpu()
        else:
            self.device = device

        self.verbose = verbose

        self.base_options = model_and_diffusion_defaults()
        self.base_options['inpaint'] = (model_name == 'base-inpaint')
        self.base_options['use_fp16'] = (self.device.type == 'cuda')
        self.base_options['timestep_respacing'] = '100' # As in notebook
        self.base_model, self.base_diffusion = self._load_model_internal(
            model_name=model_name,
            options=self.base_options
        )

        self.up_options = model_and_diffusion_defaults_upsampler()
        self.up_options['use_fp16'] = (self.device.type == 'cuda')
        self.up_options['timestep_respacing'] = '100'
        self.up_model, self.up_diffusion = self._load_model_internal(
            model_name='upsample',
            options=self.up_options
        )

        # Patch attention modules to expose weights
        self._patch_attention_modules()

        # Register module names for all attention blocks
        self._register_attention_module_names(self.base_model, prefix="base")
        self._register_attention_module_names(self.up_model, prefix="upsample")

    def _patch_attention_modules(self):
        """Patch QKVAttention to capture attention weights during forward passes"""
        
        # Original QKVAttention forward method
        original_forward = QKVAttention.forward
        
        # Storage for attention weights
        self.collected_attention_weights = []
        
        # Define the patched forward method that captures attention weights
        def patched_forward(self_attn, qkv, encoder_out=None):
            """Patched forward method that captures attention weights during inference"""
            # Call the original forward method to get the result
            result = original_forward(self_attn, qkv, encoder_out)
            
            # Extract and store attention weights
            try:
                bs, width, length = qkv.shape
                assert width % 3 == 0
                ch = width // 3
                    
                # Split qkv into query, key, value
                q, k, v = qkv.reshape(bs, 3, ch, length).unbind(dim=1)
                
                # Calculate attention weights (batch_size, sequence_length, sequence_length)
                # This matches how attention is typically calculated in transformers
                scale = 1 / math.sqrt(math.sqrt(ch))
                weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
                weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
                
                # Get spatial dimensions for proper reshaping of attention maps
                # For spatial attention, we need to reshape to the image dimensions
                side_length = int(math.sqrt(length))
                
                # Store the attention weights along with metadata
                if hasattr(self_attn, 'current_module_name'):
                    # Here we store the actual attention weights
                    self.collected_attention_weights.append({
                        'module_name': self_attn.current_module_name,
                        'is_cross_attention': encoder_out is not None,
                        'attention_weights': weight.detach().cpu(),  # Save weights to CPU to save GPU memory
                        'spatial_shape': (side_length, side_length)  # Save spatial dimensions for reshaping
                    })
            except Exception as e:
                # If anything goes wrong, we don't want to crash the generation process
                print(f"Warning: Failed to capture attention weights: {e}")
                pass
            
            # Return the unmodified result
            return result
        
        # Replace the original forward with our patched version
        QKVAttention.forward = patched_forward
        
        # Add a method to register module names
        def add_module_name(self, name):
            """Add module name to the attention block for identification"""
            self.current_module_name = name
            return self
        
        QKVAttention.add_module_name = add_module_name
    
    def _register_attention_module_names(self, model, prefix=""):
        """Register names for all QKVAttention modules in the model"""
        for name, module in model.named_modules():
            if isinstance(module, QKVAttention):
                full_name = f"{prefix}.{name}" if prefix else name
                module.add_module_name(full_name)

    def _load_model_internal(self, model_name, options):
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        if (self.device.type == 'cuda' and options.get('use_fp16', False)):
            model.convert_to_fp16()
        model.to(self.device)
        model.load_state_dict(load_checkpoint(model_name, device=self.device))
        return model, diffusion

    def _base_model_fn(self, x_t, ts, weights_param, **kwargs):
        half = x_t[:1]
        combined = th.cat([half] * x_t.size(0), dim=0)
        model_input_kwargs = {k: v for k, v in kwargs.items() if k in ['tokens', 'mask']}
        model_out = self.base_model(combined, ts, **model_input_kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = eps[:-1], eps[-1:]
        half_eps = uncond_eps + (weights_param * (cond_eps - uncond_eps)).sum(dim=0, keepdims=True)
        eps_final = th.cat([half_eps] * x_t.size(0), dim=0)
        return th.cat([eps_final, rest], dim=1)
    
    def visualize_attention_maps(self, image_tensor, attention_data, prompt, similarity_threshold=0.1, 
             output_path=None, show_plot=True):
        """
        Visualize attention maps from the diffusion model.
        
        Args:
            image_tensor: The generated image as a tensor
            attention_data: List of dictionaries containing attention weights from different timesteps
            prompt: The text prompt used to generate the image
            similarity_threshold: Threshold for filtering attention maps
            output_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        # Convert image tensor to numpy for visualization
        image_tensor = image_tensor.cpu()
        
        # Handle different possible input formats
        if len(image_tensor.shape) == 4:  # [batch, channels, height, width]
            image_tensor = image_tensor.squeeze(0)
        
        if len(image_tensor.shape) == 2:  # [height, width] (grayscale)
            # Convert grayscale to RGB
            image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Now tensor should have shape [channels, height, width]
        image_array = ((image_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image_array = image_array.permute(1, 2, 0).numpy()
        
        # Check if we have any attention data
        if not attention_data:
            print("No attention data available to visualize!")
            return None
        
        # Split the prompt to get individual concepts
        concepts = [p.strip() for p in prompt.replace(' and ', ',').split(',')]
        
        # Create a figure for visualization
        fig = plt.figure(figsize=(18, 6 * len(concepts)))
        
        # Tokenize the prompt to get token indices for each concept
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Get token IDs for the full prompt and individual concepts
        full_prompt_tokens = tokenizer.encode(prompt)
        concept_token_ids = {concept: tokenizer.encode(concept)[1:-1] for concept in concepts}  # Remove BOS/EOS tokens
        
        # Find where each concept's tokens appear in the full prompt
        concept_indices = {}
        for concept, token_ids in concept_token_ids.items():
            # Find start position of this concept's tokens in the full prompt
            for i in range(len(full_prompt_tokens) - len(token_ids) + 1):
                if full_prompt_tokens[i:i+len(token_ids)] == token_ids:
                    concept_indices[concept] = list(range(i, i+len(token_ids)))
                    break
        
        # Filter for cross-attention maps
        cross_attention_maps = []
        
        # Get attention maps from the middle of generation process (more stable attention)
        middle_timestep_idx = len(attention_data) // 2
        if middle_timestep_idx < len(attention_data):
            timestep_data = attention_data[middle_timestep_idx]
            
            # Extract the maps from this timestep
            for attention_item in timestep_data.get('maps', []):
                if attention_item.get('is_cross_attention', False):
                    cross_attention_maps.append(attention_item)
        
        # If we don't have cross-attention maps from middle, try to find from any timestep
        if not cross_attention_maps and attention_data:
            for timestep_data in attention_data:
                for attention_item in timestep_data.get('maps', []):
                    if attention_item.get('is_cross_attention', False):
                        cross_attention_maps.append(attention_item)
                        
        # Still no cross-attention maps found
        if not cross_attention_maps:
            print("No cross-attention maps found in the attention data!")
            return None
        
        # Process each concept in the prompt
        concept_similarities = {}  # Store similarity scores for each concept
        
        for idx, concept in enumerate(concepts):
            concept = concept.strip()
            if not concept:
                continue
            
            # Get image dimensions
            h, w = image_array.shape[:2]
            
            # Create an aggregate attention map for this concept
            aggregated_map = np.zeros((h, w))
            maps_averaged = 0
            
            # Get token indices for this concept
            concept_token_idx = concept_indices.get(concept, [])
            if not concept_token_idx:
                print(f"Could not locate token indices for concept: {concept}")
                continue
            
            # Process cross-attention maps
            for attn_map_data in cross_attention_maps:
                if not isinstance(attn_map_data.get('attention_weights'), th.Tensor):
                    continue
                    
                # Get the attention weights and reshape to spatial dimensions
                attention_weights = attn_map_data['attention_weights']
                spatial_shape = attn_map_data.get('spatial_shape', (int(np.sqrt(attention_weights.shape[-1])), 
                                                                int(np.sqrt(attention_weights.shape[-1]))))
                
                try:
                    # Extract attention only for the specific concept tokens
                    concept_attention = attention_weights[:, :, concept_token_idx].mean(-1)
                    
                    # Reshape to spatial dimensions
                    spatial_attn = concept_attention.reshape(-1, *spatial_shape)
                    
                    # Average across batch dimension if present
                    if spatial_attn.shape[0] > 1:
                        spatial_attn = spatial_attn.mean(0)
                    else:
                        spatial_attn = spatial_attn.squeeze(0)
                    
                    # Resize the attention map to match image dimensions
                    from scipy.ndimage import zoom
                    zoom_factors = (h / spatial_attn.shape[0], w / spatial_attn.shape[1])
                    resized_attention = zoom(spatial_attn.numpy(), zoom_factors, order=1)
                    
                    # Normalize the attention map
                    min_val, max_val = resized_attention.min(), resized_attention.max()
                    if max_val > min_val:
                        resized_attention = (resized_attention - min_val) / (max_val - min_val)
                    
                    # Accumulate attention
                    aggregated_map += resized_attention
                    maps_averaged += 1
                except Exception as e:
                    print(f"Error processing attention map for concept '{concept}': {e}")
                    continue
            
            # Visualize the concept's attention map
            if maps_averaged > 0:
                # Average and normalize
                aggregated_map /= maps_averaged
                
                # Calculate similarity metrics
                mean_similarity = np.mean(aggregated_map)
                max_similarity = np.max(aggregated_map)
                coverage = np.mean(aggregated_map > similarity_threshold)
                
                # Store similarity metrics
                concept_similarities[concept] = {
                    'mean': mean_similarity,
                    'max': max_similarity,
                    'coverage': coverage * 100  # Convert to percentage
                }
                
                # Apply threshold for visualization clarity
                aggregated_map = np.where(aggregated_map > similarity_threshold, aggregated_map, 0)
                
                # Plot
                ax1 = fig.add_subplot(len(concepts), 3, idx*3 + 1)
                ax1.set_title(f"Original Image")
                ax1.imshow(image_array)
                ax1.axis('off')
                
                ax2 = fig.add_subplot(len(concepts), 3, idx*3 + 2)
                # Include similarity metrics in the title
                similarity_text = f"Attention Map: {concept}\nMean: {mean_similarity:.3f}, Max: {max_similarity:.3f}, Coverage: {coverage*100:.1f}%"
                ax2.set_title(similarity_text)
                im = ax2.imshow(aggregated_map, cmap='jet')
                ax2.axis('off')
                
                # Add a colorbar to show similarity scale
                cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                cbar.set_label('Similarity')
            else:
                print(f"No valid attention maps found for concept: {concept}")
        
        plt.tight_layout()
        
        # Save the figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            if self.verbose:
                print(f"Attention map visualization saved to {output_path}")
        
        # Display the figure if show_plot is True
        if show_plot:
            plt.show()
        
        return fig, concept_similarities  # Return both the figure and similarity metrics

    def _sample_base(self, prompts, batch_size_param, guidance_weights, progress=True, save_intermediate_steps: int = None, image_idx_start: int = 0, prompt_for_filename: str = "prompt"):
        if batch_size_param != 1 and self.verbose:
            print(f"Warning: _sample_base is designed for batch_size_param=1. Received {batch_size_param}. Proceeding with 1.")
        
        tokens_list = [self.base_model.tokenizer.encode(prompt) for prompt in prompts]
        outputs = [self.base_model.tokenizer.padded_tokens_and_mask(
            tokens, self.base_options['text_ctx']
        ) for tokens in tokens_list]

        cond_tokens_list, cond_masks_list = zip(*outputs)
        cond_tokens_list, cond_masks_list = list(cond_tokens_list), list(cond_masks_list)

        uncond_tokens, uncond_mask = self.base_model.tokenizer.padded_tokens_and_mask(
            [], self.base_options['text_ctx']
        )

        model_kwargs = dict(
            tokens=th.tensor(
                cond_tokens_list + [uncond_tokens], device=self.device
            ),
            mask=th.tensor(
                cond_masks_list + [uncond_mask],
                dtype=th.bool,
                device=self.device,
            ),
        )
        
        num_cfg_inputs = len(prompts) + 1 
        wrapped_base_model_fn = partial(self._base_model_fn, weights_param=guidance_weights)

        self.base_model.del_cache()
        
        iter_count = 0
        final_sample_guided = None
        collected_attention_over_timesteps = [] # To store attention maps from relevant steps

        self.collected_attention_weights = []

        for out in self.base_diffusion.p_sample_loop_progressive(
            wrapped_base_model_fn,
            (num_cfg_inputs, 3, self.base_options["image_size"], self.base_options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=progress,
            model_kwargs=model_kwargs,
        ):
            final_sample_guided = out["sample"][:1] # We are interested in the guided sample

            if save_intermediate_steps and save_intermediate_steps > 0 and iter_count % save_intermediate_steps == 0:
                if self.collected_attention_weights:
                    timestep_tensor = out.get("t", None)
                    current_diffusion_step = None
                    if isinstance(timestep_tensor, th.Tensor) and timestep_tensor.numel() > 0:
                        current_diffusion_step = timestep_tensor[0].item()
                    else:
                        current_diffusion_step = iter_count
                    
                    collected_attention_over_timesteps.append({
                        'step': current_diffusion_step,
                        'maps': self.collected_attention_weights.copy()
                    })
                    # Reset for next iteration
                    self.collected_attention_weights = []
            iter_count += 1
        
        self.base_model.del_cache()
        return final_sample_guided, collected_attention_over_timesteps
    
    # def _upsample_internal(self, base_samples, prompts_for_upsampling, upsample_temp, progress=True, save_intermediate_steps: int = None, image_idx_start: int = 0, prompt_for_filename: str = "prompt"):
    #     if isinstance(prompts_for_upsampling, list):
    #         upsample_prompt_text = " ".join(prompts_for_upsampling)
    #     else:
    #         upsample_prompt_text = prompts_for_upsampling

    #     tokens = self.up_model.tokenizer.encode(upsample_prompt_text)
    #     tokens, mask = self.up_model.tokenizer.padded_tokens_and_mask(
    #         tokens, self.up_options['text_ctx']
    #     )

    #     num_samples_to_upsample = base_samples.shape[0]

    #     model_kwargs = dict(
    #         low_res=((base_samples + 1) * 127.5).round() / 127.5 - 1,
    #         tokens=th.tensor(
    #             [tokens] * num_samples_to_upsample, device=self.device
    #         ),
    #         mask=th.tensor(
    #             [mask] * num_samples_to_upsample,
    #             dtype=th.bool,
    #             device=self.device,
    #         ),
    #     )

    #     self.up_model.del_cache()
    #     up_shape = (num_samples_to_upsample, 3, self.up_options["image_size"], self.up_options["image_size"])
        
    #     iter_count = 0
    #     final_up_samples = None
    #     collected_attention_over_timesteps = [] # To store attention maps

    #     self.collected_attention_weights = []

    #     for out in self.up_diffusion.ddim_sample_loop_progressive(
    #         self.up_model,
    #         up_shape,
    #         noise=th.randn(up_shape, device=self.device) * upsample_temp,
    #         device=self.device,
    #         clip_denoised=True,
    #         progress=progress,
    #         model_kwargs=model_kwargs,
    #         eta=0.0 
    #     ):
    #         final_up_samples = out["sample"]
            
    #         if save_intermediate_steps and save_intermediate_steps > 0 and iter_count % save_intermediate_steps == 0:
    #             if self.collected_attention_weights:
    #                 timestep_tensor = out.get("t", None)
    #                 current_diffusion_step = None
    #                 if isinstance(timestep_tensor, th.Tensor) and timestep_tensor.numel() > 0:
    #                     current_diffusion_step = timestep_tensor[0].item()
    #                 else:
    #                     current_diffusion_step = iter_count
                    
    #                 collected_attention_over_timesteps.append({
    #                     'step': current_diffusion_step,
    #                     'maps': self.collected_attention_weights.copy()
    #                 })
    #                 # Reset for next iteration
    #                 self.collected_attention_weights = []
    #         iter_count += 1
            
    #     self.up_model.del_cache()
    #     return final_up_samples, collected_attention_over_timesteps

    def generate(self, prompt_text: str, num_images: int = 1, upsample: bool = False, 
             upsample_temp: float = 0.997, 
             save_intermediate_steps: int = 20, # Controls attention map collection frequency
             base_progress: bool = True, upsample_progress: bool = True,
             return_attention_maps: bool = False):
        processed_prompts, weights_values = prepare_prompt(prompt_text)
        
        if not processed_prompts:
            raise ValueError("Prompt text resulted in no processable prompts.")
        if len(processed_prompts) != len(weights_values) and self.verbose:
            print(f"Warning: Mismatch or default weights assigned. Prompts: {len(processed_prompts)}, Weights: {len(weights_values)}")

        if self.verbose:
            print(f"Using prompts: {processed_prompts} with weights: {weights_values}")
        
        safe_prompt_filename = "".join([c if c.isalnum() else "_" for c in prompt_text])[:50]
        if not safe_prompt_filename: safe_prompt_filename = "untitled"

        guidance_weights_tensor = th.tensor(weights_values).reshape(-1, 1, 1, 1).to(self.device)

        all_final_samples = []
        all_attention_data = []  # To store attention maps
        
        for i in range(num_images):
            if self.verbose: print(f"Generating base image {i+1}/{num_images}...")
            # _sample_base returns (final_sample_guided, collected_attention_over_timesteps)
            base_sample_single, base_collected_attentions = self._sample_base(
                processed_prompts, 
                batch_size_param=1, 
                guidance_weights=guidance_weights_tensor,
                progress=base_progress,
                save_intermediate_steps=save_intermediate_steps,
                image_idx_start=i,
                prompt_for_filename=safe_prompt_filename
            )
            
            current_final_sample = base_sample_single
            current_collected_attentions = base_collected_attentions
            current_stage_name = "base"

            if base_sample_single is None or base_sample_single.nelement() == 0:
                print(f"Warning: Base image {i+1} generation failed or returned None. Skipping.")
                continue 

            if upsample:
                if self.verbose: print(f"Upsampling base image {i+1}/{num_images}...")
                upsampled_sample_single, upsample_collected_attentions = self._upsample_internal(
                    base_sample_single.unsqueeze(0) if base_sample_single.ndim == 3 else base_sample_single, 
                    processed_prompts, 
                    upsample_temp,
                    progress=upsample_progress,
                    save_intermediate_steps=save_intermediate_steps,
                    image_idx_start=i,
                    prompt_for_filename=safe_prompt_filename
                )
                if upsampled_sample_single is not None and upsampled_sample_single.nelement() > 0:
                    current_final_sample = upsampled_sample_single.squeeze(0) if upsampled_sample_single.ndim == 4 and upsampled_sample_single.shape[0] == 1 else upsampled_sample_single
                    current_collected_attentions = upsample_collected_attentions
                    current_stage_name = "upsample"
                else:
                    print(f"Warning: Upsampling for image {i+1} failed. Using base image.")

            if current_final_sample is not None and current_final_sample.nelement() > 0:
                all_final_samples.append(current_final_sample)
                all_attention_data.append(current_collected_attentions)
            else:
                print(f"Warning: Final sample for image {i+1} is None or empty.")

        if not all_final_samples:
            print("No images were successfully generated.")
            return th.empty(0).to(self.device)
            
        if return_attention_maps:
            return th.stack(all_final_samples) if all_final_samples else th.empty(0).to(self.device), all_attention_data
        else:
            return th.stack(all_final_samples) if all_final_samples else th.empty(0).to(self.device)

    def __str__(self) -> str:
        base_params = sum(p.numel() for p in self.base_model.parameters()) if self.base_model else 0
        up_params = sum(p.numel() for p in self.up_model.parameters()) if self.up_model else 0

        table = []
        table.append(f"{'ComposeGLIDE Instance Configuration':^60}")
        table.append("=" * 60)
        table.append(f"{'Device:':<30} {str(self.device):<28}")
        table.append(f"{'Verbose:':<30} {str(self.verbose):<28}")
        table.append("-" * 60)
        
        table.append(f"{'Base Model':<60}")
        table.append(f"  {'Parameters:':<28} {f'{base_params:,}':<28}")
        table.append(f"  {'FP16 Enabled:':<28} {str(self.base_options.get('use_fp16', 'N/A')):<28}")
        table.append(f"  {'Timestep Respacing:':<28} {str(self.base_options.get('timestep_respacing', 'N/A')):<28}")
        table.append(f"  {'Image Size:':<28} {str(self.base_options.get('image_size', 'N/A')):<28}")
        table.append("-" * 60)

        table.append(f"{'Upsampler Model':<60}")
        table.append(f"  {'Parameters:':<28} {f'{up_params:,}':<28}")
        table.append(f"  {'FP16 Enabled:':<28} {str(self.up_options.get('use_fp16', 'N/A')):<28}")
        table.append(f"  {'Timestep Respacing:':<28} {str(self.up_options.get('timestep_respacing', 'N/A')):<28}")
        table.append(f"  {'Image Size:':<28} {str(self.up_options.get('image_size', 'N/A')):<28}")
        table.append("=" * 60)
        
        return "\n".join(table)
