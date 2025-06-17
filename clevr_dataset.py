import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CLEVR_Dataset(Dataset):
    def __init__(self, root_dir, scenes_json, image_size=256, transform=None, debug=False):
        """
        CLEVR dataset loader that handles scene descriptions
        
        Args:
            root_dir: Root directory containing the images and scene files
            scenes_json: Path to the scenes JSON file with object descriptions
            image_size: Size to resize images to
            transform: Additional transforms to apply
            debug: Print debug information
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.image_size = image_size
        self.debug = debug
        
        # Load scene descriptions
        print(f"Loading scene descriptions from {scenes_json}")
        with open(scenes_json, 'r') as f:
            self.scenes = json.load(f)['scenes']
        
        if debug:
            print(f"Loaded {len(self.scenes)} scenes")
            print(f"Sample scene: {self.scenes[0]}")
            
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])
        
        print(f"Loaded CLEVR dataset with {len(self.scenes)} images")
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        # Get image filename from scene data
        scene = self.scenes[idx]
        image_filename = scene['image_filename']
        
        # Construct the full path to the image
        img_path = os.path.join(self.image_dir, image_filename)
        
        # Check if the image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        
        # Load and process the image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.base_transform(image)
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {str(e)}")
        
        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Create a descriptive prompt for the scene
        prompt = self.create_prompt(scene)
        
        return image, prompt
    
    def create_prompt(self, scene):
        """Create a text prompt based on scene objects"""
        objects = scene['objects']
        
        if not objects:
            return "an empty scene with a gray surface"
        
        # Start with a base description
        prompt = "a scene with "
        
        # Describe up to 3 objects to keep prompts manageable
        obj_descriptions = []
        for obj in objects[:3]:
            color = obj['color']
            size = obj['size']
            material = obj['material']
            shape = obj['shape']
            
            obj_desc = f"a {size} {color} {material} {shape}"
            obj_descriptions.append(obj_desc)
        
        if len(objects) > 3:
            prompt += ", ".join(obj_descriptions) + f" and {len(objects) - 3} other objects"
        else:
            if len(obj_descriptions) == 1:
                prompt += obj_descriptions[0]
            else:
                prompt += " and ".join([", ".join(obj_descriptions[:-1]), obj_descriptions[-1]])
        
        return prompt

def get_clevr_dataloader(root_dir, scenes_json, batch_size=4, num_workers=0, debug=False):
    """Create a DataLoader for the CLEVR dataset"""
    dataset = CLEVR_Dataset(root_dir, scenes_json, debug=debug)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    return dataloader
