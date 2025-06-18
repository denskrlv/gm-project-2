import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebA_Dataset(Dataset):
    def __init__(self, root_dir, attr_file, image_size=256, transform=None, debug=False):
        """
        FFHQ dataset loader that properly handles CSV attribute files
        
        Args:
            root_dir: Root directory containing the images
            attr_file: Path to the CSV attributes file
            image_size: Size to resize images to
            transform: Additional transforms to apply
            debug: Print debug information
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.debug = debug
        
        # Read the CSV file properly
        print(f"Loading attributes from {attr_file}")
        self.attr_df = pd.read_csv(attr_file)
        
        # Print column information
        if debug:
            print(f"CSV columns: {self.attr_df.columns.tolist()}")
            print(f"First row: {self.attr_df.iloc[0].to_dict()}")
        
        # Find the image filename column
        if 'image_id' in self.attr_df.columns:
            self.filename_col = 'image_id'
        else:
            # Try to find the image column
            for col in self.attr_df.columns:
                if 'image' in col.lower() or 'file' in col.lower() or 'id' in col.lower():
                    self.filename_col = col
                    break
            else:
                # Assume first column contains filenames
                self.filename_col = self.attr_df.columns[0]
                
        print(f"Using '{self.filename_col}' as image filename column")
        
        # Map standard attributes to possible column names
        attr_mappings = {
            'Male': ['Male', 'male', 'Gender', 'gender', 'sex', 'Sex'],
            'Wearing_Hat': ['Wearing_Hat', 'hat', 'Hat', 'Wearing_Hat'],
            'Smiling': ['Smiling', 'smiling', 'Smile', 'smile'],
            'Eyeglasses': ['Eyeglasses', 'eyeglasses', 'Glasses', 'glasses']
        }
        
        # Find matching columns
        self.attr_mapping = {}
        for attr, possible_names in attr_mappings.items():
            for name in possible_names:
                if name in self.attr_df.columns:
                    self.attr_mapping[attr] = name
                    print(f"Found '{attr}' attribute as column '{name}'")
                    break
        
        # Verify we found at least some attributes
        self.selected_attrs = list(self.attr_mapping.keys())
        if not self.selected_attrs:
            print("WARNING: No matching attributes found. Available columns:")
            print(self.attr_df.columns.tolist())
            
        # Verify root directory exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
            
        # Verify some image files exist
        test_img = self.attr_df.iloc[0][self.filename_col]
        possible_paths = [
            os.path.join(root_dir, test_img),
            os.path.join(root_dir, os.path.basename(test_img))
        ]
        
        found_path = False
        for path in possible_paths:
            if os.path.exists(path):
                found_path = True
                print(f"Images confirmed at: {path}")
                break
                
        if not found_path:
            print(f"WARNING: Could not find test image. Tried paths:")
            for path in possible_paths:
                print(f"- {path}")
                
        print(f"Loaded FFHQ dataset with {len(self.attr_df)} images and {len(self.selected_attrs)} selected attributes")
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.attr_df)
    
    def __getitem__(self, idx):
        # Get image filename and STRIP WHITESPACE
        filename = self.attr_df.iloc[idx][self.filename_col].strip()
        
        # Construct the full path to the image
        img_path = os.path.join(self.root_dir, filename)
        
        # Check if the image exists
        if not os.path.exists(img_path):
            # This error message is now much more reliable
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
        
        # Get attributes and create a prompt
        attrs = {}
        for attr, col_name in self.attr_mapping.items():
            attrs[attr] = int(self.attr_df.iloc[idx][col_name])
        
        prompt = self.create_prompt(attrs)
        
        return image, prompt
    
    def create_prompt(self, attrs):
        """Create a text prompt based on image attributes"""
        if 'Male' in attrs:
            if attrs['Male'] == 1:
                prompt = "a photo of a man"
            else:
                prompt = "a photo of a woman"
        else:
            prompt = "a photo of a person"
        
        # Add hat information
        if 'Wearing_Hat' in attrs and attrs['Wearing_Hat'] == 1:
            prompt += " wearing a hat"
            
        # Add smile information
        if 'Smiling' in attrs and attrs['Smiling'] == 1:
            prompt += " smiling"
            
        # Add glasses information
        if 'Eyeglasses' in attrs and attrs['Eyeglasses'] == 1:
            prompt += " wearing glasses"
            
        return prompt

def get_ffhq_dataloader(root_dir, attr_file, batch_size=4, num_workers=0, debug=False):
    dataset = CelebA_Dataset(root_dir, attr_file, debug=debug)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    return dataloader
