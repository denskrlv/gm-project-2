import kagglehub
from src.compose_glide import ComposeGlide


compose_glide = ComposeGlide(verbose=True)
print(compose_glide)


path = kagglehub.dataset_download("kushsheth/face-vae")
print("Path to dataset files:", path)
