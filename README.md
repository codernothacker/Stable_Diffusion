# stable_diffusion
# Stable Diffusion Implementation in PyTorch

This project implements Stable Diffusion from scratch using PyTorch, based on the architecture and techniques outlined in the original Stable Diffusion paper. It uses pre-trained weights and tokenizer from the Hugging Face model hub.

## Features

- Text-to-Image generation
- Implementation of U-Net architecture
- Latent Diffusion Model approach

## Installation

1. Clone the repository:
```bash
git clone https://github.com/codernothacker/Stable_Diffusion.git
cd STABLE_DIFFUSION
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model weights and tokenizer files:
   - Download `vocab.json` and `merges.txt` from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer) and save them in the `data` folder
   - Download `v1-5-pruned-emaonly.ckpt` from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and save it in the `data` folder

## Project Structure
```
STABLE_DIFFUSION/
├── data/
│   ├── merges.txt
│   ├── v1-5-pruned-emaonly.ckpt
│   └── vocab.json
├── images/
│   └── dog.jpg
├── sd/
│   ├── attention.py
│   ├── clip.py
│   ├── decoder.py
│   ├── diffusion.py
│   ├── encoder.py
│   └── pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```
## Usage

```python
from stable_diffusion import StableDiffusion

model = StableDiffusion()

# Text to Image
image = model.text_to_image("A dog wearing glasses")

# Image to Image
transformed_image = model.image_to_image(source_image, "A dog running in the park")

# Inpainting
inpainted_image = model.inpaint(image, mask, "A dog with a hat")
```

## Technical Details

This implementation includes:
- Latent Diffusion Model for efficient computation
- CLIP for text encoding
- U-Net architecture for the denoising process
- Classifier-Free Guidance for improved generation
- Various normalization techniques (Layer Normalization, Group Normalization)

## Acknowledgments

Special thanks to:
1. [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion/)
2. [Stable Diffusion TensorFlow](https://github.com/divamgupta/stable-diffusion-tensorflow)
3. [Stable Diffusion PyTorch](https://github.com/kjsman/stable-diffusion-pytorch)
4. [Hugging Face Diffusers](https://github.com/huggingface/diffusers/)
5. [Umar Jamil](https://www.youtube.com/watch?v=ZBKpAp_6TGI)

