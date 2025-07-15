import open_clip
import torch
import numpy as np
from PIL import Image


def pad_to_dim(tensor, target_size, dim=1):
    current_size = tensor.size(dim)
    pad_amount = target_size - current_size

    if pad_amount <= 0:
        return tensor

    pad_config = [0] * (2 * tensor.dim())
    pad_config[-(2 * dim + 1)] = pad_amount

    return torch.nn.functional.pad(tensor, pad_config)


def convert_img_for_clip(img_batch):
    imgs = []

    for i in range(img_batch.size(0)):
        im = img_batch[i].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        imgs.append(im)
    
    return imgs


class CLIPEmbeddingCalculator:

    def __init__(self, device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.model.visual.output_tokens = False
        self.reg_model, _, self.reg_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai'
        )
        self.reg_model.visual.output_tokens = True
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

    def get_word_embedding(self, text):
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float32):
            words = text.split()
            text_tokens = self.tokenizer(words).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.unsqueeze(0).permute(0, 2, 1)

    def get_text_embedding(self, text):
        text_tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float32):
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features


    def get_image_embedding(self, images):
        images = [self.preprocess(img) for img in images]
        image_vector = torch.stack(images, dim=0)

        # image_vector = self.preprocess(image_batch).unsqueeze(0)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float32):
            image_features = self.model.encode_image(image_vector)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features


    def get_image_region_embedding(self, images):
        self.reg_model.visual.output_tokens = True

        images = [self.reg_preprocess(img) for img in images]
        image = torch.stack(images, dim=0)

        # image = self.reg_preprocess(image).to(self.device)

        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float32):
            # Step 1: Extract patch embeddings (tokens)
            x = self.reg_model.visual.conv1(image)  # shape: [B, C, H/patch, W/patch]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, H*W]
            x = x.permute(0, 2, 1)  # [B, HW, C]

            # Step 2: Add class token and positional embeddings
            cls_token = self.reg_model.visual.class_embedding.to(x.dtype)
            cls_token = cls_token.expand(x.shape[0], 1, -1)
            x = torch.cat([cls_token, x], dim=1)  # [B, 1+HW, C]

            x = x + self.reg_model.visual.positional_embedding.to(x.dtype)[:x.shape[1]]
            x = self.reg_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # [T, B, C]

            # Step 3: Transformer
            x = self.reg_model.visual.transformer(x)  # [T, B, C]
            x = x.permute(1, 0, 2)  # [B, T, C]

            # Step 4: Remove CLS token, reshape to grid
            x = x[:, 1:, :]  # [B, HW, C]
            B, HW, C = x.shape
            S = int(HW**0.5)
            x = x.permute(0, 2, 1).reshape(B, C, S, S)  # [B, C, S, S]

        return x

    def image_encoders(self, image_tensor):
        rgb_images = convert_img_for_clip(image_tensor)

        image_vector = self.get_image_embedding(rgb_images)
        image_region_vectors = self.get_image_region_embedding(rgb_images)

        return image_vector, image_region_vectors
