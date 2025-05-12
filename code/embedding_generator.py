import torch
import torch.nn as nn
import open_clip


# ('ViT-L-14', 'openai')
# class embedding_generator(nn.Module):
#     def __init__(self):
#         super(embedding_generator, self).__init__()

#         self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
#         self.model.eval()
#         self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

#     def forward(self, x, is_text=True):

#         if is_text:
#             text = self.tokenizer(x)
#             with torch.no_grad(), torch.autocast("cuda"):
#                 text_features = self.model.encode_text(text)
#                 text_features /= text_features.norm(dim=-1, keepdim=True)

#             return text_features
#         else:
#             image = self.preprocess(x).unsqueeze(0)

#             with torch.no_grad(), torch.autocast("cuda"):
#                 image_features = self.model.encode_image(image)
#                 image_features /= image_features.norm(dim=-1, keepdim=True)
            
#             return image_features


def pad_to_dim(tensor, target_size, dim=1):
    current_size = tensor.size(dim)
    pad_amount = target_size - current_size

    if pad_amount <= 0:
        return tensor

    pad_config = [0] * (2 * tensor.dim())
    pad_config[-(2 * dim + 1)] = pad_amount

    return torch.nn.functional.pad(tensor, pad_config)


class EmbeddingGenerator():

    def __init__(self):
        super(EmbeddingGenerator, self).__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.model.visual.output_tokens = True
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

    def get_word_embedding(self, text, cap_lens):
        word_embs = []
        max_cap_len = torch.max(cap_lens)

        with torch.no_grad(), torch.autocast("cuda"):
            for sent in text:
                words = sent.split()
                text_tokens = self.tokenizer(words)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                word_emb = pad_to_dim(text_features, max_cap_len, dim=0)
                word_embs.append(word_emb)
            
            result = torch.stack(word_embs, dim=0).permute(0, 2, 1)
        
        return result

    def get_text_embedding(self, text):
        text_tokens = self.tokenizer(text)
        with torch.no_grad(), torch.autocast("cuda"):
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_embedding(self, images):
        if isinstance(images, list):
            images = torch.stack([self.preprocess(img) for img in images])

        with torch.no_grad():
            patch_tokens = self.model.visual(images)
        
        print(patch_tokens.shape)

        global_feat = patch_tokens[:, 0]              # CLS token: [B, 768]
        patch_tokens = patch_tokens[:, 1:]            # [B, num_patches, 768]
        region_feats = patch_tokens.transpose(1, 2)  # [B, 768, num_patches]

        print(global_feat.shape, region_feats.shape)
        return global_feat, region_feats
