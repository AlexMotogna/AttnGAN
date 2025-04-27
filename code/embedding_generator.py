import torch
import torch.nn as nn
import open_clip


# ('ViT-L-14', 'openai')
class embedding_generator(nn.Module):
    def __init__(self):
        super(embedding_generator, self).__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

    def forward(self, x, is_text=True):

        if is_text:
            text = self.tokenizer(x)
            with torch.no_grad(), torch.autocast("cuda"):
                text_features = self.model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            return text_features
        else:
            image = self.preprocess(x).unsqueeze(0)

            with torch.no_grad(), torch.autocast("cuda"):
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features


