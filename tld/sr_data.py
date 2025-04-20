import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel

from TitokTokenizer.modeling.titok import TiTok
from tld.diffusion import encode_text
from tld.sr_train import SR_COCODataset
from tld.configs import ModelConfig, DataConfig, TrainConfig
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

#titok = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet").to('cuda')

print("Loaded Titok")

data_config = DataConfig(
        latent_path="latents.npy", text_emb_path="text_emb.npy", val_path="val_emb.npy"
    )

config = ModelConfig(
        data_config=data_config,
        train_config=TrainConfig(),
    )

train_config = config.train_config
dataconfig = config.data_config

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
vae = vae.to('cuda')

if not os.path.exists(config.latents_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    
    train_dataset = SR_COCODataset(img_dir=dataconfig.img_path,
                        ann_file=dataconfig.img_ann_path)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)

    '''
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
    clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loaded clip_model")

    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to('cuda')

    print("Loaded DINO")
    
    x_list, y_list, z_list = [], [], []
    '''
    x_list = []

    print('processing data!')
    
    for x, y, z in tqdm(train_loader):
        '''
        x = x.to('cuda') 
        z = z.to('cuda') 

        z -= torch.min(z)
        z /= torch.max(z)
        z = dino_processor(images=z, return_tensors="pt").to('cuda')
    
        z = dino_model(**z)
        z = z[0]

        cls = z[:, 0]
        max_pooled = torch.max(z[:, 1:], dim = 1)[0]
        pooled = torch.cat([cls, max_pooled], dim=1)

        if config.use_titok:
            with torch.no_grad():
                x, _ = titok.encode(x)
            x = x.squeeze(2)

        y = clip_preprocess(text = y, return_tensors = "pt", padding = True).to('cuda')
        y = encode_text(y, clip_model)

        x_list.append(x.detach().cpu().numpy().astype(np.float32))
        y_list.append(y.detach().cpu().numpy().astype(np.float32))
        z_list.append(pooled.detach().cpu().numpy().astype(np.float32))
        '''
        x = x.to('cuda')
        x -= torch.min(x)
        x /= torch.max(x)
        x = x * 2 - 1
        x = vae.encode(x, return_dict=False)[0].sample()
        x_list.append(x.detach().cpu().numpy().astype(np.float32))

    #x_val, y_val, z_val = x_list[0], y_list[0], z_list[0]
    x_val = x_list[0]

    x_all = np.concatenate(x_list[1:], axis=0)
    #y_all = np.concatenate(y_list[1:], axis=0)
    #z_all = np.concatenate(z_list[1:], axis=0)

    np.savez("preprocess_vae.npz", x_all = x_all, x_val = x_val)
    #np.savez(dataconfig.latent_path, x_all = x_all, x_val = x_val)
    #np.savez(dataconfig.text_emb_path, y_all = y_all, y_val = y_val)
    #np.savez(dataconfig.lr_latent_path, z_all = z_all, z_val = z_val)
