import os
import glob
import math
from functools import partial
import torch

import ipywidgets as widgets
import io
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn

import utils.logger_setup as logger_setup
# import utils.logger_setup_vit as logger_setup
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

# Step 2: Creating a Vision Transformer
# normalise the torch
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

#用于执行无梯度截断正态分布初始化。这两个函数在模型初始化中使用，确保权重被适当地初始化。
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

#对输入进行随机丢弃一部分元素，实现随机深度（Stochastic Depth）。
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#用于在残差块的主路径上应用 drop_path 函数。
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#一个多层感知机（MLP）类，包含两个线性层和一个激活函数，用于在残差块中对特征进行非线性映射。
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 自注意力机制类，用于在残差块中计算注意力权重并应用它们。
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

# 一个残差块类，包含一个自注意力模块和一个MLP模块。
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 图像到块嵌入类，将输入图像分割成块并将它们映射到嵌入空间
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# Vision Transformer模型的主要实现。包含多个残差块、嵌入层等。（还需要学里面每一步代码具体在做什么）
class VisionTransformer(nn.Module):
    """
    Vision Transformer
    """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size

        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]  # return CLS token and attention_features maps

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                # print(f"return attention of the last block: {x.shape}")
                # print(blk(x, return_attention=True).shape)
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

# Vision Transformer 模型的生成器类，用于实例化和配置特定模型。
class VitGenerator(object):
    def __init__(self, name_model, patch_size, device, evaluate=True, random=False, verbose=False):
        self.name_model = name_model
        self.patch_size = patch_size
        self.evaluate = evaluate
        self.device = device
        self.verbose = verbose
        self.model = self._getModel()
        self._initializeModel()
        if not random:
            self._loadPretrainedWeights()

    def _getModel(self):
        if self.verbose:
            logger_setup.logger.debug(f"[INFO] Initializing {self.name_model} with patch size of {self.patch_size}")
            # print((f"[INFO] Initializing {self.name_model} with patch size of {self.patch_size}"))
        if self.name_model == 'vit_tiny':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        elif self.name_model == 'vit_small':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        elif self.name_model == 'vit_base':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        else:
            raise f"No model found with {self.name_model}"

        return model

    def _initializeModel(self):
        if self.evaluate:
            for p in self.model.parameters():
                p.requires_grad = False

            self.model.eval()

        self.model.to(self.device)

    def _loadPretrainedWeights(self):
        if self.verbose:
            logger_setup.logger.debug("[INFO] Loading weights")
            # print(("[INFO] Loading weights"))
        url = None
        if self.name_model == 'vit_small' and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"

        elif self.name_model == 'vit_small' and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

        elif self.name_model == 'vit_base' and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"

        elif self.name_model == 'vit_base' and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        if url is None:
            logger_setup.logger.debug(f"Since no pretrained weights have been found with name {self.name_model} and patch size {self.patch_size}, random weights will be used")
            # print((f"Since no pretrained weights have been found with name {self.name_model} and patch size {self.patch_size}, random weights will be used"))

        else:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)
        logger_setup.logger.debug(url)
        # print(url)

    def get_last_selfattention(self, img):
        return self.model.get_last_selfattention(img.to(self.device))

    def __call__(self, x):
        return self.model(x)

# Step 3: Creating Visualization Functions
def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img

def visualize_predict(model, img, img_size, patch_size, device, png_path, npy_path, video_name, frame_number, qp, fig_name, combined_name):
    img_pre = transform(img, img_size)
    attention = visualize_attention(model, img_pre, patch_size, device)
    # save activation maps as png
    # get_activation_png(img, png_path, fig_name, attention)
    # save activation features as npy
    activations_dict, frame_npy_path = get_activation_npy(npy_path, video_name, frame_number, qp, fig_name, combined_name, attention)
    return activations_dict, frame_npy_path

def visualize_attention(model, img, patch_size, device):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions

def get_activation_png(img, png_path, fig_name, attention):
    n_heads = attention.shape[0]

    # attention maps
    for i in range(n_heads):
        plt.imshow(attention[i], cmap='viridis') #cmap='viridis', cmap='inferno'
        plt.title(f"Head n: {i + 1}")
        plt.axis('off')  # Turn off axis ticks and labels

        # Save figures
        fig_path = f'{png_path}{fig_name}_head_{i + 1}.png'
        print(fig_path)
        plt.savefig(fig_path)
        plt.close()

    # head mean map
    plt.figure(figsize=(10, 10))
    image_name = fig_name.replace('vit_feature_map_', '')
    text = [f"{image_name}", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='viridis')
        plt.title(text[i])
        plt.axis('off')  # Turn off axis ticks and labels
    fig_path1 = f'{png_path}{fig_name}_head_mean.png'
    print(fig_path1)
    print("----------------" + '\n')
    plt.savefig(fig_path1)
    plt.close()

    # combine
    # plt.figure(figsize=(20, 20))
    # for i in range(n_heads):
    #     plt.subplot(n_heads//3, 3, i+1)
    #     plt.imshow(attention[i], cmap='inferno')
    #     plt.title(f"Head n: {i+1}")
    # plt.tight_layout()
    # fig_path2 = png_path + fig_name + '_heads.png'
    # print(fig_path2 + '\n')
    # plt.savefig(fig_path2)
    # plt.close()

def get_activation_npy(npy_path, video_name, frame_number, qp, fig_name, combined_name, attention):
    mean_attention = attention.mean(axis=0)
    logger_setup.logger.debug(f'Attention array shape: {mean_attention.shape}')
    frame_npy_path = f'../features/vit/{video_name}/frame_{frame_number}_{combined_name}.npy'

    return mean_attention, frame_npy_path


class Loader(object):
    def __init__(self):
        self.uploader = widgets.FileUpload(accept='image/*', multiple=False)
        self._start()

    def _start(self):
        display(self.uploader)

    def getLastImage(self):
        try:
            for uploaded_filename in self.uploader.value:
                uploaded_filename = uploaded_filename
            img = Image.open(io.BytesIO(
                bytes(self.uploader.value[uploaded_filename]['content'])))

            return img
        except:
            return None

    def saveImage(self, path):
        with open(path, 'wb') as output_file:
            for uploaded_filename in self.uploader.value:
                content = self.uploader.value[uploaded_filename]['content']
                output_file.write(content)

def process_video_frame(image_path, video_name, qp, model, patch_size, device):
    filename = os.path.basename(image_path)
    if 'residual' in filename or 'next' in filename or 'ori' in filename:
        parts = filename.split('_')
        if 'residual_of' in filename:
            frame_number = parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
            if 'residual_of_imp' in filename:
                frame_number = parts[-4] + "_" + parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        elif 'residual_imp' in filename:
            frame_number = parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        elif '_residual_merged_frag' in filename:
            frame_number = parts[-4] + "_" + parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        elif 'ori_frag' in filename:
            frame_number = parts[-3] + "_" + parts[-2] + "_" + parts[-1].split('.')[0]
        else:
            frame_number = parts[-2] + "_" + parts[-1].split('.')[0]
    else:
        frame_number = int(filename.split('_')[-1].split('.')[0])

    img = Image.open(image_path)
    # resize image
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_size = img.size[::-1]
    # resize the video by 2
    # factor_reduce = 2
    # img_size = tuple(np.array(img.size[::-1]) // factor_reduce)

    # visualise output folder
    png_path = f'../visualisation/vit/{video_name}/frame_{frame_number}/'
    npy_path = f'../features/vit/{video_name}/frame_{frame_number}/'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(npy_path, exist_ok=True)

    if qp == "original":
        fig_name = f"vit_feature_map_original"
        combined_name = f"vit_feature_map_original"
    else:
        fig_name = f"vit_feature_map_qp_{qp}"
        combined_name = f"vit_feature_map_qp_{qp}"

    # activations_dict, frame_npy_path = visualize_predict(model, img, img_size, patch_size, device, png_path, npy_path, video_name, frame_number, qp, fig_name, combined_name)
    attention_features, frame_feature_npy_path = extract_features(model, img, img_size, patch_size, device, video_name, frame_number, combined_name)
    return attention_features, frame_feature_npy_path

def extract_features(model, img, img_size, patch_size, device, video_name, frame_number, combined_name):
    img = transform(img, img_size)
    img = img.unsqueeze(0).to(device)
    cls_token, attention_features = model(img)

    attention_features = attention_features.squeeze(0).cpu().numpy()
    logger_setup.logger.debug(f'Attention array shape: {attention_features.shape}')
    frame_feature_npy_path = f'../features/vit/{video_name}/frame_attention_{frame_number}_{combined_name}.npy'
    return attention_features, frame_feature_npy_path

if __name__ == '__main__':
    # Step 4: Visualizing Images
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    name_model = 'vit_base'
    patch_size = 16

    model = VitGenerator(name_model, patch_size,
                         device, evaluate=True, random=False, verbose=True)

    image_type = 'resolution_ugc' #original_ugc, encoded_ugc

    # for original video:
    if image_type == 'original_ugc':
        metadata_path = "../../metadata/YOUTUBE_UGC_metadata_original.csv"
        ugcdata = pd.read_csv(metadata_path)

    # for encoded video:
    elif image_type == 'encoded_ugc':
        codec_name = 'x264'
        metadata_path = f"../../metadata/YOUTUBE_UGC_metadata_{codec_name}_metrics.csv"
        ugcdata = pd.read_csv(metadata_path)

    elif image_type == 'test':
        metadata_path = f"../../metadata/test_metadata.csv"
        ugcdata = pd.read_csv(metadata_path)

    # for no reference video:
    elif image_type == 'resolution_ugc':
        resolution = '2160P'
        metadata_path = f"../../metadata/YOUTUBE_UGC_{resolution}_metadata.csv"
        ugcdata = pd.read_csv(metadata_path)

    else:
        raise ValueError(f"Unsupported image_type: {image_type}")

    for i in range(len(ugcdata)):
        video_name = ugcdata['vid'][i]

        if image_type == 'original_ugc':
            qp = 'original'
            sampled_frame_path = os.path.join('../..', 'video_sampled_frame', 'original_sampled_frame', f'{video_name}')

        elif image_type == 'encoded_ugc':
            qp = ugcdata['QP'][i]
            sampled_frame_path = os.path.join('../..', 'video_sampled_frame', f'encoded_sampled_frame_qp_{qp}', f'{video_name}')

        elif image_type == 'test':
            qp = ugcdata['QP'][i]
            sampled_frame_path = os.path.join('../..', 'video_sampled_frame', f'encoded_sampled_frame_qp_{qp}', f'{video_name}')

        elif image_type == 'resolution_ugc':
            qp = 'original'
            sampled_frame_path = os.path.join('../..', 'video_sampled_frame', f'original_sampled_frame_{resolution}', f'video_{i+1}')

        print(f"Processing video: {video_name} at QP {qp}")
        image_paths = glob.glob(os.path.join(sampled_frame_path, f'{video_name}_*.png'))
        for image_path in image_paths:
            print(f"{image_path}")
            process_video_frame(image_path, video_name, qp, model, patch_size, device)
