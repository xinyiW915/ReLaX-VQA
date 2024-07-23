import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from torchvision import transforms
import torch

from main_fragment_pool import get_patch_diff, extract_important_patches
from extractor import visualise_vgg_layer, visualise_resnet_layer, visualise_vit_layer

def map_attention_to_original(original_frame, attention_map, positions, patch_size):
    full_attention = np.zeros_like(original_frame[:,:,0], dtype=float)  # 确保注意力图层是二维的

    for (pos, att) in zip(positions, attention_map):
        y, x = pos
        start_y = y * patch_size
        start_x = x * patch_size
        full_attention[start_y:start_y+patch_size, start_x:start_x+patch_size] = att

    full_attention = (full_attention / np.max(full_attention)) * 255
    full_attention = full_attention.astype(np.uint8)
    heatmap = cv2.applyColorMap(full_attention, cv2.COLORMAP_JET)
    result = cv2.addWeighted(original_frame, 0.6, heatmap, 0.4, 0)
    return result

def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img

def plot_diff(frame_diff):
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_diff, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Patch Differences')
    plt.axis('off')
    plt.show()
    plt.close()

def get_activation_png(attention, residual_name, patch_size=16, img_dim=224):
    n_heads = attention.shape[0]
    # # attention maps
    # for i in range(n_heads):
    #     plt.imshow(attention[i], cmap='viridis') #cmap='viridis', cmap='inferno'
    #     plt.title(f"Head n: {i + 1}")
    #     plt.axis('off')  # Turn off axis ticks and labels
    #     plt.show()
    #     plt.close()

    patch_per_dim = img_dim // patch_size
    head_mean = np.mean(attention, axis=0).reshape((patch_per_dim, patch_size, patch_per_dim, patch_size))
    patch_means = head_mean.mean(axis=(1, 3)).reshape((patch_per_dim, patch_per_dim))
    plt.figure(figsize=(10, 10))
    plt.imshow(patch_means, cmap='viridis')
    plt.title(f"{residual_name} - Head Mean Attention")
    plt.axis('off')
    plt.show()
    plt.close()
    return patch_means

def process_frame_with_attention(imp_path, positions, residual_name):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    name_model = 'vit_base'
    patch_size = 16

    # resize image residual imp and get ViT attention feature
    residual_imp = Image.open(imp_path)
    if residual_imp.size != (224, 224):
        residual_imp = residual_imp.resize((224, 224), Image.Resampling.LANCZOS)
    img_size = residual_imp.size[::-1]
    img_pre = transform(residual_imp, img_size)
    model = visualise_vit_layer.VitGenerator(name_model, patch_size, device, evaluate=True, random=False, verbose=True)
    attentions = visualise_vit_layer.visualize_attention(model, img_pre, patch_size, device)
    patch_means = get_activation_png(attentions, residual_name)

    # map attentions to original frame
    mapped_frame = map_attention_to_original(original_frame, patch_means.flatten(), positions, patch_size)
    plt.imshow(cv2.cvtColor(mapped_frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Frame with Mapped Attention: {residual_name}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    video_name = '5636101558'  # 'TelevisionClip_1080P-68c6', 'Sports_2160P-0455', '5636101558'
    frame_number = 4
    patch_size = 16
    target_size = 224
    top_n = 196
    colormap = 'viridis'  # hot, viridis, plasma, inferno

    original_frame_path = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}.png'
    # next_frame = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}_next.png'
    original_frame = cv2.imread(original_frame_path)

    # get patch position for frame diff
    residual_name = 'Residual frame'
    residual_path = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}_residual.png'
    residual_frame = cv2.imread(residual_path)
    frame_diff = get_patch_diff(residual_frame, patch_size)
    _, positions_frame_diff = extract_important_patches(residual_frame, frame_diff, patch_size, target_size, top_n)
    plot_diff(frame_diff)
    residual_imp_path = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}_residual_imp.png'
    process_frame_with_attention(residual_imp_path, positions_frame_diff, residual_name)

    # get patch position for optical flow
    residual_name = 'Optical flow'
    residual_of_path = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}_residual_of.png'
    residual_of = cv2.imread(residual_of_path)
    of_diff = get_patch_diff(residual_of, patch_size)
    _, positions_of_diff = extract_important_patches(residual_of, of_diff, patch_size, target_size, top_n)
    residual_of_imp_path = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}_residual_of_imp.png'
    process_frame_with_attention(residual_of_imp_path, positions_of_diff, residual_name)

    # get patch position for origianl fragment
    fragment_name = 'Origianl fragment'
    ori_frag_path = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}_ori_frag.png'
    process_frame_with_attention(ori_frag_path, positions_frame_diff, fragment_name)

    # get patch position for residual merged fragment
    fragment_name = 'Merged fragment'
    merged_frag_path = f'../visualisation/visualisation_example/original_{video_name}/{video_name}_{frame_number}_residual_merged_frag.png'
    process_frame_with_attention(merged_frag_path, positions_frame_diff, fragment_name)



