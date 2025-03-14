import warnings
warnings.filterwarnings("ignore")
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from thop import profile
is_flop_cal = False

# get the activation
def get_activation(model, layer, input_img_data):
    model.eval()
    activations = []
    inputs = []

    def hook(module, input, output):
        activations.append(output)
        inputs.append(input[0])

    hook_handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        model(input_img_data)
    hook_handle.remove()
    return activations, inputs

def get_activation_map(frame, layer_name, resnet50, device):
    # image pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations (resize and normalize)
    frame_tensor = transform(frame)

    # adding index 0 changes the original [C, H, W] shape to [1, C, H, W]
    if frame_tensor.dim() == 3:
        frame_tensor = frame_tensor.unsqueeze(0)
    # print(f'Image dimension: {frame_tensor.shape}')

    # getting the activation of a given layer
    layer_obj = eval(layer_name)
    activations, inputs = get_activation(resnet50, layer_obj, frame_tensor)
    activated_img = activations[0][0]
    activation_array = activated_img.cpu().numpy()

    # calculate FLOPs for layer
    if is_flop_cal == True:
        flops, params = profile(layer_obj, inputs=(inputs[0],), verbose=False)
        if params == 0 and isinstance(layer_obj, torch.nn.Conv2d):
            params = layer_obj.in_channels * layer_obj.out_channels * layer_obj.kernel_size[0] * layer_obj.kernel_size[1]
            if layer_obj.bias is not None:
                params += layer_obj.out_channels
        # print(f"FLOPs for {layer_name}: {flops}, Params: {params}")
    else:
        flops, params = None, None
    return activated_img, activation_array, flops, params

def process_video_frame(video_name, frame, frame_number, all_layers, resnet50, device):
    # create a dictionary to store activation arrays for each layer
    activations_dict = {}
    total_flops = 0
    total_params = 0
    for layer_name in all_layers:
        fig_name = f"resnet50_feature_map_layer_{layer_name}"
        combined_name = f"resnet50_feature_map"

        activated_img, activation_array, flops, params = get_activation_map(frame, layer_name, resnet50, device)
        if is_flop_cal == True:
            total_flops += flops
            total_params += params

        # save activation maps as png
        # png_path = f'../visualisation/resnet50/{video_name}/frame_{frame_number}/'
        # npy_path = f'../features/resnet50/{video_name}/frame_{frame_number}/'
        # os.makedirs(png_path, exist_ok=True)
        # os.makedirs(npy_path, exist_ok=True)
        # get_activation_png(png_path, fig_name, activated_img)
        # save activation features as npy
        # get_activation_npy(npy_path, fig_name, activation_array)
        # save to the dictionary
        activations_dict[layer_name] = activated_img

    # print(f"total FLOPs for Resnet50 layerstack: {total_flops}, Params: {total_params}")
    frame_npy_path = f'../features/resnet50/{video_name}/frame_{frame_number}_{combined_name}.npy'
    return activations_dict, frame_npy_path, total_flops, total_params

def get_activation_png(png_path, fig_name, activated_img, n=8):
    fig = plt.figure(figsize=(10, 10))

    # visualise activation map for 64 channels
    for i in range(n):
        for j in range(n):
            idx = (n * i) + j
            if idx >= activated_img.shape[0]:
                break
            ax = fig.add_subplot(n, n, idx + 1)
            ax.imshow(activated_img[idx].cpu().numpy(), cmap='viridis')
            ax.axis('off')

    # save figures
    fig_path = f'{png_path}{fig_name}.png'
    print(fig_path)
    print("----------------" + '\n')
    plt.savefig(fig_path)
    plt.close()

def get_activation_npy(npy_path, fig_name, activation_array):
    np.save(f'{npy_path}{fig_name}.npy', activation_array)

if __name__ == '__main__':
    device_name = "gpu"
    if device_name == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
    # pre-trained ResNet-50 model to device
    resnet50 = models.resnet50(pretrained=True).to(device)

    all_layers = ['resnet50.conv1',
                  'resnet50.layer1[0]', 'resnet50.layer1[1]', 'resnet50.layer1[2]',
                  'resnet50.layer2[0]', 'resnet50.layer2[1]', 'resnet50.layer2[2]', 'resnet50.layer2[3]',
                  'resnet50.layer3[0]', 'resnet50.layer3[1]', 'resnet50.layer3[2]', 'resnet50.layer3[3]',
                  'resnet50.layer4[0]', 'resnet50.layer4[1]', 'resnet50.layer4[2]']

    video_type = 'test'
    # Test
    if video_type == 'test':
        metadata_path = "../../metadata/test_videos.csv"
    # NR:
    elif video_type == 'resolution_ugc':
        resolution = '360P'
        metadata_path = f"../../metadata/YOUTUBE_UGC_{resolution}_metadata.csv"
    else:
        metadata_path = f'../../metadata/{video_type.upper()}_metadata.csv'

    ugcdata = pd.read_csv(metadata_path)
    for i in range(len(ugcdata)):
        video_name = ugcdata['vid'][i]
        sampled_frame_path = os.path.join('../..', 'video_sampled_frame', 'sampled_frame', f'{video_name}')

        print(f"Processing video: {video_name}")
        image_paths = glob.glob(os.path.join(sampled_frame_path, f'{video_name}_*.png'))
        frame_number = 0
        for image in image_paths:
            print(f"{image}")
            frame_number += 1
            process_video_frame(video_name, image, frame_number, all_layers, resnet50, device)

# # ResNet-50 layers to visualize
# layers_to_visualize_resnet50 = {
#     'conv1': 0,
#     'layer1.0.conv1': 2,
#     'layer1.0.conv2': 3,
#     'layer1.1.conv1': 5,
#     'layer1.1.conv2': 6,
#     'layer1.2.conv1': 8,
#     'layer1.2.conv2': 9,
#     'layer2.0.conv1': 11,
#     'layer2.0.conv2': 12,
#     'layer2.1.conv1': 14,
#     'layer2.1.conv2': 15,
#     'layer2.2.conv1': 17,
#     'layer2.2.conv2': 18,
#     'layer2.3.conv1': 20,
#     'layer2.3.conv2': 21,
#     'layer3.0.conv1': 23,
#     'layer3.0.conv2': 24,
#     'layer3.0.downsample.0': 25,
#     'layer3.1.conv1': 27,
#     'layer3.1.conv2': 28,
#     'layer3.2.conv1': 30,
#     'layer3.2.conv2': 31,
#     'layer3.3.conv1': 33,
#     'layer3.3.conv2': 34,
#     'layer4.0.conv1': 36,
#     'layer4.0.conv2': 37,
#     'layer4.0.downsample.0': 38,
#     'layer4.1.conv1': 40,
#     'layer4.1.conv2': 41,
#     'layer4.2.conv1': 43,
#     'layer4.2.conv2': 44,
# }
