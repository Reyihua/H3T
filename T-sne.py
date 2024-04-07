import typing
import io
import os
import argparse

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms
from models.model_tsne import VisionTransformer, CONFIGS, AdversarialNetwork
from data.data_list_image import Normalize

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
from utils.transform import get_transform
from data.data_list_image import ImageList
def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def postprocess_activations(activations):
    output = activations
    output *= 255
    return 255 - output.astype('uint8')

def visual(feat):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


maker = ['o','v']
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink', 'red', 'tomato', 'ivory', 'green', 'bisque', 'tan', 'fuchsia', 'navy', 'magenta', 'slateblue', 'plum', 'lightgreen', 'slategray', 'lightpink', 'deeppink', 'crimson', 'indigo', 'khaki', 'gold' ]
Label_Com = ['a', 'b', 'c', 'd']
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }
color = ['red', 'blue']


def plotlabels(S_lowDWeights, Trure_labels, name, s2, t2):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)

    for index in range(31):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=20, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=0.7)
   
    t2 = t2.reshape((-1, 1))
    S2 = np.hstack((s2, t2))
    S2 = pd.DataFrame({'x': S2[:, 0], 'y': S2[:, 1], 'label': S2[:, 2]})

    for index in range(31):
        X = S2.loc[S2['label'] == index]['x']
        Y = S2.loc[S2['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=20, marker=maker[1], c=colors[index], edgecolors=colors[index], alpha=0.7)

        plt.xticks([])
        plt.yticks([])

    #plt.title(fontsize=15, fontweight='normal', pad=20, y=-0.2)


def visualize(args):
    print("a")
    # Prepare Model
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=args.num_classes, 
                                zero_head=False, img_size=args.img_size, vis=True)
    
    model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    model.load_state_dict(torch.load(model_checkpoint))
    
    model.eval()
    
    ad_net = AdversarialNetwork(config.hidden_size//12, config.hidden_size//12)
    ad_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
    ad_net.load_state_dict(torch.load(ad_checkpoint))
    
    ad_net.eval()
    
    ad_net2 = AdversarialNetwork(config.hidden_size//3, config.hidden_size//3)
    ad_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv_2.bin" % args.name)
    ad_net2.load_state_dict(torch.load(ad_checkpoint))
    
    ad_net2.eval()
    print("b")
    transform_source, transform_target, transform_test = get_transform("office", 256)
    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.image_path).readlines(), transform=transform_source, mode='RGB'),
        batch_size=16, shuffle=True, num_workers=4)
    target_loader = torch.utils.data.DataLoader(
        ImageList(open(args.image_path2).readlines(), transform=transform_target, mode='RGB'),
        batch_size=16, shuffle=True, num_workers=4)
    len_source = len(source_loader)
    len_target = len(target_loader) 
    print(len_source)
    model.cuda()
    ad_net.cuda()
    ad_net2.cuda()
    source = 0
    target = 0
    s_index=0
    t_index=0
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    for global_step in range(1, 1000):
        print(global_step)
        
        if (global_step-1) % (len_source-1) != 0 or global_step==1:
            data_source = iter_source.next()
            x_s, y_s = tuple(t.cuda() for t in data_source)
            _, _, feat_s = model(x_s, ad_net=ad_net, ad_net2=ad_net2)
            feats = feat_s.cpu()
            feats = feats.detach().numpy()
            y_s = y_s.cpu()
            y_s = y_s.detach().numpy()
            if s_index==0:
                feature_s = feats
                index_s = y_s
                s_index = 1
            else:
                feature_s = np.concatenate((feature_s, feats),axis=0)
                index_s = np.concatenate((index_s, y_s),axis=0)
        else:
            break
    
    for global_step in range(1, 1000):
        print(global_step)
        
        if (global_step-1) % (len_target-1) != 0 or global_step==1:
            data_target = iter_target.next()
            x_t, y_t = tuple(t.cuda() for t in data_target)
            _, _, feat_t = model(x_t, ad_net=ad_net, ad_net2=ad_net2)
            featt = feat_t.cpu()
            featt = featt.detach().numpy()
            y_t = y_t.cpu()
            y_t = y_t.detach().numpy()
            if t_index==0:
                feature_t = featt
                index_t = y_t
                t_index = 1
            else:
                feature_t = np.concatenate((feature_t, featt),axis=0)
                index_t = np.concatenate((index_t, y_t),axis=0)
        else:
            break
        
    print(len(feature_s))
    print(len(feature_t))
    
    fig = plt.figure(figsize=(7, 5))
    feature = np.concatenate((feature_s, feature_t),axis=0)
    feature = np.split(visual(feature), (2816,))
    print(len(feature))
    
    plotlabels(feature[0], index_s, '(b) HPTrans', feature[1], index_t)
    plt.savefig('97.pdf',bbox_inches='tight', pad_inches = -0.1)
    plt.axis('off')
    plt.show()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="svhn2mnist",
                        help="Which downstream task.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--num_classes", default=10, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--image_path", help="Path of the test image.")
    parser.add_argument("--image_path2", help="Path of the test image.")
    parser.add_argument("--output_dir", default="output5", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--save_dir", default="attention_visual", type=str,
                        help="The directory where attention maps will be saved.")
    args = parser.parse_args()
    visualize(args)

if __name__ == "__main__":
    main()
        
        
