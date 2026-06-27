import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_size = (512, 512)

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

def load_image(path):
    x = Image.open(path)
    x = transform(x)
    y = x.unsqueeze(0)
    y = y.to(device)
    return y

def deprocess(tensor):
    tensor = tensor.squeeze(0)
    std1 = torch.tensor(std).reshape(3, 1, 1).to(device)
    mean1 = torch.tensor(mean).reshape(3, 1, 1).to(device)
    tensor = tensor * std1 + mean1
    tensor = tensor.clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    return img

vgg = models.vgg19(pretrained = True).features.to(device)

for p in vgg.parameters():
    p.requires_grad = False

vgg = vgg.eval()
def feature_maps(tensor, vgg):
    i = 0
    j = 0
    features = {}
    for layer in vgg:
        tensor = layer(tensor)
        if i == 0 or i == 5 or i == 10 or i == 19:
            j+=1
            features[f"conv{j}_1"] = tensor
        elif i == 21:
            features["conv4_2"] = tensor
        elif i == 28:
            features["conv5_1"] = tensor
            break
        else:
            pass
        i+=1

    return features

def gram_matrix(tensor):
    tensor = tensor.squeeze(0)
    N, H, W = tensor.shape
    t1 = tensor.reshape(N, H*W)
    t2 = t1 @ t1.T
    return t2

def content_loss_fn(F_x, F_c):
    content_loss = 0.5 * nn.MSELoss(reduction = 'sum')(F_x, F_c)
    return content_loss

def style_loss_fn(features_x, features_s):
    style_loss = 0
    for layer in features_x:
        if layer == 'conv4_2':
            continue
        gram_x = gram_matrix(features_x[layer])
        gram_s = gram_matrix(features_s[layer])
        _, N, H, W = features_x[layer].shape
        M = H*W
        style_loss += 0.20 * 1/(4*(N**2)*(M**2)) * nn.MSELoss(reduction = 'sum')(gram_x, gram_s)

    return style_loss

def total_loss_fn(content, style, alpha, beta):
    return alpha*content + beta*style

def run_nst(content_path, style_path, alpha, beta, steps):
    content = load_image(content_path)
    style = load_image(style_path)
    content_features = feature_maps(content, vgg)
    style_features = feature_maps(style, vgg)
    x = content.clone().detach()
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=0.01)
    
    for i in range(steps):
        optimizer.zero_grad()
        F_c = content_features['conv4_2']
        feature_x = feature_maps(x, vgg)
        content_loss = content_loss_fn(feature_x['conv4_2'], F_c)
        style_loss = style_loss_fn(feature_x, style_features)
        total_loss = total_loss_fn(content_loss, style_loss, alpha, beta)
        total_loss.backward()
        optimizer.step()
        x.data.clamp_(0,1)
    
    return deprocess(x)  