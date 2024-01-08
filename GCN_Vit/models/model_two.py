import os
import torch
import torch.nn.functional as F
import utils.vision_transformer as vits
import utils.vision_transformer4k as vits4k
import utils.vision_transformerWSI_two as vitswsi
from torchvision import transforms
from einops import rearrange
from utils.GCN_utils import create_relation_matrix, get_edge_index
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, SAGPooling, BatchNorm


def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda')):
    r"""
    Builds ViT-256 Model.

    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.

    Returns:
    - model256 (torch.nn): Initialized model.
    """

    checkpoint_key = 'teacher'
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()
    model256.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    return model256


def get_vit4k(pretrained_weights, arch='vit4k_xs', device=torch.device('cuda')):
    r"""
    Builds ViT-4K Model.

    Args:
    - pretrained_weights (str): Path to ViT-4K Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.

    Returns:
    - model256 (torch.nn): Initialized model.
    """

    checkpoint_key = 'teacher'
    # device = torch.device("cpu")
    model4k = vits4k.__dict__[arch](num_classes=0)
    for p in model4k.parameters():
        p.requires_grad = False
    model4k.eval()
    model4k.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model4k.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    return model4k


class GCN_block(torch.nn.Module):
    def __init__(self, in_chan=384):
        super(GCN_block, self).__init__()
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 192, add_self_loops=False)
        self.lin = torch.nn.Linear(512, 256)
        self.lin2 = torch.nn.Linear(256, 1)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)
        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)

    def forward(self, data):
        # 原代码中创建data时自带data.batch，经修改发现不带data.batch，故自己创建
        data.batch = create_tensor_with_specified_number(data.x.shape[0], data.batch_size).cuda()
        # print(data.x.shape, data.edge_index.shape, data.batch, data.batch.shape)
        x = self.conv1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x = self.conv4(x, edge_index)

        x1 = global_mean_pool(x, batch=batch, size=data.batch.max() + 1)
        x2 = global_max_pool(x, batch=batch, size=data.batch.max() + 1)

        x = x1 + x2

        return x


def create_tensor_with_specified_number(length, number):
    assert length >= number, "Length should be greater than or equal to the specified number."

    # 创建一个长度为 length 的全零张量
    tensor = torch.zeros(length, dtype=torch.int64)

    # 计算每个数字的重复次数
    repeat = length // number

    # 填充指定数字
    for i in range(number):
        start = i * repeat
        end = (i + 1) * repeat
        tensor[start:end] = i

    return tensor


def prepare_img_tensor(img: torch.Tensor, patch_size=256):
    """
    Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
    are divisible by 256.

    (Note: "_256" for w / h is should technically be renamed as "_ps", but may not be easier to read.
    Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)

    Args:
        - img (torch.Tensor): [1 x C x W' x H'] image tensor.
        - patch_size (int): Desired patch size to evenly subdivide the image.

    Return:
        - img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
        - w_256 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
        - h_256 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
    """
    make_divisble = lambda l, patch_size: (l - (l % patch_size))
    b, c, w, h = img.shape
    load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
    w_256, h_256 = w // patch_size, h // patch_size
    img_new = transforms.CenterCrop(load_size)(img)
    return img_new, w_256, h_256


class vit_gcn(torch.nn.Module):
    def __init__(self,
                 in_chan=384,
                 model256_path: str = './Checkpoints/new_weights.pth',
                 model4k_path: str = './Checkpoints/vit4k_xs_dino.pth',
                 arch='vitWSI_xs'):
        super().__init__()
        self.model256 = get_vit256(pretrained_weights=model256_path)
        self.model4k = get_vit4k(pretrained_weights=model4k_path)
        self.gcn = GCN_block(in_chan=in_chan)
        self.modelwsi = vitswsi.__dict__[arch](num_classes=[1, 2])
        # self.norm = BatchNorm(384)

    def forward(self, data):
        x = data.img
        x = x.reshape(-1, 3, 4096, 4096)
        batch_256, w_256, h_256 = prepare_img_tensor(x)  # 1. [1 x 3 x W x H]
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)  # 2. [1 x 3 x w_256 x h_256 x 256 x 256]
        batch_256 = rearrange(batch_256,
                              'b c p1 p2 w h -> (b p1 p2) c w h')  # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)

        features_cls256 = []  # [256 x 384]
        for mini_bs in range(0, batch_256.shape[0],
                             256):  # 3. B may be too large for ViT-256. We further take minibatches of 256.
            minibatch_256 = batch_256[mini_bs:mini_bs + 256]
            features_cls256.append(self.model256(
                minibatch_256))  # 3. Extracting ViT-256 features from [256 x 3 x 256 x 256] image batches.

        features_cls256 = torch.vstack(features_cls256)  # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.
        # features_cls256 = self.norm(features_cls256)

        # vit-4k
        features_vit256 = features_cls256.reshape(-1, w_256, h_256, 384).transpose(1, 2).transpose(1, 3)
        features_vit4k = self.model4k.forward(features_vit256)  # 5. [B x 192], where 192 == dim of ViT-4K [ClS] token.

        # 计算邻接矩阵，GCN要用
        features_edge256 = create_relation_matrix(
            features_cls256.reshape(-1, w_256 * h_256, 384).detach().cpu().numpy())
        features_edge256 = get_edge_index(torch.from_numpy(features_edge256)).cuda()
        # 将特征和邻接矩阵加入data中，GCN要用
        data.x = features_cls256
        data.edge_index = torch.cat(torch.unbind(features_edge256, dim=0), dim=1)

        features_gcn4k = self.gcn(data)  # [B x 192]

        # vit-4k+gcn
        features_mix = features_vit4k + features_gcn4k  # [B x 192]
        features_mix = features_mix.unsqueeze(dim=1)  # [B x 1 x 192]

        # vit-wsi
        x1, x2 = self.modelwsi(features_mix)

        return x1, x2


class gcn(torch.nn.Module):
    def __init__(self,
                 in_chan=384,
                 model256_path: str = './Checkpoints/new_weights.pth',
                 arch='vitWSI_xs'):
        super().__init__()
        self.model256 = get_vit256(pretrained_weights=model256_path)
        self.gcn = GCN_block(in_chan=in_chan)
        self.modelwsi = vitswsi.__dict__[arch](num_classes=[1, 2])

    def forward(self, data):
        x = data.img
        x = x.reshape(-1, 3, 4096, 4096)
        batch_256, w_256, h_256 = prepare_img_tensor(x)  # 1. [1 x 3 x W x H]
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)  # 2. [1 x 3 x w_256 x h_256 x 256 x 256]
        batch_256 = rearrange(batch_256,
                              'b c p1 p2 w h -> (b p1 p2) c w h')  # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)

        features_cls256 = []
        for mini_bs in range(0, batch_256.shape[0],
                             256):  # 3. B may be too large for ViT-256. We further take minibatches of 256.
            minibatch_256 = batch_256[mini_bs:mini_bs + 256]
            features_cls256.append(self.model256(
                minibatch_256))  # 3. Extracting ViT-256 features from [256 x 3 x 256 x 256] image batches.

        features_cls256 = torch.vstack(features_cls256)  # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.

        # 计算邻接矩阵，GCN要用
        features_edge256 = create_relation_matrix(features_cls256.reshape(-1, w_256 * h_256, 384).detach().cpu().numpy())
        features_edge256 = get_edge_index(torch.from_numpy(features_edge256)).cuda()
        # 将特征和邻接矩阵加入data中，GCN要用
        data.x = features_cls256
        data.edge_index = torch.cat(torch.unbind(features_edge256, dim=0), dim=1)

        features_gcn4k = self.gcn(data)  # [B x 192]
        features_gcn4k = features_gcn4k.unsqueeze(dim=1)  # [B x 1 x 192]

        # vit-wsi
        x1, x2 = self.modelwsi(features_gcn4k)

        return x1, x2


class vit(torch.nn.Module):
    def __init__(self,
                 in_chan=384,
                 model256_path: str = './Checkpoints/new_weights.pth',
                 model4k_path: str = './Checkpoints/vit4k_xs_dino.pth',
                 arch='vitWSI_xs'):
        super().__init__()
        self.model256 = get_vit256(pretrained_weights=model256_path)
        self.model4k = get_vit4k(pretrained_weights=model4k_path)
        self.modelwsi = vitswsi.__dict__[arch](num_classes=[1, 2])

    def forward(self, data):
        x = data.img
        x = x.reshape(-1, 3, 4096, 4096)
        batch_256, w_256, h_256 = prepare_img_tensor(x)  # 1. [1 x 3 x W x H]
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)  # 2. [1 x 3 x w_256 x h_256 x 256 x 256]
        batch_256 = rearrange(batch_256,
                              'b c p1 p2 w h -> (b p1 p2) c w h')  # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)

        features_cls256 = []
        for mini_bs in range(0, batch_256.shape[0],
                             256):  # 3. B may be too large for ViT-256. We further take minibatches of 256.
            minibatch_256 = batch_256[mini_bs:mini_bs + 256]
            features_cls256.append(self.model256(
                minibatch_256))  # 3. Extracting ViT-256 features from [256 x 3 x 256 x 256] image batches.

        features_cls256 = torch.vstack(features_cls256)  # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.

        features_vit256 = features_cls256.reshape(-1, w_256, h_256, 384).transpose(1, 2).transpose(1, 3)
        features_vit4k = self.model4k.forward(features_vit256)  # 5. [B x 192], where 192 == dim of ViT-4K [ClS] token.

        features_vit4k = features_vit4k.unsqueeze(dim=1)  # [B x 1 x 192]

        # vit-wsi
        x1, x2 = self.modelwsi(features_vit4k)

        return x1, x2


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 3, 4096, 4096)).cuda()
    model = vit_gcn().to(device)
    # print(model(x))
    a, b = model(x)
    print(a)

    print(b)
    print(b.shape)
    print(torch.argmax(b))
