import torch
import numpy as np
import slideio
import pandas as pd
from torch_geometric.data import Dataset, Data


class MyDataset(Dataset):
    def __init__(self, path: str, bag_size=4096):
        super(MyDataset, self).__init__()
        t = pd.read_csv(path, encoding="gbk")
        self.bag_size = bag_size
        self.x = list(t['x'])
        self.bag_id = list(t['bag_id'])
        self.y = list(t['y'])
        self.wsi_path = list(t['wsi'])
        self.li_wsi_name = [int(x.split('_')[0]) for x in list(t['ID'])]
        self.status = list(t['rec'])
        self.survival_time = list(t['PFS'])
        # self.label = list(t[['替代型', '纤维型', '膨胀型']].values)
        self.label = list(t[['纤维型', '膨胀型']].values)

        self.num_bag = len(self.wsi_path)

    def get(self, index):
        slide = slideio.open_slide(self.wsi_path[index], "SVS")
        scene = slide.get_scene(0)
        img = scene.read_block((int(self.x[index]), int(self.y[index]), self.bag_size, self.bag_size),
                               (self.bag_size, self.bag_size))
        img = np.transpose(img, (2, 0, 1))
        # 将numpy数组转换为torch张量
        img = torch.from_numpy(img).float()

        # li_wsi_name = [int(x.split('_')[0]) for x in self.li_wsi_name]
        li_wsi_name = torch.tensor(self.li_wsi_name[index])

        status = torch.tensor(self.status[index])
        survival_time = torch.tensor(self.survival_time[index])
        label = torch.tensor(self.label[index]).float()

        data = Data(img=img, id=li_wsi_name, status=status, time=survival_time, label=label.unsqueeze(0))
        return data

    def len(self):
        return len(self.x)