import os, json, cv2, torch
from torch.utils.data import Dataset
import numpy as np
from .transforms import get_transform

class SurgBloodDataset(Dataset):
    def __init__(self, cfg, split='train'):
        root = cfg.data.root
        self.img_size = cfg.data.img_size
        self.T = cfg.data.window_size
        self.split = split
        split_file = os.path.join(root, cfg.data.train_split if split=='train' else cfg.data.val_split)
        with open(split_file) as f:
            self.clips = [x.strip() for x in f if x.strip()]
        self.root = root
        self.samples = self._build_samples()
        
        # 初始化数据变换
        self.transform = get_transform(cfg, is_train=(split == 'train'))

    def _build_samples(self):
        items = []
        for clip in self.clips:
            fdir = os.path.join(self.root, 'frames', clip)
            frames = sorted([x for x in os.listdir(fdir) if x.endswith('.jpg')])
            # 只从 i >= T-1 开始构建样本，确保所有样本都是完整的T帧窗口
            for i in range(self.T - 1, len(frames)):
                start = i - (self.T - 1)
                items.append((clip, frames[start:i+1]))
        return items

    def _read_img(self, path):
        
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32)/255.0
        return torch.from_numpy(im).permute(2,0,1)

    def _read_mask_point(self, clip, frame):
        ap = os.path.join(self.root, 'annotations', clip)
        mpath = os.path.join(ap, frame.replace('.jpg','_mask.png'))
        ppath = os.path.join(ap, frame.replace('.jpg','_point.json'))
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        mask = (cv2.resize(mask, (self.img_size,self.img_size))>127).astype(np.float32)[None]
        with open(ppath) as f: j = json.load(f)
        exists = torch.tensor([j.get("exists",0)], dtype=torch.float32)
        if j.get("exists",0)==1:
            # 获取原始图像尺寸，如果没有则使用默认值 1280x720
            original_size = j.get("image_size", [1280, 720])
            if isinstance(original_size, list) and len(original_size) == 2:
                orig_w, orig_h = original_size
            else:
                # 如果格式不对，使用默认值
                orig_w, orig_h = 1280, 720
                print(f"Warning: Invalid image_size format in {ppath}, using default 1280x720")
            
            sx, sy = self.img_size/orig_w, self.img_size/orig_h
            x = j["x"]*sx; y = j["y"]*sy
        else:
            x=y=0.0
        point = torch.tensor([x,y], dtype=torch.float32)
        return torch.from_numpy(mask), point, exists

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        clip, frames = self.samples[idx]
        # 所有样本都已经是完整的T帧窗口，无需填充
        assert len(frames) == self.T, f"Expected {self.T} frames, got {len(frames)}"
        ims = [ self._read_img(os.path.join(self.root,'frames',clip,f)) for f in frames ]
        frames_t = torch.stack(ims, dim=0)  # [T,3,H,W]
        # 监督取最后一帧
        gt_mask, gt_point, gt_exists = self._read_mask_point(clip, frames[-1])
        
        # 应用数据变换
        frames_t, gt_mask, gt_point, gt_exists = self.transform(frames_t, gt_mask, gt_point, gt_exists)
        
        # 返回字典格式（便于损失函数使用）
        return {
            'frames': frames_t,
            'mask': gt_mask,
            'point_coords': gt_point,
            'point_exists': gt_exists,
            'clip_name': clip,
            'frame_name': frames[-1]
        }
