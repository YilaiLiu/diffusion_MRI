import os
import numpy as np
import pandas as pd
import albumentations
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset

resize_ifneeded=transforms.Lambdad(
    keys=["t1", "t1ce", "t2", "flair"],
    func=lambda x: transforms.Resize((160,160,128))(x) if np.any(np.array(x.shape[1:]) > np.array((160,160,128))) else x
)
brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),#设置为None时依赖meta_dict去获取channel信息
        transforms.Lambdad(keys=["t1", "t1ce", "t2", "flair"], func=lambda x: x[0, :, :, :]),#选取第一个通道，感觉不太有必要
        transforms.AddChanneld(keys=["t1", "t1ce", "t2", "flair"]),#转为1,H,W,D
        transforms.EnsureTyped(keys=["t1", "t1ce", "t2", "flair"]),#转化为tensor
        transforms.Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAI", allow_missing_keys=True),#根据meta数据orientation调整图像方向
        transforms.CropForegroundd(keys=["t1", "t1ce", "t2", "flair"], source_key="t1", allow_missing_keys=True),#自动识别背景裁剪，可以通过margin参数调整边缘的像素值
        #这里加一个
        
        transforms.SpatialPadd(keys=["t1", "t1ce", "t2", "flair"], spatial_size=(160, 160, 128), allow_missing_keys=True),
        transforms.RandSpatialCropd( keys=["t1", "t1ce", "t2", "flair"],
            roi_size=(160, 160, 128),
            random_center=True, 
            random_size=False,
        ),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t1ce", "t2", "flair"], lower=0, upper=99.75, b_min=0, b_max=1),
    ]
)

def get_brats_dataset(data_path):
    transform = brats_transforms
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz") 
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz") 
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz") 
        flair = os.path.join(sub_path, f"{subject}_flair.nii.gz") 
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")

        data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair, "subject_id": subject})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)



class CustomBase(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data = get_brats_dataset(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)