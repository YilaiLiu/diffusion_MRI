import os
import numpy as np
import pandas as pd
import albumentations
from torch.utils.data import Dataset,ConcatDataset

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
        transforms.SpatialPadd(keys=["t1", "t1ce", "t2", "flair"], spatial_size=(160, 160, 128), allow_missing_keys=True),
        transforms.RandSpatialCropd( keys=["t1", "t1ce", "t2", "flair"],
            roi_size=(160, 160, 128),
            random_center=True, 
            random_size=False,
        ),#如果大于目标分辨率就进行中心裁剪
        transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t1ce", "t2", "flair"], lower=0, upper=99.75, b_min=0, b_max=1),
    ]#99.75而不是100避免极端的高强度噪音影响了图像
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



###mode:Neon-n,teenager-t,young Adult-a,middle-aged-m,elder-e


my_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1", "t2"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1", "t2"], allow_missing_keys=True),
        transforms.Lambdad(keys=["t1", "t2"], func=lambda x: x[0, :, :, :]),
        transforms.AddChanneld(keys=["t1", "t2"]),
        transforms.EnsureTyped(keys=["t1", "t2"]),
        transforms.Orientationd(keys=["t1", "t2"], axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys=["t1", "t2"], source_key="t1", allow_missing_keys=True),
        transforms.SpatialPadd(keys=["t1", "t2"], spatial_size=(160, 160, 128), allow_missing_keys=True),
        transforms.RandSpatialCropd( keys=["t1", "t2"],
            roi_size=(160, 160, 128),
            random_center=True, 
            random_size=False,
        ),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t2"], lower=0, upper=99.75, b_min=0, b_max=1),
    ]
)

def get_my_dataset(data_path):
    transform = my_transforms 
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz") 
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz") 


        data.append({"t1":t1, "t2":t2, "subject_id": subject})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)




###mode:Neon-n,teenager-t,young Adult-a,middle-aged-m,elder-e


class MyBase(Dataset):
    def __init__(self,data_paths,data_modes):
        super().__init__()
        paths = data_paths.split(',') if ',' in data_paths else [data_paths]
        modes = data_modes.split(',') if ',' in data_modes else [data_modes]

        # 确保 paths 和 modes 长度一致
        if len(paths) != len(modes):
            raise ValueError("the lengths of data_paths and data_modes are not the same")
        combined_dataset=[]
        for data_path,data_mode in zip(paths,modes):
            dataset = get_my_dataset(data_path)
            for item in dataset.data:
                item["mode"] = data_mode
            combined_dataset.append(dataset)
        self.data=ConcatDataset(combined_dataset)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class MyTrain(MyBase):
    def __init__(self, data_paths,data_modes, **kwargs):
        super().__init__(data_paths=data_paths,data_modes=data_modes)


class MyTest(MyBase):
    def __init__(self, data_paths,data_modes, **kwargs):
        super().__init__(data_paths=data_paths,data_modes=data_modes)