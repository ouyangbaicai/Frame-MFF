import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from Utilities.CUDA_Check import GPUorCPU
from torchvision.io import read_image, ImageReadMode
# 若为灰度图像本文件需要更改两个地方：
# 1、注释掉归一化（class Dataloader_Eval）
# 2、评估数据集需要注释掉RGB格式的模式 （class Dataloader_Eval）

DEVICE = GPUorCPU().DEVICE
model_input_image_size_height = 256
model_input_image_size_width = 256
random_crop_size = 224

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class DataLoader_Train(Dataset):
    train_valid_transforms = transforms.Compose(
        [
            # transforms.CenterCrop(224),
            transforms.Resize((model_input_image_size_height, model_input_image_size_width), antialias=False),
            transforms.RandomCrop(random_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ZeroOneNormalize(),
        ]
    )

    train_valid_transforms_Norm = transforms.Compose(
        [
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    def __init__(self, file_list_A, file_list_B, file_list_GT, file_list_DM, file_list_res_DM, file_list_EDGE_A, file_list_EDGE_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.file_list_GT = file_list_GT
        self.file_list_DM = file_list_DM
        self.file_list_res_DM = file_list_res_DM
        self.file_list_EDGE_A = file_list_EDGE_A
        self.file_list_EDGE_B = file_list_EDGE_B
        self.transform1 = self.train_valid_transforms
        self.transform2 = self.train_valid_transforms_Norm

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B) == len(self.file_list_GT) == len(self.file_list_DM):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        seed = torch.random.seed()

        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_A = self.transform1(img_A)
        imgA_transformed = self.transform2(img_A)

        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_B = self.transform1(img_B)
        imgB_transformed = self.transform2(img_B)

        imgGT_path = self.file_list_GT[idx]
        img_GT = read_image(imgGT_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        imgGT_transformed = self.transform1(img_GT)

        imgDM_path = self.file_list_DM[idx]
        img_DM = read_image(imgDM_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgDM_transformed = self.transform1(img_DM)

        imgresDM_path = self.file_list_res_DM[idx]
        img_res_DM = read_image(imgresDM_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgresDM_transformed = self.transform1(img_res_DM)

        imgEDGE_path_A = self.file_list_EDGE_A[idx]
        img_EDGE_A = read_image(imgEDGE_path_A, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgEDGE_transformed_A = self.transform1(img_EDGE_A)

        imgEDGE_path_B = self.file_list_EDGE_B[idx]
        img_EDGE_B = read_image(imgEDGE_path_B, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgEDGE_transformed_B = self.transform1(img_EDGE_B)

        return imgA_transformed, imgB_transformed, imgGT_transformed, imgDM_transformed, imgresDM_transformed, imgEDGE_transformed_A, imgEDGE_transformed_B


class Dataloader_Eval(Dataset):
    eval_transforms = transforms.Compose(
        [
            # ZeroOneNormalize(),  # 修改地方1：若为3通道图去掉注释，若为灰度图需要加注释
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    def __init__(self, file_list_A, file_list_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.transform1 = self.eval_transforms
        self.transform2 = self.eval_transforms

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        imgA_path = self.file_list_A[idx]
        # 修改地方2：若为三通道使用下面两句话
        # img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        # imgA_transformed = self.transform1(img_A).to(DEVICE)
        # 修改地方2：若为灰度通道使用下面三句话
        img_A = Image.open(imgA_path).convert('L')
        to_tensor = transforms.ToTensor()
        img_A_tensor = to_tensor(img_A).to(DEVICE)
        imgA_transformed = img_A_tensor.repeat(3, 1, 1)


        imgB_path = self.file_list_B[idx]
        # 修改地方2：若为三通道使用下面两句话
        # img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        # imgB_transformed = self.transform1(img_B).to(DEVICE)
        # 修改地方2：若为灰度通道使用下面三句话
        img_B = Image.open(imgB_path).convert('L')
        to_tensor = transforms.ToTensor()
        img_B_tensor = to_tensor(img_B).to(DEVICE)
        imgB_transformed = img_B_tensor.repeat(3, 1, 1)


        return imgA_transformed, imgB_transformed
