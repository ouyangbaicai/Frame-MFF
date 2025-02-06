import os
import sys
import glob
import time
import cv2
import torch
from tqdm import tqdm
from torch import einsum
from Nets.Network import Network
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
import torchvision.models as models
from torchvision.models import ResNet101_Weights
DEVICE = GPUorCPU.DEVICE

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class Fusion:
    def __init__(self,
                 modelpath='RunTimeData/2025-01-23 08.07.39/model19.ckpt',
                 dataroot='C:\\Users\\HP\\Desktop\\ouyangbaicai\\MSI-DTrans\\Datasets\\Eval',
                 dataset_name='Lytro',
                 threshold=0.005,
                 window_size=5,
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME != None:
            self.SAVEPATH = '/' + self.DATASET_NAME
            self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("Test Dataset required!")
            pass

    def LoadWeights(self, modelpath):
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        model = Network(resnet).to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        # Verified_img_tensor = Consistency.Binarization(img_tensor)
        # if threshold != 0:
        Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results/' + savepath):
            os.makedirs('./Results/' + savepath, exist_ok=True)
        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()
                D, Fused = model(A, B, 1)
                D = torch.where(D[-1].unsqueeze(0) > 0.5, 1., 0.)
                D = self.ConsisVerif(D, threshold)
                D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()
                Fused = einsum('c w h -> w h c', Fused[0]).clone().detach().cpu().numpy()
                A = cv2.imread(eval_list_A[cnt - 1])
                B = cv2.imread(eval_list_B[cnt - 1])
                IniF = A * D + B * (1 - D)
                D = D * 255
                cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '-dm.png', D)
                cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '.png', IniF)
                cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '-fused.png', Fused*255)
                cnt += 1
                running_time.append(time.time() - start_time)
        running_time_total = 0
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        print("\navg_process_time: {} s".format(running_time_total / (len(running_time) - 1)))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()