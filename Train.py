import os
import sys
import glob
import torch
from Loss_funcs import loss_f
import datetime
from torch import nn
from tqdm import tqdm
from Nets.Network import Network
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
from Utilities.Logging_SaveModel import Logging_SaveModel
from Loss_funcs.SSIM import *
import torchvision.models as models
from torchvision.models import ResNet101_Weights
DEVICE = GPUorCPU.DEVICE


class NetTrain:
    def __init__(self,
                 data_path='C:\\Users\\HP\\Desktop\\ouyangbaicai\\MSI-DTrans\\Datasets\\Train&Valid\\MFFdatasets',
                 set_size=500,
                 batchsize=1,
                 epochs=200,
                 lr=0.0002,
                 gamma=0.88,
                 scheduler_step=1,
                 lmd=0.,
                 patience=6):
        # Hyper parameters
        self.DEVICE = GPUorCPU().DEVICE
        self.DATAPATH = data_path
        self.SETSIEZE = set_size
        self.BATCHSIZE = batchsize
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.LMD = lmd
        self.PATIENCE = patience
        # Form parameter dictionary
        self.hyperparas = {'set_size': self.SETSIEZE, 'batchsize': self.BATCHSIZE,
                           'epochs': self.EPOCHS, 'lr': self.LR, 'gamma': self.GAMMA,
                           'scheduler_step': self.SCHEDULER_STEP, 'lmd': self.LMD,
                           'patience': self.PATIENCE}

    def __call__(self, *args, **kwargs):
        TRAIN_LOADER, VALID_LOADER = self.PrepareDataLoader(self.DATAPATH, self.SETSIEZE, self.BATCHSIZE)
        MODEL, OPTIMIZER, SCHEDULER = self.BuildModel(self.DEVICE, self.LR, self.SCHEDULER_STEP, self.GAMMA)
        self.TrainingProcess(MODEL, OPTIMIZER, SCHEDULER, TRAIN_LOADER, VALID_LOADER, self.EPOCHS, self.LMD)

    def PrepareDataLoader(self, datapath, setsize, batchsize):
        train_list_A = sorted(glob.glob(os.path.join(datapath, 'train/sourceA', '*.*')))[:setsize]
        train_list_B = sorted(glob.glob(os.path.join(datapath, 'train/sourceB', '*.*')))[:setsize]
        train_list_GT = sorted(glob.glob(os.path.join(datapath, 'train/groundtruth', '*.*')))[:setsize]
        train_list_DM = sorted(glob.glob(os.path.join(datapath, 'train/decisionmap', '*.*')))[:setsize]
        train_list_res_DM = sorted(glob.glob(os.path.join(datapath, 'train/res_decisionmap', '*.*')))[:setsize]
        valid_list_A = sorted(glob.glob(os.path.join(datapath, 'validate/sourceA', '*.*')))[:setsize // 9]
        valid_list_B = sorted(glob.glob(os.path.join(datapath, 'validate/sourceB', '*.*')))[:setsize // 9]
        valid_list_GT = sorted(glob.glob(os.path.join(datapath, 'validate/groundtruth', '*.*')))[:setsize // 9]
        valid_list_DM = sorted(glob.glob(os.path.join(datapath, 'validate/decisionmap', '*.*')))[:setsize // 9]
        valid_list_res_DM = sorted(glob.glob(os.path.join(datapath, 'validate/res_decisionmap', '*.*')))[:setsize // 9]
        tqdm.write(f"Train Data A: {len(train_list_A)}")
        tqdm.write(f"Train Data B: {len(train_list_B)}")
        tqdm.write(f"Train Data GT: {len(train_list_GT)}\n")
        tqdm.write(f"Valid Data A: {len(valid_list_A)}")
        tqdm.write(f"Valid Data B: {len(valid_list_B)}")
        tqdm.write(f"Valid Data GT: {len(valid_list_GT)}\n")
        train_data = DLr.DataLoader_Train(train_list_A, train_list_B, train_list_GT, train_list_DM, train_list_res_DM)
        valid_data = DLr.DataLoader_Train(valid_list_A, valid_list_B, valid_list_GT, valid_list_DM, valid_list_res_DM)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        tqdm.write(f"Train Data Size:{len(train_data)} , Train Loader Amount: {len(train_data)}/{batchsize} = {len(train_loader)}")
        tqdm.write(f"Valid Data Size:{len(valid_data)} , Valid Loader Amount: {len(valid_data)}/{batchsize} = {len(valid_loader)}\n")
        return train_loader, valid_loader

    def BuildModel(self, device, lr, scheduler_step, gamma):
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        model = Network(resnet).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=gamma)
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model, optimizer, scheduler

    def MixLoss(self, DM, res_DM, pre_DM):
        loss_l1 = nn.L1Loss()
        GT_DM = torch.cat([res_DM.repeat(6, 1, 1, 1), DM.repeat(6, 1, 1, 1)], dim=0)
        return 0.1*loss_l1(pre_DM, GT_DM) + 0.9*loss_l1(pre_DM[-1].unsqueeze(0), DM)

    def LossFuncF(self, pre_FUSED, GT):
        LossFunc = loss_f.fusion_loss_mff()
        loss = LossFunc(pre_FUSED, GT)
        return loss

    def TrainingProcess(self, model, optimizer, scheduler, train_loader, valid_loader, epochs, lmd):
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        LS = Logging_SaveModel(savepath='RunTimeData', hyperparas=self.hyperparas)

        tqdm.write('Training start...\n')

        for epoch in range(epochs):
            ######################################### Train #########################################
            x = 0
            epoch_loss = 0
            epoch_accuracy = 0
            train_loader_tqdm = tqdm(train_loader, colour='green', leave=False, file=sys.stdout)
            for A, B, GT, DM, res_DM in train_loader_tqdm:
                x+=1
                optimizer.zero_grad()
                # Automatic mixed precision training.
                with torch.autocast(device_type=self.DEVICE, dtype=torch.float16):
                    DM_img, Fused_img = model(A, B, 0)
                    loss = 0.9 * self.MixLoss(DM, res_DM, DM_img) + 0.1 * self.LossFuncF(Fused_img, GT)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ''''''''''''''''''''''''''' Epoch loss calculation & progress visualization '''''''''''''''''''''''''''
                epoch_loss += loss / len(train_loader)
                epoch_accuracy += (1 - loss) / len(train_loader)
                train_loader_tqdm.set_description("[%s] Epoch %s" % (str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')), str(epoch + 1)))
                train_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################
            ######################################### Valid #########################################
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                valid_loader_tqdm = tqdm(valid_loader, colour='yellow', leave=False, file=sys.stdout)
                for A, B, GT, DM, res_DM in valid_loader_tqdm:
                    DM_img, Fused_img = model(A, B, 0)
                    loss = 0.9 * self.MixLoss(DM, res_DM, DM_img) + 0.1 * self.LossFuncF(Fused_img, GT)
                    # loss = loss_g_con(pre, A, B)
                    ''''''''''''''''''''' Epoch loss calculation & progress visualization '''''''''''''''''''''
                    epoch_val_accuracy += (1 - loss) / len(valid_loader)
                    epoch_val_loss += loss / len(valid_loader)
                    valid_loader_tqdm.set_description("[Validating...] Epoch %s" % str(epoch + 1))
                    valid_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################
            # Print epoch loss and accuracy.
            tqdm.write(f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S'))}] Epoch {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}")
            # Dynamic learning rate.
            scheduler.step()
            # Logging and Save model weights.
            log_contents = f"Epoch {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            LS(model, epoch + 1, log_contents, epoch_val_loss, save_every_model=True)
            if LS.ENDTRAIN:
                # Early stopping mechanism has been triggered.
                print("Early stopping!!!")
                # End training.
                break
        # Epoch loop ends.


if __name__ == '__main__':
    t = NetTrain()
    t()
