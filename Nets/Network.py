import torch
import math
import random
import torch.fft
from einops import rearrange
from thop import profile, clever_format
from torch import nn, einsum
from torch.nn import functional as F
import torchvision.models as models
from torchvision.models import ResNet101_Weights
from Utilities.CUDA_Check import GPUorCPU
DEVICE = GPUorCPU.DEVICE


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)

    def forward(self, input_, hiddenState, cellState):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided

        prev_hidden = hiddenState
        prev_cell = cellState

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.relu(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.relu(cell)

        return hidden, cell


class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        out = self.sigmoid(e_t) * x
        return out


class DynamicConv3D(nn.Module):
    def __init__(self, dim_in, dim_out, location):
        super(DynamicConv3D, self).__init__()
        self.location = location
        self.conv1 = nn.Conv3d(dim_in, dim_in, 3, 1, 1)
        self.bn = nn.BatchNorm3d(dim_in)
        self.ac1 = nn.ReLU()

    def forward(self, x):

        x = self.ac1(self.bn(self.conv1(x)))
        if self.location == 1:
            x = torch.mean(x, dim=1, keepdim=True)
        else:
            x, _ = torch.max(x, dim=2, keepdim=True)

        return x


class Network(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        for p in resnet.parameters():
            p.requires_grad = False
        self.resnet_conv = resnet.conv1
        self.resnet_conv.stride = 1
        self.resnet_conv.padding = (0, 0)

        self.Dynamic_conv = DynamicConv3D(dim_in=64, dim_out=64, location=2)

        self.att = SimAM()
        self.att_conv = nn.Conv2d(64, 48, kernel_size=1, padding=0, stride=1)

        self.local_ConvLSTM = ConvLSTMCell(input_size=64, hidden_size=64)
        self.global_ConvLSTM = ConvLSTMCell(input_size=64, hidden_size=64)

        self.net_out_DM = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )
        self.net_out_FUSE = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

        self.update_conv = nn.Sequential(
            nn.GroupNorm(2, 128),
            nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1),
            nn.Mish(),
        )

    def soft_match(self, query_frame, ref_frames):

        b, channels, h, w = query_frame.shape

        sim_h, sim_w = self.similarity(query_frame, ref_frames)

        sim_h_mapping = torch.matmul(query_frame.permute(0, 1, -1, -2), nn.functional.softmax(sim_h, dim=0)).permute(0, 1, -1, -2)
        sim_w_mapping = torch.matmul(query_frame, nn.functional.softmax(sim_w, dim=0))

        if h != w:
            sim_h_mapping = torch.nn.functional.interpolate(sim_h_mapping, size=(h, w), mode='bilinear', align_corners=False)
            sim_w_mapping = torch.nn.functional.interpolate(sim_w_mapping, size=(h, w), mode='bilinear', align_corners=False)

        sim_mapping = torch.cat([sim_h_mapping, sim_w_mapping], dim=1)

        sim_mapping, _ = torch.topk(sim_mapping, k=channels, dim=1)
        # sim_fg_mapping = torch.matmul(query_frame.permute(0, -1, -2, 1), sim_fg)
        # sim_bg_mapping = torch.matmul(query_frame.permute(0, -1, -2, 1), sim_bg)
        #
        # sim_fg_query_mapping = torch.cat([sim_fg_mapping.permute(0, -1, 1, 2), query_frame], dim=1)
        # sim_bg_query_mapping = torch.cat([sim_bg_mapping.permute(0, -1, 1, 2), ref_frames], dim=1)
        #
        # sim_fg_mapping, _ = torch.topk((sim_fg_query_mapping), k=channels, dim=1)
        # sim_bg_mapping, _ = torch.topk((sim_bg_query_mapping), k=channels, dim=1)

        return sim_mapping

    def predict_fg_bg(self, query_frame, ref_frames):

        sim_mapping = self.soft_match(query_frame, ref_frames)

        # fg_prob, bg_prob = self.softmax(sim_fg_mapping.squeeze(0), sim_bg_mapping.squeeze(0))

        return sim_mapping

    # def segment(self, query_frame, ref_frames, frame_label, last_frame_dm):
    #
    #     h, w = query_frame.shape[-2], query_frame.shape[-1]
    #
    #     fgs, bgs = self.predict_fg_bg(query_frame, ref_frames, last_frame_dm)
    #
    #     if frame_label:
    #         attention_weights = fgs / (fgs + bgs)
    #
    #         attention_weights = torch.nn.functional.interpolate(attention_weights.unsqueeze(0), size=(h, w),
    #                                                             mode='bilinear', align_corners=False)
    #
    #         weighted_features = query_frame * attention_weights
    #         output = self.final_conv(weighted_features)
    #
    #     else:
    #         attention_weights = bgs / (fgs + bgs)
    #
    #         attention_weights = torch.nn.functional.interpolate(attention_weights.unsqueeze(0), size=(h, w),
    #                                                             mode='bilinear', align_corners=False)
    #
    #         weighted_features = query_frame * attention_weights
    #         output = self.final_conv(weighted_features)
    #
    #     return output

    @staticmethod
    def similarity(X, Y):

        b, c, h, w = X.shape

        # normalize along channels
        Xnorm_h = Network.l2_normalization(X, dim=1).squeeze(0).permute(1, 0, -1).reshape(h, c*w)
        Xnorm_w = Network.l2_normalization(X, dim=1).squeeze(0).permute(-1, 0, 1).reshape(w, c*h)

        Ynorm_h = Network.l2_normalization(Y, dim=1).squeeze(0).permute(1, 0, -1).reshape(h, c*w).T
        Ynorm_w = Network.l2_normalization(Y, dim=1).squeeze(0).permute(-1, 0, 1).reshape(w, c*h).T

        return torch.matmul(Xnorm_h, Ynorm_h), torch.matmul(Xnorm_w, Ynorm_w)

    @staticmethod
    def l2_normalization(X, dim, eps=1e-12):
        return X / (torch.norm(X, p=2, dim=dim, keepdim=True) + eps)

    @staticmethod
    def softmax(*args):
        stacked = torch.stack(args, dim=1)  # stack along channel dim
        res = nn.functional.softmax(stacked, dim=1)  # compute softmax along channel dim
        return res.unbind(1)


    def sobel(self, x):
        _, c, h, w = x.shape

        if c == 3:
            gray_image = 0.2989 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            gray_image = gray_image.unsqueeze(1)
        else:
            gray_image = x

        kernel_x = torch.tensor([[[[-3, 0, 3],
                                   [-10, 0, 10],
                                   [-3, 0, 3]]]], dtype=x.dtype, device=DEVICE, requires_grad=False)
        kernel_y = torch.tensor([[[[3, 10, 3],
                                   [0, 0, 0],
                                   [-3, -10, -3]]]], dtype=x.dtype, device=DEVICE, requires_grad=False)

        edge_x = F.conv2d(gray_image, kernel_x, padding=1)
        edge_y = F.conv2d(gray_image, kernel_y, padding=1)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge

    def frame_create(self, A, B, label):
        frame_list = B
        lmd = 0.95
        if label == 0:
            # frame_list = torch.cat([frame_list, B], dim=0)
            for i in range(10):
                frame_list = torch.cat([frame_list, lmd * B + (1 - lmd) * A], dim=0)
                lmd -= 0.1
            frame_list = torch.cat([frame_list, A], dim=0)
        else:
            frame_list = torch.cat([B.repeat(6, 1, 1, 1), A.repeat(6, 1, 1, 1)], dim=0)

        frame_list = frame_list.view(-1, 12, frame_list.size(1), frame_list.size(2), frame_list.size(3))

        return frame_list

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):

        out_tensor = F.pad(tensors, padding, mode=mode, value=value)
        return out_tensor

    def forward(self, A, B, label):

        _, _, h, w = A.shape

        A = self.tensor_padding(tensors=A, padding=(3, 3, 3, 3), mode='replicate')
        B = self.tensor_padding(tensors=B, padding=(3, 3, 3, 3), mode='replicate')
        frame_list = self.frame_create(A, B, label)

        b, t, c, _, _ = frame_list.shape

        Decisionmap_list = []
        for i in range(t):
            if i == 0:
                local_cell_next = self.resnet_conv(frame_list[:, 0, :, :, :])
                local_hidden_next = nn.functional.leaky_relu(local_cell_next)
                feature_bank = local_cell_next
                global_cell_next = local_cell_next
                Decisionmap_list.append(self.net_out_DM(local_hidden_next))

            else:
                x = self.resnet_conv(frame_list[:, i, :, :, :])
                x_att = self.att(x)
                forward_frame_fea = self.resnet_conv(frame_list[:, :i, :, :, :].squeeze(0))
                forward_frame_fea = forward_frame_fea.unsqueeze(0).permute(0, 2, 1, 3, 4)
                forward_frame_fea = self.Dynamic_conv(forward_frame_fea).squeeze(0)
                forward_frame_fea = forward_frame_fea.permute(1, 0, 2, 3)

                local_cell_next, local_hidden_next = self.local_ConvLSTM(local_cell_next, local_hidden_next, x_att)
                global_cell_next, global_hidden_next = self.global_ConvLSTM(global_cell_next, feature_bank, self.att(forward_frame_fea))

                softmatting = self.predict_fg_bg(global_hidden_next, local_hidden_next)
                x_hard = self.update_conv(torch.cat([global_hidden_next, local_hidden_next], dim=1))
                x_soft = self.update_conv(torch.cat([softmatting, local_hidden_next], dim=1))
                x = x_hard + x_soft

                feature_bank = self.update_conv(torch.cat([feature_bank, local_hidden_next*self.net_out_DM(x)], dim=1))

                Decisionmap_list.append(self.net_out_DM(x))

        Decisionmap = torch.stack(Decisionmap_list, dim=0).squeeze(1)
        Fused_img = self.net_out_FUSE(feature_bank)

        return Decisionmap, Fused_img


if __name__ == '__main__':
    test_tensor_A = torch.zeros(1, 3, 260, 260).to(DEVICE)
    test_tensor_B = torch.zeros((1, 3, 260, 260)).to(DEVICE)
    resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
    model = Network(resnet).to(DEVICE)
    model(test_tensor_A, test_tensor_B, 1)
    # print(model)
    flops, params = profile(model, inputs=(test_tensor_A, test_tensor_B, 1))
    flops, params = clever_format([flops, params], "%.10f")
    print('flops: {}, params: {}'.format(flops, params))

    # Pre, Fused_img = model(test_tensor_A, test_tensor_B, 1)
    # print(Pre.shape, Fused_img.shape)