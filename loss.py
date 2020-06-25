import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
import os
import scipy.misc
from glob import glob
from scipy import io


class FocalLoss3d_ver1(nn.Module):
    def __init__(self, gamma=2, pw=10, threshold=1.0, erode=3, backzero=0):
        super().__init__()
        self.gamma = gamma
        self.pw=pw
        self.threshold=threshold
        self.erode=erode
        self.backzero=backzero

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits

        weight=torch.clamp(target,min=0,max=self.threshold)
        weight=weight-self.threshold*(weight==self.threshold).to(torch.float)
        kernel=torch.ones(1, 1, 2*self.erode+1, 2*self.erode+1, 2*self.erode+1).to(torch.device("cuda"))
        weight=F.conv3d(weight , kernel , padding=self.erode)
        weight=(weight>0).to(torch.float)
        weight=1+weight*self.pw

        if self.backzero != 0 :
            mask=((input<0).to(torch.float))*((target==0).to(torch.float))
            mask=(1-mask)
            weight=weight*mask

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        diff=target-input
        loss=torch.abs(diff)*weight

        return loss.mean()

class FocalLoss3d_ver2(nn.Module):
    def __init__(self, gamma=2, pw=10, erode=2, is_weight=0):
        super().__init__()
        self.gamma = gamma
        self.pw=pw
        self.erode=erode
        self.is_weight=is_weight

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
        if self.is_weight!=0:
            weight_mask_=(target<0.1).to(torch.float)*(target>-0.1).to(torch.float)
            kernel = torch.ones(1, 1, 3, 3, 3).to(torch.device("cuda"))
            weight_mask=F.conv3d(weight_mask_, kernel, padding=1)
            weight_mask=((weight_mask)>0).to(torch.float)-weight_mask_
            kernel=torch.ones(1, 1, 2*self.erode+1, 2*self.erode+1, 2*self.erode+1).to(torch.device("cuda"))
            weight_mask=F.conv3d(weight_mask, kernel, padding=self.erode).to(torch.float)
            weight_mask = 1 + weight_mask * self.pw/pow(self.erode,3)
            diff=target-input
            loss = torch.abs(diff)*weight_mask
            return loss.mean()
        elif self.is_weight==0:
            diff=target-input
            loss = torch.abs(diff)
            return loss.mean()

        return None

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold=0.01

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        loss=0
        a=(input>self.threshold).to(torch.float)
        b=(target>self.threshold).to(torch.float)
        if(b.sum()>0):
            loss += 1-2*((a*b).sum())/(a.sum()+b.sum()+self.threshold)
        a = (input < -self.threshold).to(torch.float)
        b = (target < -self.threshold).to(torch.float)
        if (b.sum() > 0):
            loss += 1-2*((a*b).sum())/(a.sum()+b.sum()+self.threshold)

        return loss

class DiceDis(nn.Module):
    def __init__(self, ratio1=0, gamma=2, pw=10, erode=2, is_weight=0):
        super().__init__()
        self.l1=DiceLoss()
        self.l2=FocalLoss3d_ver2(gamma,pw,erode,is_weight)
        self.ratio1=ratio1

    def forward(self, input, target):
        return self.ratio1*self.l1(input,target)+self.l2(input,target)


class Classifying(nn.Module):
    def __init__(self, gamma=2, pw=1, erode=2, is_weight=0):
        super().__init__()
        self.gamma = gamma
        self.pw = pw
        self.erode = erode
        self.is_weight = is_weight

    def forward(self, input, target):
        if self.is_weight != 0:
            weight_mask_ = (target < 0.1).to(torch.float) * (target > -0.1).to(torch.float)
            kernel = torch.ones(1, 1, 3, 3, 3).to(torch.device("cuda"))
            weight_mask = F.conv3d(weight_mask_, kernel, padding=1)
            weight_mask = ((weight_mask) > 0).to(torch.float) - weight_mask_
            kernel = torch.ones(1, 1, 2 * self.erode + 1, 2 * self.erode + 1, 2 * self.erode + 1).to(torch.device("cuda"))
            weight_mask = F.conv3d(weight_mask, kernel, padding=self.erode).to(torch.float)
            weight_mask = 1 + weight_mask * self.pw/pow(self.erode,3)
            target=torch.squeeze((target>0.1).to(torch.long)+2*(target<-0.1).to(torch.long))
            #weight_mask_= ((1+torch.squeeze((target>0.1).to(torch.float)+5*(target<-0.1).to(torch.float))))

            loss = F.cross_entropy(input, target, reduce=None)
            loss = loss*torch.pow((1-torch.exp(-loss)),self.gamma)
            loss = loss * weight_mask
            return loss.mean()
        elif self.is_weight == 0:
            target=torch.squeeze((target>0.1).to(torch.long)+2*(target<-0.1).to(torch.long),dim=1)
            weight_mask=(1+torch.squeeze((target>0.1).to(torch.float)+5*(target<-0.1).to(torch.float),dim=1))
            loss = F.cross_entropy(input, target, reduce=None)
            loss = loss * torch.pow((1 - torch.exp(-loss)), self.gamma)
            loss = loss * weight_mask
            return loss.mean()

        return None


class CircularLoss(nn.Module):
    def __init__(self, gamma=2, pw=10, erode=2, is_weight=0):
        super().__init__()
        self.coeff=0.2

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        #mask=(target>0).cpu().numpy()
        #center=ndimage.measurements.center_of_mass(mask)
        #radius=pow(mask.sum()*3/4/3.14,1/3)
        #mask=
        #mask=ndimage.morphology.distance_transform_edt(1-mask)*(input>0.01).to(torch.float)
        #loss=loss.mean()

        if self.is_weight!=0:
            weight_mask=(target<0.1).to(torch.float)
            kernel = torch.ones(1, 1, 3, 3, 3).to(torch.device("cuda"))
            weight_mask=F.conv3d(weight_mask, kernel, padding=1)
            weight_mask=((weight_mask)>0).to(torch.float)
            kernel=torch.ones(1, 1, 2*self.erode+1, 2*self.erode+1, 2*self.erode+1).to(torch.device("cuda"))
            weight_mask=F.conv3d(weight_mask, kernel, padding=self.erode).to(torch.float)
            weight_mask = 1 + weight_mask * self.pw/pow(self.erode,3)
            diff=target-input
            loss = torch.abs(diff)*weight_mask
            return loss.mean()
        elif self.is_weight==0:
            diff=target-input
            loss = torch.abs(diff)
            return loss.mean()

        return None


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha, torch_device):
        super().__init__()
        self.alpha = alpha
        self.beta  = 1 - alpha
        self.smooth = 1.0

    def forward(self, target_, output_):
        output_ = F.sigmoid(output_)

        target_f = target_.contiguous().view(-1)
        output_f = output_.contiguous().view(-1)

        """
        P : set of predicted, G : ground truth label
        Tversky Index S is
        S(P, G; a, b) = PG / (PG + aP\G + bG\P)

        Tversky Loss T is
        PG = sum of P * G
        G\P = sum of G not P
        P\G = sum of P not G
        T(a, b) = PG / (PG + aG\P + bP\G)
        """

        PG = (target_f * output_f).sum()
        G_P = ((1 - target_f) * output_f).sum()
        P_G = ((1 - output_f) * target_f).sum()

        loss = (PG + self.smooth) / (PG + (self.alpha * G_P) + (self.beta * P_G) + self.smooth)
        return loss

if __name__ == "__main__":
    img_path="/data1/0000Deployed_JY/strain_/20180416_20180416.131920.821.CD4-K562-003_000000_segmented_auto_001_.mat"
    data_pack = io.loadmat(img_path)
    # 2D ( 1 x H x W )
    input_np = data_pack['input']
    target_np = data_pack['target']

    target=torch.from_numpy(target_np).view(1,1,128,128,64)

    weight_mask_ = ((target < 0.1).to(torch.float) * (target > -0.1).to(torch.float))
    kernel = torch.ones(1, 1, 3, 3, 3)
    weight_mask = F.conv3d(weight_mask_, kernel, padding=1)
    weight_mask = ((weight_mask) > 0).to(torch.float) - weight_mask_
    kernel = torch.ones(1, 1, 5, 5, 5)
    weight_mask = F.conv3d(weight_mask, kernel, padding=2).to(torch.float)
    weight_mask = 1 + weight_mask * 10 / pow(2, 3)
    diff = target - 0
    loss = torch.abs(diff) * weight_mask

    data = {}
    coeff_mag = 2000
    data['loss'] = loss.numpy()
    data['weight_mask'] = weight_mask.numpy()
    scipy.io.savemat("/home/jysong/PyCharmProjects_JY/181007_3DcellSegmentation_regressionVer" + "/losstest.mat", data)
