from multiprocessing import Process
import os
import argparse
import torch
import torch.nn as nn
import sys
import utils
from Logger import Logger

from models.unet3D_glob import Unet3D_glob
from trainers.CNNTrainer import CNNTrainer
from datas.TBLoader import TBloader
import copyreg
torch.backends.cudnn.benchmark = True
from loss import FocalLoss, TverskyLoss, FocalLoss3d_ver1, FocalLoss3d_ver2, DiceDis


"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "DeepIS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="0",
                        help="Select CPU Number workers")
    parser.add_argument('--model', type=str, default='unet_gh',
                        choices=['fusion', "unet", "gcn_c", "gcn_r", "unet_reduced"])
    # TODO : Weighted BCE
    parser.add_argument('--feature_scale', type=int, default=4)

    parser.add_argument('--in_channel', type=int, default=1)

    # FusionNet Parameters
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--clamp', type=tuple, default=None)

    parser.add_argument('--augment', type=str, default='',
                        help='The type of augmentaed ex) crop,rotate ..  | crop | flip | elastic | rotate |')

    # TODO : Weighted BCE
    parser.add_argument('--loss', type=str, default='l1',
                        choices=["l1", "l2"])

    #parser.add_argument('--data', type=str, default='data',
    #                    choices=['All', 'Balance', 'data', "Only_Label"],
    #                    help='The dataset | All | Balance | Only_Label |')

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--test', type=int, default=0, help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')
    parser.add_argument('--lrG', type=float, default=0.00005)
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))

    return parser.parse_args()


def reconstruct_torch_dtype(torch_dtype: str):
    # a dtype string is "torch.some_dtype"
    dtype = torch_dtype.split('.')[1]
    return getattr(torch, dtype)

def pickle_torch_dtype(torch_dtype : torch.dtype):
    return reconstruct_torch_dtype, (str(torch_dtype),)

if __name__ == "__main__":
    arg = arg_parse()
    ##########################################################################################################################################
    ##
    ## Static Setting Part (Don't change)
    ##
    ##########################################################################################################################################
    arg.save_dir = "nets_1013_unet_glob1007_disdice_FRE_pw1_erode2_feat1_trans64"
    arg.save_dir = "%s/outs/%s"%(os.getcwd(), arg.save_dir)

    #copyreg.pickle(torch.dtype, pickle_torch_dtype)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    # Unet3D: in models/ DataParallel: /
    net = Unet3D_glob(feature_scale=1, trans_feature=64)
    net = nn.DataParallel(net).to(torch_device)

    logger = Logger(arg.save_dir)

    backzero=1
    recon_loss=FocalLoss3d_ver1(backzero=backzero)
    val_loss=FocalLoss3d_ver1(gamma=2, pw=1, threshold=1.0, erode=3,backzero=backzero)
    model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss,logger=logger)

    ##########################################################################################################################################
    ##
    # TODO : Manual Setting Part (Change)
    ##
    ##########################################################################################################################################

    ## 1. choose model / First pick : epoch[0324] / Second pick : epoch[0315]
    model.load(filename="epoch[0324]_losssum[0.053456].pth.tar") #model.load(filename="epoch[0315]_losssum[0.056779].pth.tar")

    ## 2. choose file path for the input -> fpath_input
    fpath_input = "/data1/Moosung_CART/Fig07_supplementary_crop/41BB_01"

    ## 3. choose file path for the output -> fpath_output
    fpath_output="/data1/Moosung_CART/Fig07_supplementary_crop_result"

    ## 4. run the code
    ######phase 1######
    if(True):
        if (True):
            if os.path.exists(fpath_output) is False:
                os.mkdir(fpath_output)
            test_loader = TBloader(fpath_input, batch_size=1, transform=None,
                                        cpus=arg.cpus, shuffle=True,
                                        drop_last=True, rotate_num=0)
            model.test(test_loader, savedir=fpath_output)

    ######phase 2######
    if(False):
        if os.path.exists(sdir) is False:
            os.mkdir(sdir)

        if (arg.alchemy == 0):
            filenames=os.listdir(f_path_test)
            for filename in filenames:
                sdir_=os.path.join(sdir,filename)
                if os.path.exists(sdir_) is False:
                    os.mkdir(sdir_)
                test_loader = TBloader(os.path.join(f_path_test,filename), batch_size=1, transform=None,
                                        cpus=arg.cpus, shuffle=True,
                                        drop_last=True, rotate_num=0)
                model.test(test_loader, savedir=sdir_)
        elif(arg.alchemy ==1):
            filenames = os.listdir(f_path_test)
            for filename in filenames:
                for rotate_num in [0, 1, 2, 3]:
                    sdir_ = os.path.join(sdir, filename)
                    sdir_ = sdir_ + '_' + str(rotate_num)
                    if os.path.exists(sdir_) is False:
                        os.mkdir(sdir_)
                    test_loader = TBloader(os.path.join(f_path_test, filename),
                                                batch_size=1,
                                                transform=None,
                                                cpus=arg.cpus, shuffle=True,
                                                drop_last=True, rotate_num=0)
                    model.test(test_loader, savedir=sdir_)
    ######phase 2######


