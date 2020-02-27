import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset,load_dictionary
from train import train
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from models import FrameQA_model
from models import Count_model
from models import Trans_model
from models import Action_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--task', type=str, default='Count',help='FrameQA, Count, Action, Trans')
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='temporalAtt', help='temporalAtt')
    parser.add_argument('--max_len',type=int, default=20)
    parser.add_argument('--char_max_len', type=int, default=15)
    parser.add_argument('--num_frame', type=int, default=36)
    parser.add_argument('--output', type=str, default='saved_models/%s/exp-11')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--sentense_file_path',type=str, default='./data/dataset')
    parser.add_argument('--glove_file_path', type=str, default='/home/zengpengpeng/project/vqa1/bottom-up-attention-vqa-master/data/glove/glove.6B.300d.txt')
    parser.add_argument('--feat_category',type=str,default='resnet')
    parser.add_argument('--feat_path',type=str,default='/mnt/data2/lixiangpeng/dataset/tgif/features')
    parser.add_argument('--Multi_Choice',type=int, default=5)
    parser.add_argument('--vid_enc_layers', type=int, default=1)
    parser.add_argument('--test_phase', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print('parameters:', args)
    print('task:',args.task,'model:', args.model)

    # dictionary = Dictionary.load_from_file('./dictionary.pkl')
    dictionary = load_dictionary(args.sentense_file_path, args.task)


    train_dset = VQAFeatureDataset(args, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Train')
    # val_dset = VQAFeatureDataset(args, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Valid')
    eval_dset = VQAFeatureDataset(args, dictionary, args.sentense_file_path,args.feat_category,args.feat_path, mode='Test')
    batch_size = args.batch_size

    model_name = args.task+'_model'
    model = getattr(locals()[model_name], 'build_%s' % args.model)(args.task, args.vid_enc_layers, train_dset, args.num_hid, dictionary, args.glove_file_path).cuda()

    print('========start train========')
    model = model.cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    # val_loader = DataLoader(val_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    train(model, train_loader, eval_loader, args.epochs, args.output)
