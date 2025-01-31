import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import savemat
from torch import optim
from model import S2VNet
from utils import AvgrageMeter, accuracy, output_metric, NonZeroClipper, print_args
from dataset import prepare_dataset
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--fix_random', action='store_true', default=True, help='fix randomness')
parser.add_argument('--gpu_id', default='1', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--dataset', choices=['Indian', 'Berlin', 'Augsburg'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--model_name', choices=['s2vnet'], default='s2vnet', help='S2VNet')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        if 's2vnet' in args.model_name:
            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, feature_abu, edm_per = model(batch_data)

            band = re_unmix.shape[1] // 2  # 2 represents the number of decoder layer
            output_linear = re_unmix[:,0:band] + re_unmix[:,band:band*2]
            re_unmix = re_unmix_nonlinear + output_linear

            # compute kl loss
            kl_div = -0.5 * (edm_var_2 + 1 - edm_var_1 ** 2 - edm_var_2.exp())
            kl_div = kl_div.sum() / batch_pred.shape[0]
            kl_div = torch.max(kl_div, torch.tensor(0).cuda())

            # compute tv loss
            edm_per_diff = edm_per[1:, :] - edm_per[:(edm_per.shape[0] - 1), :]
            edm_per_diff = edm_per_diff.abs()
            loss_tv = edm_per_diff.mean()  # endmember tv_loss

            b_x, h_x, w_x = feature_abu.shape[0], feature_abu.shape[-2], feature_abu.shape[-1]
            h_tv = torch.pow((feature_abu[:, :, 1:, :] - feature_abu[:, :, :h_x - 1, :]), 2).sum()
            w_tv = torch.pow((feature_abu[:, :, :, 1:] - feature_abu[:, :, :, :w_x - 1]), 2).sum()
            loss_tv_abu = (h_tv + w_tv) / (b_x * 2 * h_x * w_x)  # abundance tv_loss

            sad_loss = torch.mean(torch.acos(torch.sum(batch_data * re_unmix, dim=1)/
                        (torch.norm(re_unmix, dim=1, p=2) * torch.norm(batch_data, dim=1, p=2)+1e-5)))
            loss = criterion(batch_pred, batch_target) + sad_loss + 0.01 * kl_div + 0.01 * loss_tv + 0.01 * loss_tv_abu
        else:
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        if 's2vnet' in args.model_name:
            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, _, _ = model(batch_data)

            band = re_unmix.shape[1]//2  # 2 represents the number of decoder layer
            output_linear = re_unmix[:,0:band] + re_unmix[:,band:band*2]
            re_unmix = re_unmix_nonlinear + output_linear

            sad_loss = torch.mean(torch.acos(torch.sum(batch_data * re_unmix, dim=1)/
                        (torch.norm(re_unmix, dim=1, p=2) * torch.norm(batch_data, dim=1, p=2))))
            loss = criterion(batch_pred, batch_target) + sad_loss
        else:
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

def test_epoch(model, test_loader):
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        if args.model_name == 's2vnet':
            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, _, _ = model(batch_data)
        else:
            batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())

    return pre

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.fix_random:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    ## prepare dataset
    label_train_loader, label_test_loader, label_true_loader, band, height, width, num_classes, label, total_pos_true = prepare_dataset(args)
    # create model
    if args.model_name == 's2vnet':
        model = S2VNet(band, num_classes, args.patches)
    else:
        raise KeyError("{} model is unknown.".format(args.model_name))
    model = model.cuda()
    print("Model Name: {}".format(args.model_name))

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    apply_nonegative = NonZeroClipper()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
    #-------------------------------------------------------------------------------
    if args.flag_test == 'test':
        model.eval()
        model.load_state_dict(torch.load('./results/model.pkl'))
        pre_u = test_epoch(model, label_true_loader)

        prediction_matrix = np.zeros((height, width), dtype=float)
        for i in range(total_pos_true.shape[0]):
            prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
        savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
    else:
        print("start training")
        tic = time.time()
        min_val_obj, best_OA = 0.5, 0
        for epoch in range(args.epoches):
            scheduler.step()

            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
            OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
            print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                            .format(epoch+1, train_obj, train_acc))

            model.unmix_decoder.apply(apply_nonegative) # regularize unmix decoder

            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
                OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
                print("OA: {:.4f} AA: {:.4f} Kappa: {:.4f}"
                            .format(OA2, AA_mean2, Kappa2))
                print("*************************")

                if OA2 > min_val_obj and epoch > 10:
                    model_save_path = os.path.join('./results/', args.dataset+'_'+args.model_name+'_p'+str(args.patches)+
                                                    '_'+str(round(OA2*100, 2))+'_epoch'+str(epoch)+'.pkl')
                    torch.save(model.state_dict(), model_save_path)

                    min_val_obj = OA2
                    best_epoch = epoch
                    best_OA = OA2
                    best_AA = AA_mean2
                    best_Kappa = Kappa2
                    best_each_AA = AA2

        toc = time.time()
        print("Running Time: {:.2f}".format(toc-tic))
        print("**************************************************")
        if best_OA == 0:
            model_save_path = os.path.join('./results/', args.dataset+'_'+args.model_name+'_p'+str(args.patches)+
                                            '_'+str(round(OA2*100, 2))+'_epoch'+str(epoch)+'.pkl')
            torch.save(model.state_dict(), model_save_path)

    print("Final result:")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print(AA2)
    print("**************************************************")
    print("Best Epoch: {:03d} | Best OA: {:.4f} | Best AA: {:.4f} | Best Kappa: {:.4f}".format(best_epoch, best_OA, best_AA, best_Kappa))
    print(best_each_AA)
    print("**************************************************")
    print("Parameter:")
    print_args(vars(args))

if __name__ == '__main__':
    main()
