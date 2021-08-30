import argparse
import torch
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *


from codes.model import * 
parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='hazelnut', type=str)
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=64, type=int)

parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

args = parser.parse_args()


def train():
    obj = args.obj
    D = args.D
    lr = args.lr
        
    with task('Networks'):
        
        # 加上pretrained feature extractor 
        vgg16_model = Vgg16().cuda()
        model = DifferNet().cuda()
        model.to(c.device)
        enc = EncoderHier(64, D).cuda()

        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()

        # modules = [enc, cls_64, cls_32, model]
        modules = [model]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

       

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')
        train_x = NHWC2NCHW(train_x)

        rep = 100
        datasets = dict()

        '''
        依據不同的patch_size產生不同的dataset (position dataset 和 svdd dataset)
        '''
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)


    print('Start training')
    train_loss = list()
    for i_epoch in range(args.epochs):
        if i_epoch == 0 or i_epoch != 0:
            for module in modules:
                module.train()

            for d in loader:
                '''
                position_dataset: patch1, patch2, pos 來自同張圖片的兩個patch
                svdd_dataset: patch1, patch2, label
                '''
                d = to_device(d, 'cuda', non_blocking=True)
                opt.zero_grad()

                y_cat = list() #存feature

                # images patch
                patch1_64, patch2_64 = d['svdd_64']
                patch1_32, patch2_32 = d['svdd_32']

                # get feature of image patches 
                fe64_patch1 = vgg16_model(patch1_64)
                fe64_patch2 = vgg16_model(patch2_64)
                fe32_patch1 = vgg16_model(patch1_32)
                fe32_patch2 = vgg16_model(patch2_32)

              
                y_cat = list()
                for layer_feature in range(5):
                    y_cat.append(torch.mean(fe64_patch1[layer_feature], dim=(2,3)))
                for layer_feature in range(5):
                    y_cat.append(torch.mean(fe64_patch2[layer_feature], dim=(2,3)))
                for layer_feature in range(5):
                    y_cat.append(torch.mean(fe32_patch1[layer_feature], dim=(2,3)))
                for layer_feature in range(5):
                    y_cat.append(torch.mean(fe32_patch2[layer_feature], dim=(2,3)))
               
                # normalizing flow's input feature y
                y = torch.cat(y_cat, dim=1) 

                z = model(y)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                train_loss.append(t2np(loss))
                loss.backward()
                opt.step()
                
              
                # loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])  # arg: (position_classifier, encoder_hier, data)
                # loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                # loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                # loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

                # loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

                # loss.backward()
                # opt.step()
        mean_train_loss = np.mean(train_loss)
        if c.verbose:
            print('Epoch: {:d} \t train loss: {:.4f}'.format(i_epoch, mean_train_loss))

        # eval 
        model.eval() 

        '''
        改成 patch 之後
        把圖切成小patch, 用NF把每一個patch map到latent z 
        分別計算每個patch的anomaly score,
        再concat成一張完整的圖, 如果有其中一塊超過threshold就判定他為anomaly
        再去比對label算auroc 
        '''
        




        # aurocs = eval_encoder_NN_multiK(enc, obj)
        # log_result(obj, aurocs)
        # enc.save(obj)



def log_result(obj, aurocs):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |mult| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj})')


if __name__ == '__main__':
    train()
