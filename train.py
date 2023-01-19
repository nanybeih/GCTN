import argparse
from torch import optim
from torch.utils.data.dataloader import DataLoader

from metrics import *
from model import *
from utils import *
import pickle
import torch
import sys
import logging

parser = argparse.ArgumentParser()


parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='hotel',
                    help='eth,hotel,univ,zara1,zara2')
# Training specifc parameters                                                               ``
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=10,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum of lr')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='weight_decay on l2 reg')
parser.add_argument('--lr_sh_rate', type=int, default=500,
                    help='number of steps to drop the lr')
parser.add_argument('--milestones', type=int, default=[50,100],
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='STTransformer_sinembedding_v5_kersize13_embed64_32_4',
                    help='personal tag for the model')
parser.add_argument('--gpu_num', default="0", type=str)
parser.add_argument('--KSTEPS', type=int, default=20,
                    help='KSTEPS')
parser.add_argument('--l2_loss_weight', type=float, default=1000.0,
                    help='l2_loss_weight')
args = parser.parse_args()

parser2 = argparse.ArgumentParser()

parser2.add_argument('--input_dim', type=int, default=2)
parser2.add_argument('--pred_dim', type=int, default=2)
parser2.add_argument('--input_embed_size', type=int, default=64,
                    help='eth,hotel,univ,zara1,zara2')
# Training specifc parameters
parser2.add_argument('--dec_hidden_size', type=int, default=128,
                    help='hidden_size')
parser2.add_argument('--enc_hidden_size', type=int, default=128,
                    help='enc_hidden_size')
parser2.add_argument('--latent_dim', type=int, default=32,
                    help='gadient clipping')
parser2.add_argument('--dec_with_z', type=bool, default=True,
                    help='dec_with_z')

opt = parser2.parse_args()

FORMAT = '[%(asctime)s-%(levelname)-%(filename)s-%(lineno)4d]-%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT,stream=sys.stdout)
logger = logging.getLogger(__name__)
haddler = logging.FileHandler("{}_{}.txt".format(args.tag, args.dataset))
logger.addHandler(haddler)
logger.info("Training initiating....")
logger.info(args)

ade_fde_train = {'ade': [], 'fde': [], '_traj_fde': []}
ade_fde_val = {'ade': [], 'fde': [], '_traj_fde': []}
metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999, 'min_train_epoch': -1,
                    'min_train_loss': 9999999999999999}
constant_ade_fde_val_metrics = {'min_ade_val_epoch': -1, 'min_val_ade': 9999999999999999, 'min_fde_val_epoch': -1,
                    'min_val_fde': 9999999999999999}
constant_ade_fde_train_metrics = {'min_ade_train_epoch': -1, 'min_train_ade': 9999999999999999, 'min_fde_train_epoch': -1,
                    'min_train_fde': 9999999999999999}
def train(epoch, model, optimizer, checkpoint_dir, loader_train):
    global constant_ade_fde_train_metrics, ade_fde_train, metrics, constant_metrics, logger
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
    ade_ = []
    traj_fde = []
    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr, A_obs, A_pred = batch


        optimizer.zero_grad()


        if batch_count % args.batch_size != 0 and cnt != turn_point:
            traj_out, loss_dict, probability = model(V_obs.permute(0,3,2,1), A_obs.squeeze(0), V_tr.squeeze(0).permute(1,0,2), KSTEPS=20)

            V_tr = V_tr.squeeze(0)
            traj_loss = cvae_traj_loss(traj_out, V_tr.permute(1, 0, 2), best_of_many=True)
            KSTEPS = traj_out.shape[1]
            pred_traj = torch.cumsum(traj_out, dim=2)
            last_obs = obs_traj.squeeze(0).permute(0, 2, 1)[:, -1, :]
            pred_traj = pred_traj + last_obs.unsqueeze(1).unsqueeze(2).repeat(1, KSTEPS, traj_out.shape[-2], 1)

            l = traj_loss + loss_dict['loss_kld']

            target = pred_traj_gt.permute(1, 0, 3, 2).repeat(1, KSTEPS, 1, 1)
            traj_rmse = torch.sqrt(torch.sum((pred_traj - target) ** 2, dim=-1)).mean(dim=2)
            traj_goal_rmse = torch.sqrt(torch.sum((pred_traj[:,:,-1,:] - target[:, :, -1, :]) ** 2, dim=-1))

            if KSTEPS != 1:
                best_traj_idx = torch.argmin(traj_rmse, dim=1)

                traj_err = traj_rmse[range(len(best_traj_idx)), best_traj_idx]

                ade_.append(traj_err)

                best_traj_goal_idx = torch.argmin(traj_goal_rmse, dim=1)
                traj_goal_err = traj_goal_rmse[range(len(best_traj_goal_idx)), best_traj_goal_idx]
                traj_fde.append(traj_goal_err)

            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss = loss + l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            loss_batch += loss.item()
            logger.info('TRAIN: , \t Epoch: {},\t Loss: {}'.format(epoch,loss_batch / batch_count))
    metrics['train_loss'].append(loss_batch / batch_count)

    ade_fde_train['ade'].append(torch.cat(ade_,dim=0).mean().item())
    ade_fde_train['_traj_fde'].append(torch.cat(traj_fde, dim=0).mean().item())
    if ade_fde_train['ade'][-1] < constant_ade_fde_train_metrics['min_train_ade']:
        constant_ade_fde_train_metrics['min_train_ade'] = ade_fde_train['ade'][-1]
        constant_ade_fde_train_metrics['min_ade_train_epoch'] = epoch
    if ade_fde_train['_traj_fde'][-1] < constant_ade_fde_train_metrics['min_train_fde']:
        constant_ade_fde_train_metrics['min_train_fde'] = ade_fde_train['_traj_fde'][-1]
        constant_ade_fde_train_metrics['min_fde_train_epoch'] = epoch
    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')  # OK


def vald(epoch, model, checkpoint_dir, loader_val):
    global constant_ade_fde_val_metrics, ade_fde_val, metrics, constant_metrics, logger
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
    ade_ = []
    traj_fde = []
    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr, A_obs, A_pred = batch

        with torch.no_grad():

            traj_out, loss_dict, probability = model(V_obs.permute(0,3,2,1), A_obs.squeeze(0), V_tr.squeeze(0).permute(1,0,2), KSTEPS=20) #V_tr.squeeze(0).permute(1,0,2)
            V_tr = V_tr.squeeze(0)
            traj_loss = cvae_traj_loss(traj_out, V_tr.permute(1,0,2), best_of_many=True)

            l = traj_loss + loss_dict['loss_kld']

            KSTEPS = traj_out.shape[1]
            pred_traj = torch.cumsum(traj_out, dim=2)
            last_obs = obs_traj.squeeze(0).permute(0, 2, 1)[:, -1, :]
            pred_traj = pred_traj + last_obs.unsqueeze(1).unsqueeze(2).repeat(1, KSTEPS, traj_out.shape[-2], 1)
            target = pred_traj_gt.permute(1, 0, 3, 2).repeat(1, KSTEPS, 1, 1)

            traj_rmse = torch.sqrt(torch.sum((pred_traj - target) ** 2, dim=-1)).mean(dim=2)
            traj_goal_rmse = torch.sqrt(torch.sum((pred_traj[:,:,-1,:] - target[:, :, -1, :]) ** 2, dim=-1))

            if KSTEPS != 1:
                best_traj_idx = torch.argmin(traj_rmse, dim=1)
                traj_err = traj_rmse[range(len(best_traj_idx)), best_traj_idx]
                ade_.append(traj_err)


                best_traj_goal_idx = torch.argmin(traj_goal_rmse, dim=1)
                traj_goal_err = traj_goal_rmse[range(len(best_traj_goal_idx)), best_traj_goal_idx]
                traj_fde.append(traj_goal_err)


            if batch_count % args.batch_size != 0 and cnt != turn_point:
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l

            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                loss_batch += loss.item()
                logger.info('VALD:, \t Epoch:, {}, \t Loss: {}'.format(epoch,loss_batch / batch_count))
    metrics['val_loss'].append(loss_batch / batch_count)


    ade_fde_val['ade'].append(torch.cat(ade_,dim=0).mean().item())
    ade_fde_val['_traj_fde'].append(torch.cat(traj_fde, dim=0).mean().item())
    if ade_fde_val['ade'][-1] < constant_ade_fde_val_metrics['min_val_ade']:
        constant_ade_fde_val_metrics['min_val_ade'] = ade_fde_val['ade'][-1]
        constant_ade_fde_val_metrics['min_ade_val_epoch'] = epoch
    if ade_fde_val['_traj_fde'][-1] < constant_ade_fde_val_metrics['min_val_fde']:
        constant_ade_fde_val_metrics['min_val_fde'] = ade_fde_val['_traj_fde'][-1]
        constant_ade_fde_val_metrics['min_fde_val_epoch'] = epoch
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args, opt):
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './dataset/' + args.dataset + '/'

    dset_train = TrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1, 
        shuffle=True,
        num_workers=16)

    dset_val = TrajectoryDataset(
        data_set + 'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  
        shuffle=False,
        num_workers=16)

    logger.info('Training started ...')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    in_channels = 2
    T_obs = 8
    T_pred = 12
    out_dims = 2
    num_layers = 1
    embed_size = 64
    heads = 4
    kernel_size = 8
    dropout = 0.1
    forward_expansion = 2
    n_tcn = 5
    model = STTransformer_sinembedding(opt, in_channels, embed_size, kernel_size, num_layers, T_obs, T_pred, heads, n_tcn, out_dims, forward_expansion, dropout).cuda()



    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #
    if args.use_lrschd:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5,
                                                            min_lr=1e-10, verbose=1)


    checkpoint_dir = './model_checkpoints/' + args.tag + '/' + args.dataset + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    logger.info('Data and model loaded')
    logger.info('Checkpoint dir:{}'.format(checkpoint_dir))

    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train)
        vald(epoch, model, checkpoint_dir, loader_val)

        if args.use_lrschd:
            scheduler.step(constant_metrics['min_val_loss'])

        logger.info(''.format('*' * 30))
        logger.info('Epoch:,  {}/{}, :{}'.format(args.dataset, args.tag, epoch))
        for k, v in metrics.items():
            if len(v) > 0:
                logger.info("metrics:{},{}".format(k, v[-1]))

        for k, v in ade_fde_train.items():
            if len(v) > 0:
                logger.info("ade_fde_train:{}，{}".format(k, v[-1]))
        for k, v in ade_fde_val.items():
            if len(v) > 0:
                logger.info("ade_fde_val:{}，{}".format(k, v[-1]))

        logger.info(constant_ade_fde_train_metrics)
        logger.info(constant_ade_fde_val_metrics)


        logger.info(constant_metrics)
        logger.info(''.format('*' * 30))

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)


if __name__ == '__main__':

    main(args, opt)
