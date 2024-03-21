import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from torchvision import transforms

import loss
import network
from data_load import mnist, svhn, usps


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def digit_load(args_):
    train_bs = args_.batch_size
    svhn_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mnist_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    usps_transforms = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    common_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if args_.dset == 's2m':
        train_source_data = svhn.SVHN('./data/svhn/', split='train', download=True, transform=svhn_transforms)
        test_source_data = svhn.SVHN('./data/svhn/', split='test', download=True, transform=svhn_transforms)
        train_target_data = mnist.MNIST_idx('./data/mnist/', train=True, download=True, transform=mnist_transforms)
        test_target_data = mnist.MNIST('./data/mnist/', train=False, download=True, transform=mnist_transforms)
    elif args_.dset == 'u2m':
        train_source_data = usps.USPS('./data/usps/', train=True, download=True, transform=usps_transforms)
        test_source_data = usps.USPS('./data/usps/', train=False, download=True, transform=usps_transforms)
        train_target_data = mnist.MNIST_idx('./data/mnist/', train=True, download=True, transform=common_transforms)
        test_target_data = mnist.MNIST('./data/mnist/', train=False, download=True, transform=common_transforms)
    elif args_.dset == 'm2u':
        train_source_data = mnist.MNIST('./data/mnist/', train=True, download=True, transform=common_transforms)
        test_source_data = mnist.MNIST('./data/mnist/', train=False, download=True, transform=common_transforms)
        train_target_data = usps.USPS_idx('./data/usps/', train=True, download=True, transform=common_transforms)
        test_target_data = usps.USPS('./data/usps/', train=False, download=True, transform=common_transforms)
    else:
        raise ValueError('dataset cannot be recognized.')

    dset_loaders = {
        "source_train": DataLoader(train_source_data, batch_size=train_bs, shuffle=True, num_workers=args_.worker),
        "source_test": DataLoader(test_source_data, batch_size=train_bs * 2, shuffle=True, num_workers=args_.worker),
        "target_train": DataLoader(train_target_data, batch_size=train_bs, shuffle=True, num_workers=args_.worker),
        # "target_te": DataLoader(train_target_data, batch_size=train_bs, shuffle=False, num_workers=args_.worker),
        "target_test": DataLoader(test_target_data, batch_size=train_bs * 2, shuffle=False, num_workers=args_.worker),
    }
    return dset_loaders


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


def train_source(args_):
    global best_net_f, best_net_b, best_net_c
    dset_loaders = digit_load(args_)
    # set base network
    if args_.dset == 'u2m':
        net_f = network.LeNetBase().cuda()
    elif args_.dset == 'm2u':
        net_f = network.LeNetBase().cuda()
    elif args_.dset == 's2m':
        net_f = network.DTNBase().cuda()
    else:
        raise ValueError('dataset cannot be recognized.')

    net_b = network.FeatBottleneck(type_=args_.classifier, feature_dim=net_f.in_features,
                                   bottleneck_dim=args_.bottleneck).cuda()
    net_c = network.FeatClassifier(type=args_.layer, class_num=args_.class_num, bottleneck_dim=args_.bottleneck).cuda()

    param_group = []
    learning_rate = args_.lr
    for k, v in net_f.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in net_b.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in net_c.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args_.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    net_f.train()
    net_b.train()
    net_c.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except StopIteration:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = net_c(net_b(net_f(inputs_source)))
        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args_.class_num, epsilon=args_.smooth)(outputs_source,
                                                                                                          labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            net_f.eval()
            net_b.eval()
            net_c.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], net_f, net_b, net_c)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], net_f, net_b, net_c)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(args_.dset, iter_num, max_iter,
                                                                                 acc_s_tr, acc_s_te)
            args_.out_file.write(log_str + '\n')
            args_.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_net_f = copy.deepcopy(net_f.state_dict())
                best_net_b = copy.deepcopy(net_b.state_dict())
                best_net_c = copy.deepcopy(net_c.state_dict())

            net_f.train()
            net_b.train()
            net_c.train()

    torch.save(best_net_f, os.path.join(args_.output_dir, "source_F.pt"))
    torch.save(best_net_b, os.path.join(args_.output_dir, "source_B.pt"))
    torch.save(best_net_c, os.path.join(args_.output_dir, "source_C.pt"))

    return net_f, net_b, net_c


def test_target(args_):
    dset_loaders = digit_load(args_)
    # set base network
    if args_.dset == 'u2m':
        net_f = network.LeNetBase().cuda()
    elif args_.dset == 'm2u':
        net_f = network.LeNetBase().cuda()
    elif args_.dset == 's2m':
        net_f = network.DTNBase().cuda()
    else:
        raise ValueError('dataset cannot be recognized.')

    net_b = network.FeatBottleneck(type_=args_.classifier, feature_dim=net_f.in_features,
                                   bottleneck_dim=args_.bottleneck).cuda()
    net_c = network.FeatClassifier(type=args_.layer, class_num=args_.class_num, bottleneck_dim=args_.bottleneck).cuda()

    args_.model_path = args_.output_dir + '/source_F.pt'
    net_f.load_state_dict(torch.load(args_.model_path))
    args_.model_path = args_.output_dir + '/source_B.pt'
    net_b.load_state_dict(torch.load(args_.model_path))
    args_.model_path = args_.output_dir + '/source_C.pt'
    net_c.load_state_dict(torch.load(args_.model_path))
    net_f.eval()
    net_b.eval()
    net_c.eval()

    acc, _ = cal_acc(dset_loaders['test'], net_f, net_b, net_c)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args_.dset, acc)
    args_.out_file.write(log_str + '\n')
    args_.out_file.flush()
    print(log_str + '\n')


def print_args(args_):
    s = "==========================================\n"
    for arg, content in args_.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(args_):
    dset_loaders = digit_load(args_)
    # set base network
    if args_.dset == 'u2m':
        net_f = network.LeNetBase().cuda()
    elif args_.dset == 'm2u':
        net_f = network.LeNetBase().cuda()
    elif args_.dset == 's2m':
        net_f = network.DTNBase().cuda()
    else:
        raise ValueError('dataset cannot be recognized.')

    net_b = network.FeatBottleneck(type_=args_.classifier, feature_dim=net_f.in_features,
                                   bottleneck_dim=args_.bottleneck).cuda()
    net_c = network.FeatClassifier(type=args_.layer, class_num=args_.class_num, bottleneck_dim=args_.bottleneck).cuda()

    args_.model_path = args_.output_dir + '/source_F.pt'
    net_f.load_state_dict(torch.load(args_.model_path))
    args_.model_path = args_.output_dir + '/source_B.pt'
    net_b.load_state_dict(torch.load(args_.model_path))
    args_.model_path = args_.output_dir + '/source_C.pt'
    net_c.load_state_dict(torch.load(args_.model_path))
    net_c.eval()
    for k, v in net_c.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in net_f.named_parameters():
        param_group += [{'params': v, 'lr': args_.lr}]
    for k, v in net_b.named_parameters():
        param_group += [{'params': v, 'lr': args_.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args_.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    # interval_iter = max_iter // args_.interval
    iter_num = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except StopIteration:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args_.cls_par > 0:
            net_f.eval()
            net_b.eval()
            mem_label = obtain_label(dset_loaders['target_te'], net_f, net_b, net_c, args_)
            mem_label = torch.from_numpy(mem_label).cuda()
            net_f.train()
            net_b.train()
        else:
            mem_label = None

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        features_test = net_b(net_f(inputs_test))
        outputs_test = net_c(features_test)

        if args_.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = args_.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args_.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.entropy(softmax_out))
            if args_.gent:
                m_softmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-m_softmax * torch.log(m_softmax + 1e-5))

            im_loss = entropy_loss * args_.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            net_f.eval()
            net_b.eval()
            acc, _ = cal_acc(dset_loaders['test'], net_f, net_b, net_c)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args_.dset, iter_num, max_iter, acc)
            args_.out_file.write(log_str + '\n')
            args_.out_file.flush()
            print(log_str + '\n')
            net_f.train()
            net_b.train()

    if args_.issave:
        torch.save(net_f.state_dict(), os.path.join(args_.output_dir, "target_F_" + args_.savename + ".pt"))
        torch.save(net_b.state_dict(), os.path.join(args_.output_dir, "target_B_" + args_.savename + ".pt"))
        torch.save(net_c.state_dict(), os.path.join(args_.output_dir, "target_C_" + args_.savename + ".pt"))

    return net_f, net_b, net_c


def obtain_label(loader, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='s2m', choices=['u2m', 'm2u', 's2m'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)
