import time
import copy
import numpy as np
import os
import sys

sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from cords.utils.models import *
from cords.utils.custom_dataset import load_dataset_custom, CustomAdvDataset, svhn_mean, svhn_std, clamp
from torch.utils.data import Subset
from cords.utils.config_utils import load_config_data
import os.path as osp
from cords.selectionstrategies.supervisedlearning import OMPGradMatchStrategy, RandomStrategy, CRAIGStrategy
from ray import tune


# TODO: Get rid of explicitly defining the std, mu, and upper and lower_bound
mu  = 0.5
std = 0.5

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

class TrainRobustClassifier:
    def __init__(self, config_file):
        self.config_file = config_file
        self.configdata = load_config_data(self.config_file)
        assert self.configdata['train_args']['robust_train'] == True

    """
    #Model Creation
    """

    def create_model(self):

        if self.configdata['model']['architecture'] == 'ResNet18':
            model = ResNet18(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MnistNet':
            model = MnistNet()
        elif self.configdata['model']['architecture'] == 'ResNet164':
            model = ResNet164(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MobileNet':
            model = MobileNet(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MobileNetV2':
            model = MobileNetV2(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MobileNet2':
            model = MobileNet2(output_size=self.configdata['model']['numclasses'])
        model = model.to(self.configdata['train_args']['device'])
        return model

    """
    #Loss Type, Optimizer and Learning Rate Scheduler
    """

    def loss_function(self):
        if self.configdata['loss']['type'] == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        return criterion, criterion_nored

    def optimizer(self, model):

        if self.configdata['optimizer']['type'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.configdata['optimizer']['lr'],
                                  momentum=self.configdata['optimizer']['momentum'],
                                  weight_decay=self.configdata['optimizer']['weight_decay'])
        elif self.configdata['optimizer']['type'] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.configdata['optimizer']['lr'])
        elif self.configdata['optimizer']['type'] == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.configdata['optimizer']['lr'])

        return optimizer

    def generate_cumulative_timing(self, mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing / 3600

    def save_ckpt(self, state, ckpt_path):
        torch.save(state, ckpt_path)

    def load_ckp(self, ckpt_path, model, optimizer):
        checkpoint  = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss    = checkpoint['loss']
        metrics = checkpoint['metrics']
        return start_epoch, model, optimizer, loss, metrics

    def get_perturbations(self, model, dataloader, criterion, epsilon, alpha, attack_iters, delta_init, std,
                          attack_type='PGD'):

        if attack_type == 'PGD':

            # TODO: Get rid of explicitly defining the shape of images, std, lower, and upper limit
            deltas  = []
            epsilon = (epsilon / 255.) / std
            alpha   = (alpha / 255.) / std

            for batch_idx, (inputs, targets) in enumerate(dataloader):

                with torch.enable_grad():
                    model.train()
                    input_var = inputs.clone().cuda()
                    target_var = targets.clone().cuda()

                    delta = torch.zeros_like(input_var).cuda()
                    if delta_init == 'random':
                        delta[:, :, :, :].uniform_(-epsilon, epsilon)
                        delta.data = clamp(delta.data, lower_limit - input_var, upper_limit - input_var)
                    delta.requires_grad = True

                    for j in range(attack_iters):
                        output = model(input_var + delta)
                        loss = criterion(output, target_var)
                        loss.backward()

                        grad       = delta.grad.detach()
                        delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                        delta.data = clamp(delta.data, lower_limit - input_var, upper_limit - input_var)
                        delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                        delta.grad.zero_()

                    delta = delta.detach()

                with torch.no_grad():
                    deltas.append(delta.cpu())

                del delta, input_var, target_var
                torch.cuda.empty_cache()

            deltas = torch.cat(deltas, dim=0)

            return deltas

        else:
            raise NotImplementedError(f"Attack type {attack_type} undefined!")

    def attack_pgd(self, model, inputs, targets, criterion, criterion_nored, epsilon, alpha, attack_iters, restarts):
        """
        PGD attack for evaluating the CNN robust performance.
        """
        max_loss  = torch.zeros(inputs.shape[0]).cuda()
        max_delta = torch.zeros_like(inputs).cuda()

        for zz in range(restarts):
            delta = torch.zeros_like(inputs).cuda()

            delta[:, :, :, :].uniform_(-epsilon, epsilon)

            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
            delta.requires_grad = True

            for _ in range(attack_iters):
                output = model(inputs + delta)
                index = torch.where(output.max(1)[1] == targets)
                if len(index[0]) == 0:
                    break
                loss = criterion(output, targets)
                loss.backward()

                grad = delta.grad.detach()
                d = delta[index[0], :, :, :]
                g = grad[index[0], :, :, :]
                d = d + alpha * g / (norms(g) + 1e-10)
                d = clamp(d, lower_limit - inputs[index[0], :, :, :], upper_limit - inputs[index[0], :, :, :])
                d = d * epsilon / norms(d.detach()).clamp(min=epsilon)
                delta.data[index[0], :, :, :] = d
                delta.grad.zero_()

            all_loss = criterion_nored(model(inputs + delta), targets).detach()

            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return max_delta

    def train(self):
        """
        #General Training Loop with Data Selection Strategies
        """
        # Loading the Dataset
        trainset, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                   self.configdata['dataset']['name'],
                                                                   self.configdata['dataset']['feature'])

        trainadvset = CustomAdvDataset(trainset)
        validadvset = CustomAdvDataset(validset)

        N              = len(trainset)
        trn_batch_size = 128
        val_batch_size = 128
        tst_batch_size = 128

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=True, pin_memory=True)
        valloader   = torch.utils.data.DataLoader(validset, batch_size=val_batch_size, shuffle=False, pin_memory=True)
        testloader  = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size, shuffle=False, pin_memory=True)

        if self.configdata['train_args']['robust_train']:
            trainadvloader         = torch.utils.data.DataLoader(trainadvset, batch_size=trn_batch_size, shuffle=False, pin_memory=True)
            trainadvloader_no_pert = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False, pin_memory=True)
            valadvloader           = torch.utils.data.DataLoader(validadvset, batch_size=val_batch_size, shuffle=False, pin_memory=True)

        # Budget for subset selection
        bud = int(self.configdata['dss_strategy']['fraction'] * N)
        print("Budget, fraction and N:", bud, self.configdata['dss_strategy']['fraction'], N)

        # Subset Selection and creating the subset data loader
        start_idxs = np.random.choice(N, size=bud, replace=False)
        idxs       = start_idxs
        data_sub   = Subset(trainset, idxs)

        subset_trnloader = torch.utils.data.DataLoader(data_sub,
                                                       batch_size=self.configdata['dataloader']['batch_size'],
                                                       shuffle=self.configdata['dataloader']['shuffle'],
                                                       pin_memory=self.configdata['dataloader']['pin_memory'])

        # Variables to store accuracies
        gammas         = torch.ones(len(idxs)).to(self.configdata['train_args']['device'])
        substrn_losses = list()
        trn_losses     = list()
        val_losses     = list()
        tst_losses     = list()
        subtrn_losses  = list()
        timing         = list()
        trn_acc        = list()
        val_acc        = list()
        tst_acc        = list()
        subtrn_acc     = list()

        if self.configdata['train_args']['robust_train']:
            val_robust_losses = list()
            tst_robust_losses = list()
            val_robust_acc    = list()
            tst_robust_acc    = list()

        # Results logging file
        print_every  = self.configdata['train_args']['print_every']
        results_dir  = osp.abspath(osp.expanduser(self.configdata['train_args']['results_dir']))
        all_logs_dir = os.path.join(results_dir,
                                    self.configdata['dss_strategy']['type'],
                                    self.configdata['dataset']['name'],
                                    str(self.configdata['dss_strategy']['fraction']),
                                    str(self.configdata['dss_strategy']['select_every']))

        os.makedirs(all_logs_dir, exist_ok=True)
        path_logfile = os.path.join(all_logs_dir, self.configdata['dataset']['name'] + '.txt')
        logfile      = open(path_logfile, 'w')

        checkpoint_dir = osp.abspath(osp.expanduser(self.configdata['ckpt']['dir']))
        ckpt_dir       = os.path.join(checkpoint_dir,
                                      self.configdata['dss_strategy']['type'],
                                      self.configdata['dataset']['name'], 
                                      str(self.configdata['dss_strategy']['fraction']), 
                                      str(self.configdata['dss_strategy']['select_every']))

        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        model  = self.create_model()
        model1 = self.create_model()

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        # Getting the optimizer
        optimizer = self.optimizer(model)

        if self.configdata['dss_strategy']['type'] == 'GradMatch':
            # OMPGradMatch Selection strategy
            setf_model = OMPGradMatchStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'], num_cls, True,
                                              'PerClassPerGradient',
                                              valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
        elif self.configdata['dss_strategy']['type'] == 'GradMatchPB':
            setf_model = OMPGradMatchStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'], num_cls, True, 'PerBatch',
                                              valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)

        elif self.configdata['dss_strategy']['type'] == 'CRAIG':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerClass')

        elif self.configdata['dss_strategy']['type'] == 'CRAIGPB':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerBatch')

        elif self.configdata['dss_strategy']['type'] == 'CRAIG-Warm':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerClass')
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs  = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'CRAIGPB-Warm':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerBatch')
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs  = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'Random':
            # Random Selection strategy
            setf_model = RandomStrategy(trainadvloader, online=False)

        elif self.configdata['dss_strategy']['type'] == 'GradMatch-Warm':
            # OMPGradMatch Selection strategy
            setf_model = OMPGradMatchStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'],
                                              num_cls, True, 'PerClassPerGradient',
                                              valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)

            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs  = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'GradMatchPB-Warm':
            # OMPGradMatch Selection strategy
            setf_model = OMPGradMatchStrategy(trainadvloader, valadvloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'],
                                              num_cls, True, 'PerBatch', valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)

            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs  = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'Random-Warm':
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs  = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        print("=======================================", file=logfile)

        if self.configdata['ckpt']['is_load'] == True:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckp(checkpoint_path, model, optimizer)
            print("Loading saved checkpoint model at epoch " + str(start_epoch))
            for arg in load_metrics.keys():
                if arg == "val_loss":
                    val_losses = load_metrics['val_loss']
                if arg == "val_acc":
                    val_acc = load_metrics['val_acc']
                if arg == "val_robust_acc":
                    val_acc = load_metrics['val_robust_acc']
                if arg == "tst_loss":
                    tst_losses = load_metrics['tst_loss']
                if arg == "tst_acc":
                    tst_acc = load_metrics['tst_acc']
                if arg == "tst_robust_acc":
                    tst_acc = load_metrics['tst_robust_acc']
                if arg == "trn_loss":
                    trn_losses = load_metrics['trn_loss']
                if arg == "trn_acc":
                    trn_acc = load_metrics['trn_acc']
                if arg == "subtrn_loss":
                    subtrn_losses = load_metrics['subtrn_loss']
                if arg == "subtrn_acc":
                    subtrn_acc = load_metrics['subtrn_acc']
                if arg == "time":
                    timing = load_metrics['time']
        else:
            start_epoch = 0

        # Adjust the learning rate (TODO: replace this with an automated scheduler)
        lr_drop_epochs = [75, 90, 100]

        for i in range(start_epoch, self.configdata['train_args']['num_epochs']):
            subtrn_loss           = 0
            subtrn_correct        = 0
            subtrn_total          = 0
            subset_selection_time = 0

            lr = self.configdata['optimizer']['lr']

            for lr_drop_epoch in lr_drop_epochs:
                if i >= lr_drop_epoch:
                    lr *= 0.1

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if self.configdata['dss_strategy']['type'] in ['Random']:
                pass

            elif (self.configdata['dss_strategy']['type'] in ['GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']) and (((i + 1) % self.configdata['dss_strategy']['select_every']) == 0):

                start_time        = time.time()
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict        = copy.deepcopy(model.state_dict())

                trn_pertbs = self.get_perturbations(model,
                                                    dataloader=trainadvloader_no_pert,
                                                    criterion=criterion,
                                                    epsilon=self.configdata['train_args']['epsilon'],
                                                    alpha=self.configdata['train_args']['alpha'],
                                                    attack_iters=self.configdata['train_args']['attack_iters'],
                                                    delta_init=self.configdata['train_args']['delta_init'],
                                                    std=std,
                                                    attack_type=self.configdata['train_args']['attack_type'])

                trainadvset.update_deltas(trn_pertbs)

                val_pertbs = self.get_perturbations(model,
                                                    dataloader=valloader,
                                                    criterion=criterion,
                                                    epsilon=self.configdata['train_args']['epsilon'],
                                                    alpha=self.configdata['train_args']['alpha'],
                                                    attack_iters=self.configdata['train_args']['attack_iters'],
                                                    delta_init=self.configdata['train_args']['delta_init'],
                                                    std=std,
                                                    attack_type=self.configdata['train_args']['attack_type'])

                validadvset.update_deltas(val_pertbs)

                subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
                model.load_state_dict(cached_state_dict)
                idxs = subset_idxs

                if self.configdata['dss_strategy']['type'] in ['GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']:
                    gammas = torch.from_numpy(np.array(gammas)).to(self.configdata['train_args']['device']).to(torch.float32)

                subset_selection_time += (time.time() - start_time)

            elif (self.configdata['dss_strategy']['type'] in ['GradMatch-Warm', 'GradMatchPB-Warm', 'CRAIG-Warm', 'CRAIGPB-Warm']):
                start_time = time.time()
                if ((i % self.configdata['dss_strategy']['select_every'] == 0) and (i >= kappa_epochs)):

                    cached_state_dict = copy.deepcopy(model.state_dict())
                    clone_dict        = copy.deepcopy(model.state_dict())

                    trn_pertbs = self.get_perturbations(model,
                                                        dataloader=trainadvloader_no_pert,
                                                        criterion=criterion,
                                                        epsilon=self.configdata['train_args']['epsilon'],
                                                        alpha=self.configdata['train_args']['alpha'],
                                                        attack_iters=self.configdata['train_args']['attack_iters'],
                                                        delta_init=self.configdata['train_args']['delta_init'],
                                                        std=std,
                                                        attack_type=self.configdata['train_args']['attack_type'])

                    trainadvset.update_deltas(trn_pertbs)

                    val_pertbs = self.get_perturbations(model,
                                                        dataloader=valloader,
                                                        criterion=criterion,
                                                        epsilon=self.configdata['train_args']['epsilon'],
                                                        alpha=self.configdata['train_args']['alpha'],
                                                        attack_iters=self.configdata['train_args']['attack_iters'],
                                                        delta_init=self.configdata['train_args']['delta_init'],
                                                        std=std,
                                                        attack_type=self.configdata['train_args']['attack_type'])

                    validadvset.update_deltas(val_pertbs)

                    subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
                    model.load_state_dict(cached_state_dict)
                    idxs = subset_idxs
                    if self.configdata['dss_strategy']['type'] in ['GradMatch-Warm', 'GradMatchPB-Warm', 'CRAIG-Warm', 'CRAIGPB-Warm']:
                        gammas = torch.from_numpy(np.array(gammas)).to(self.configdata['train_args']['device']).to(torch.float32)

                subset_selection_time += (time.time() - start_time)

            elif self.configdata['dss_strategy']['type'] in ['Random-Warm']:
                pass

            data_sub         = Subset(trainset, idxs)
            subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=trn_batch_size, shuffle=False, pin_memory=True)

            model.train()
            batch_wise_indices = list(subset_trnloader.batch_sampler)

            if self.configdata['dss_strategy']['type'] in ['CRAIG', 'CRAIGPB', 'GradMatch', 'GradMatchPB']:
                start_time = time.time()

                epsilon = (self.configdata['train_args']['epsilon'] / 255.) / std
                alpha   = (self.configdata['train_args']['alpha'] / 255.) / std

                # TODO: factor out the repititive training loops in a single function (for subset and full trainloaders)
                for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)
                    delta           = torch.zeros_like(inputs).to(self.configdata['train_args']['device'])

                    if self.configdata['train_args']['delta_init'] == 'random':
                        delta[:, :, :, :].uniform_(-epsilon, epsilon)
                        delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)

                    delta.requires_grad = True

                    for _ in range(self.configdata['train_args']['attack_iters']):
                        output = model(inputs + delta)
                        loss = criterion(output, targets)
                        loss.backward()

                        grad       = delta.grad.detach()
                        delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                        delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
                        delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                        delta.grad.zero_()

                    delta = delta.detach()

                    optimizer.zero_grad()
                    outputs = model(inputs + delta)
                    losses  = criterion_nored(outputs, targets)
                    loss    = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (gammas[batch_wise_indices[batch_idx]].sum())

                    loss.backward()
                    optimizer.step()

                    _, predicted    = outputs.max(1)
                    subtrn_total   += targets.size(0)
                    subtrn_loss    += loss.item()
                    subtrn_correct += predicted.eq(targets).sum().item()

                train_time = time.time() - start_time

            elif self.configdata['dss_strategy']['type'] in ['CRAIGPB-Warm', 'CRAIG-Warm', 'GradMatch-Warm', 'GradMatchPB-Warm']:
                start_time = time.time()

                epsilon = (self.configdata['train_args']['epsilon'] / 255.) / std
                alpha   = (self.configdata['train_args']['alpha'] / 255.) / std

                if i < full_epochs:
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)

                        delta = torch.zeros_like(inputs).to(self.configdata['train_args']['device'])

                        if self.configdata['train_args']['delta_init'] == 'random':
                            delta[:, :, :, :].uniform_(-epsilon, epsilon)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)

                        delta.requires_grad = True

                        for _ in range(self.configdata['train_args']['attack_iters']):
                            output = model(inputs + delta)
                            loss = criterion(output, targets)
                            loss.backward()

                            grad = delta.grad.detach()
                            delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
                            delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                            delta.grad.zero_()

                        delta = delta.detach()

                        optimizer.zero_grad()
                        outputs = model(inputs + delta)
                        loss    = criterion(outputs, targets)

                        loss.backward()
                        optimizer.step()

                        _, predicted    = outputs.max(1)
                        subtrn_total   += targets.size(0)
                        subtrn_loss    += loss.item()
                        subtrn_correct += predicted.eq(targets).sum().item()

                elif i >= kappa_epochs:
                    for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)

                        delta = torch.zeros_like(inputs).to(self.configdata['train_args']['device'])

                        if self.configdata['train_args']['delta_init'] == 'random':
                            delta[:, :, :, :].uniform_(-epsilon, epsilon)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)

                        delta.requires_grad = True

                        for _ in range(self.configdata['train_args']['attack_iters']):
                            output = model(inputs + delta)
                            loss   = criterion(output, targets)
                            loss.backward()

                            grad = delta.grad.detach()
                            delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
                            delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                            delta.grad.zero_()

                        delta = delta.detach()

                        optimizer.zero_grad()
                        outputs = model(inputs + delta)
                        losses  = criterion_nored(outputs, targets)
                        loss    = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (gammas[batch_wise_indices[batch_idx]].sum())

                        loss.backward()
                        optimizer.step()

                        _, predicted    = outputs.max(1)
                        subtrn_total   += targets.size(0)
                        subtrn_loss    += loss.item()
                        subtrn_correct += predicted.eq(targets).sum().item()

                train_time = time.time() - start_time

            elif self.configdata['dss_strategy']['type'] in ['Random']:
                start_time = time.time()

                epsilon = (self.configdata['train_args']['epsilon'] / 255.) / std
                alpha   = (self.configdata['train_args']['alpha'] / 255.) / std

                for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)

                    delta = torch.zeros_like(inputs).to(self.configdata['train_args']['device'])

                    if self.configdata['train_args']['delta_init'] == 'random':
                        delta[:, :, :, :].uniform_(-epsilon, epsilon)
                        delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)

                    delta.requires_grad = True

                    for _ in range(self.configdata['train_args']['attack_iters']):
                        output = model(inputs + delta)
                        loss = criterion(output, targets)
                        loss.backward()

                        grad       = delta.grad.detach()
                        delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                        delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
                        delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                        delta.grad.zero_()

                    delta = delta.detach()

                    optimizer.zero_grad()
                    outputs = model(inputs + delta)
                    loss    = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

                    _, predicted   = outputs.max(1)
                    subtrn_total   += targets.size(0)
                    subtrn_loss    += loss.item()
                    subtrn_correct += predicted.eq(targets).sum().item()

                train_time = time.time() - start_time

            elif self.configdata['dss_strategy']['type'] in ['Random-Warm']:
                start_time = time.time()

                epsilon = (self.configdata['train_args']['epsilon'] / 255.) / std
                alpha   = (self.configdata['train_args']['alpha'] / 255.) / std

                if i < full_epochs:
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)

                        delta = torch.zeros_like(inputs).to(self.configdata['train_args']['device'])

                        if self.configdata['train_args']['delta_init'] == 'random':
                            delta[:, :, :, :].uniform_(-epsilon, epsilon)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)

                        delta.requires_grad = True

                        for _ in range(self.configdata['train_args']['attack_iters']):
                            output = model(inputs + delta)
                            loss = criterion(output, targets)
                            loss.backward()

                            grad = delta.grad.detach()
                            delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
                            delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                            delta.grad.zero_()

                        delta = delta.detach()

                        optimizer.zero_grad()
                        outputs = model(inputs + delta)
                        loss    = criterion(outputs, targets)

                        loss.backward()
                        optimizer.step()

                        _, predicted    = outputs.max(1)
                        subtrn_total   += targets.size(0)
                        subtrn_loss    += loss.item()
                        subtrn_correct += predicted.eq(targets).sum().item()

                elif i >= kappa_epochs:
                    for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)

                        delta = torch.zeros_like(inputs).to(self.configdata['train_args']['device'])

                        if self.configdata['train_args']['delta_init'] == 'random':
                            delta[:, :, :, :].uniform_(-epsilon, epsilon)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)

                        delta.requires_grad = True

                        for _ in range(self.configdata['train_args']['attack_iters']):
                            output = model(inputs + delta)
                            loss = criterion(output, targets)
                            loss.backward()

                            grad = delta.grad.detach()
                            delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                            delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
                            delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                            delta.grad.zero_()

                        delta = delta.detach()

                        optimizer.zero_grad()
                        outputs = model(inputs + delta)
                        loss    = criterion(outputs, targets)

                        loss.backward()
                        optimizer.step()

                        _, predicted    = outputs.max(1)
                        subtrn_total   += targets.size(0)
                        subtrn_loss    += loss.item()
                        subtrn_correct += predicted.eq(targets).sum().item()

                train_time = time.time() - start_time

            elif self.configdata['dss_strategy']['type'] in ['Full']:
                start_time = time.time()

                epsilon = (self.configdata['train_args']['epsilon'] / 255.) / std
                alpha   = (self.configdata['train_args']['alpha'] / 255.) / std

                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)

                    delta = torch.zeros_like(inputs).to(self.configdata['train_args']['device'])

                    if self.configdata['train_args']['delta_init'] == 'random':
                        delta[:, :, :, :].uniform_(-epsilon, epsilon)
                        delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)

                    delta.requires_grad = True

                    for _ in range(self.configdata['train_args']['attack_iters']):
                        output = model(inputs + delta)
                        loss = criterion(output, targets)
                        loss.backward()

                        grad       = delta.grad.detach()
                        delta.data = delta.data + alpha * grad / (norms(grad) + 1e-10)
                        delta.data = clamp(delta.data, lower_limit - inputs, upper_limit - inputs)
                        delta.data = delta.data * epsilon / norms(delta.detach()).clamp(min=epsilon)

                        delta.grad.zero_()

                    delta = delta.detach()

                    optimizer.zero_grad()

                    outputs = model(inputs + delta)
                    loss    = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

                    _, predicted    = outputs.max(1)
                    subtrn_total   += targets.size(0)
                    subtrn_loss    += loss.item()
                    subtrn_correct += predicted.eq(targets).sum().item()

                train_time = time.time() - start_time

            timing.append(train_time + subset_selection_time)
            print_args = self.configdata['train_args']['print_args']
            # print("Epoch timing is: " + str(timing[-1]))

            if ((i + 1) % print_every == 0):
                trn_loss    = 0
                trn_correct = 0
                trn_total   = 0
                val_loss    = 0
                val_correct = 0
                val_total   = 0
                tst_correct = 0
                tst_total   = 0
                tst_loss    = 0
                epsilon     = (self.configdata['train_args']['epsilon'] / 255.) / std  # TODO
                alpha       = (2 / 255.) / std  # TODO

                val_pgd_loss    = 0
                val_pgd_correct = 0
                val_pgd_total   = 0
                tst_pgd_loss    = 0
                tst_pgd_correct = 0
                tst_pgd_total   = 0

                model.eval()

                if "trn_loss" in print_args:
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(trainloader):

                            inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)
                            outputs         = model(inputs)
                            loss            = criterion(outputs, targets)
                            trn_loss       += loss.item()
                            trn_losses.append(trn_loss)

                            if "trn_acc" in print_args:
                                _, predicted = outputs.max(1)
                                trn_total   += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()
                                trn_acc.append(trn_correct / trn_total)

                if "val_loss" in print_args:
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(valloader):

                            inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)
                            outputs         = model(inputs)
                            loss            = criterion(outputs, targets)
                            val_loss       += loss.item()
                            val_losses.append(val_loss)

                            if "val_acc" in print_args:
                                _, predicted = outputs.max(1)
                                val_total   += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                                val_acc.append(val_correct / val_total)

                if "tst_loss" in print_args:
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(testloader):

                            inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)
                            outputs         = model(inputs)
                            loss            = criterion(outputs, targets)
                            tst_loss       += loss.item()
                            tst_losses.append(tst_loss)

                            if "tst_acc" in print_args:
                                _, predicted = outputs.max(1)
                                tst_total   += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()
                                tst_acc.append(tst_correct / tst_total)

                if "val_robust_loss" in print_args:

                    for _, (inputs, targets) in enumerate(valloader):

                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)
                        pgd_delta       = self.attack_pgd(model, inputs, targets, criterion, criterion_nored, epsilon, alpha, attack_iters=50, restarts=10)  # TODO

                        with torch.no_grad():
                            output        = model(inputs + pgd_delta)
                            loss          = criterion(output, targets)
                            val_pgd_loss += loss.item()
                            val_robust_losses.append(val_pgd_loss)

                            if "val_robust_acc" in print_args:
                                val_pgd_correct += (output.max(1)[1] == targets).sum().item()
                                val_pgd_total   += targets.size(0)
                                val_robust_acc.append(val_pgd_correct / val_pgd_total)

                if "tst_robust_loss" in print_args:

                    for _, (inputs, targets) in enumerate(testloader):

                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)
                        pgd_delta       = self.attack_pgd(model, inputs, targets, criterion, criterion_nored, epsilon, alpha, attack_iters=50, restarts=10)  # TODO

                        with torch.no_grad():
                            output        = model(inputs + pgd_delta)
                            loss          = criterion(output, targets)
                            tst_pgd_loss += loss.item()
                            tst_robust_losses.append(tst_pgd_loss)
                            if "tst_robust_acc" in print_args:
                                tst_pgd_correct += (output.max(1)[1] == targets).sum().item()
                                tst_pgd_total   += targets.size(0)
                                tst_robust_acc.append(tst_pgd_correct / tst_pgd_total)

                if "subtrn_acc" in print_args:
                    subtrn_acc.append(subtrn_correct / subtrn_total)

                if "subtrn_losses" in print_args:
                    subtrn_losses.append(subtrn_loss)

                print_str = "Epoch: " + str(i + 1)

                for arg in print_args:

                    if arg == "val_loss":
                        print_str += " , " + "Validation Loss: " + str(val_losses[-1])

                    if arg == "val_acc":
                        print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])

                    if arg == "val_robust_acc":
                        print_str += " , " + "Robust Validation Accuracy: " + str(val_robust_acc[-1])

                    if arg == "tst_loss":
                        print_str += " , " + "Test Loss: " + str(tst_losses[-1])

                    if arg == "tst_acc":
                        print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])

                    if arg == "tst_robust_acc":
                        print_str += " , " + "Robust Test Accuracy: " + str(tst_robust_acc[-1])

                    if arg == "trn_loss":
                        print_str += " , " + "Training Loss: " + str(trn_losses[-1])

                    if arg == "trn_acc":
                        print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])

                    if arg == "subtrn_loss":
                        print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])

                    if arg == "subtrn_acc":
                        print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])

                    if arg == "time":
                        print_str += " , " + "Timing: " + str(timing[-1])

                # report metric to ray for hyperparameter optimization
                if 'report_tune' in self.configdata and self.configdata['report_tune']:
                    tune.report(mean_accuracy=val_acc[-1])

                print(print_str, flush=True)

            if ((i + 1) % self.configdata['ckpt']['save_every'] == 0) and self.configdata['ckpt']['is_save'] == True:

                metric_dict = {}

                for arg in print_args:
                    if arg == "val_loss":
                        metric_dict['val_loss']        = val_losses
                    if arg == "val_acc":
                        metric_dict['val_acc']         = val_acc
                    if arg == "val_robust_loss":
                        metric_dict['val_robust_loss'] = val_robust_losses
                    if arg == "val_robust_acc":
                        metric_dict['val_robust_acc']  = val_robust_acc
                    if arg == "tst_loss":
                        metric_dict['tst_loss']        = tst_losses
                    if arg == "tst_acc":
                        metric_dict['tst_acc']         = tst_acc
                    if arg == "tst_robust_loss":
                        metric_dict['tst_robust_loss'] = tst_robust_losses
                    if arg == "tst_robust_acc":
                        metric_dict['tst_robust_acc']  = tst_robust_acc
                    if arg == "trn_loss":
                        metric_dict['trn_loss']        = trn_losses
                    if arg == "trn_acc":
                        metric_dict['trn_acc']         = trn_acc
                    if arg == "subtrn_loss":
                        metric_dict['subtrn_loss']     = subtrn_losses
                    if arg == "subtrn_acc":
                        metric_dict['subtrn_acc']      = subtrn_acc
                    if arg == "time":
                        metric_dict['time']            = timing

                ckpt_state = {
                    'epoch': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': self.loss_function(),
                    'metrics': metric_dict
                }

                # save checkpoint
                self.save_ckpt(ckpt_state, checkpoint_path)
                print("Model checkpoint saved at epoch " + str(i + 1), flush=True)

        print(self.configdata['dss_strategy']['type'] + " Selection Run---------------------------------")
        print("Final SubsetTrn:", subtrn_loss)
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                print("Validation Loss and Accuracy: ", val_losses[-1], np.array(val_acc).max())
            else:
                print("Validation Loss: ", val_losses[-1])

        if "val_robust_loss" in print_args:
            if "val_robust_acc" in print_args:
                print("Robust Validation Loss and Accuracy: ", val_robust_losses[-1], np.array(val_robust_acc).max())
            else:
                print("Robust Validation Loss: ", val_robust_losses[-1])

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                print("Test Data Loss and Accuracy: ", tst_losses[-1], np.array(tst_acc).max())
            else:
                print("Test Data Loss: ", tst_losses[-1])

        if "tst_robust_loss" in print_args:
            if "tst_robust_acc" in print_args:
                print("Robust Test Data Loss and Accuracy: ", tst_robust_losses[-1], np.array(tst_robust_acc).max())
            else:
                print("Robust Test Data Loss: ", tst_robust_losses[-1])

        print('-----------------------------------')
        print(self.configdata['dss_strategy']['type'], file=logfile)
        print('---------------------------------------------------------------------', file=logfile)

        if "val_acc" in print_args:
            val_str = "Validation Accuracy, "
            for val in val_acc:
                val_str = val_str + " , " + str(val)
            print(val_str, file=logfile)

        if "val_robust_acc" in print_args:
            val_str = "Robust Validation Accuracy, "
            for val in val_robust_acc:
                val_str = val_str + " , " + str(val)
            print(val_str, file=logfile)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy, "
            for tst in tst_acc:
                tst_str = tst_str + " , " + str(tst)
            print(tst_str, file=logfile)

        if "tst_robust_acc" in print_args:
            tst_str = "Robust Test Accuracy, "
            for tst in tst_robust_acc:
                tst_str = tst_str + " , " + str(tst)
            print(tst_str, file=logfile)

        if "time" in print_args:
            time_str = "Time, "
            for t in timing:
                time_str = time_str + " , " + str(t)
            print(timing, file=logfile)

        omp_timing = np.array(timing)
        omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        print("Total time taken by " + self.configdata['dss_strategy']['type'] + " = " + str(omp_cum_timing[-1]))
        logfile.close()

    def eval(self):
        """
        #General Testing Loop
        """
        # Loading the Dataset
        _, _, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                     self.configdata['dataset']['name'],
                                                     self.configdata['dataset']['feature'])

        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, pin_memory=True)

        # Model Creation
        model = self.create_model()

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        self.bud = 1000

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        checkpoint_dir  = osp.abspath(osp.expanduser(self.configdata['ckpt']['dir']))
        ckpt_dir        = os.path.join(checkpoint_dir,self.configdata['dss_strategy']['type'], self.configdata['dataset']['name'], str(self.configdata['dss_strategy']['fraction']), str(self.configdata['dss_strategy']['select_every']))
        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')

        if self.configdata['ckpt']['is_load'] == True:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckp(checkpoint_path, model, optimizer)
            print("Loading saved checkpoint model at epoch " + str(start_epoch))

        tst_correct     = 0
        tst_total       = 0
        tst_loss        = 0
        tst_pgd_correct = 0
        tst_pgd_total   = 0
        tst_pgd_loss    = 0
        eps             = self.configdata['train_args']['epsilon']
        epsilon         = (eps / 255.) / std  # TODO
        alpha           = (2 / 255.) / std  # TODO
        model.cuda()
        model.eval()

        # Standard accuracy (TODO: ask the user if they want to compute this explicitly!)
        if True:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):

                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)

                    outputs    = model(inputs)
                    loss      = criterion(outputs, targets)
                    tst_loss += loss.item()

                    if True:
                        _, predicted = outputs.max(1)
                        tst_total   += targets.size(0)
                        tst_correct += predicted.eq(targets).sum().item()

            print(f'Accuracy: {tst_correct/tst_total}')

        # Robust accuracy (TODO: ask the user if they want to compute this explicitly!)
        if True:

            for _, (inputs, targets) in enumerate(testloader):

                inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(self.configdata['train_args']['device'], non_blocking=True)
                pgd_delta        = self.attack_pgd(model, inputs, targets, criterion, criterion_nored, epsilon, alpha, attack_iters=50, restarts=10)  # TODO

                with torch.no_grad():
                    output = model(inputs + pgd_delta)
                    loss   = criterion(output, targets)
                    tst_pgd_loss += loss.item()

                    if True:
                        tst_pgd_correct += (output.max(1)[1] == targets).sum().item()
                        tst_pgd_total   += targets.size(0)

            print(f'Robust Accuracy for {eps} radius: {tst_pgd_correct / tst_pgd_total}')       