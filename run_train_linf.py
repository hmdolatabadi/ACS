import argparse
from scripts.robust_train_linf import TrainRobustClassifier

parser = argparse.ArgumentParser(description='Robust CIFAR-10 Training')

# TODO: write help for the arguments.
parser.add_argument('--cnfg_dir', type=str, required=True)
parser.add_argument('--ckpt_dir', type=str, required=True)
parser.add_argument('--dataset', type=str, required=False, default='cifar10')

parser.add_argument('--attack_type', type=str, required=True)
parser.add_argument('--attack_iters', type=int, required=True)
parser.add_argument('--epsilon', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)

parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--frac', type=float, required=True)
parser.add_argument('--freq', type=int, required=False, default=20)
parser.add_argument('--kappa', type=float, required=False, default=0.6)
parser.add_argument('--epochs', type=int, required=False, default=120)

args   = parser.parse_args()

config_file = args.cnfg_dir
classifier  = TrainRobustClassifier(config_file)

classifier.configdata['ckpt']['dir']                = args.ckpt_dir
classifier.configdata['train_args']['results_dir']  = args.ckpt_dir
classifier.configdata['train_args']['attack_type']  = args.attack_type
classifier.configdata['train_args']['alpha']        = args.alpha
classifier.configdata['train_args']['delta_init']   = 'random'
classifier.configdata['train_args']['epsilon']      = args.epsilon
classifier.configdata['train_args']['attack_iters'] = args.attack_iters
classifier.configdata['train_args']['print_every']  = 5
classifier.configdata['train_args']['num_epochs']   = args.epochs
classifier.configdata['dataset']['name']            = args.dataset
classifier.configdata['train_args']['print_args']   = ["val_loss", "val_acc", "tst_loss", "tst_acc", "time"]

classifier.configdata['optimizer']['lr']           = args.lr
classifier.configdata['optimizer']['weight_decay'] = 5e-4
classifier.configdata['dss_strategy']['fraction']  = args.frac
classifier.configdata['dss_strategy']['kappa']     = args.kappa
print("kappa: ", args.kappa)

classifier.configdata['dss_strategy']['select_every'] = args.freq
classifier.configdata['ckpt']['is_save'] = True
classifier.configdata['ckpt']['is_load'] = False

classifier.train()

classifier.configdata['ckpt']['is_save'] = False
classifier.configdata['ckpt']['is_load'] = True

classifier.eval()
