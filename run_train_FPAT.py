import argparse
from scripts.robust_train_FPAT import TrainRobustClassifier

parser = argparse.ArgumentParser()

parser.add_argument('--arch', type=str, default='resnet50', help='model architecture')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')
parser.add_argument('--dataset_path', type=str, default='~/datasets', help='path to datasets directory')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='./FastLPA_ckpts/GradMatchPB-Warm_imagenet12_0.5.pt')
parser.add_argument('--num_epochs', type=int, required=True, help='number of epochs trained')
parser.add_argument('--batch_size', type=int, default=100, help='number of examples/minibatch')
parser.add_argument('--val_batches', type=int, default=10, help='number of batches to validate on')
parser.add_argument('--log_dir', type=str, default='data/logs')
parser.add_argument('--parallel', type=int, default=1, help='number of GPUs to train on')

parser.add_argument('--lpips_model', type=str, required=False, help='model to use for LPIPS distance')
parser.add_argument('--only_attack_correct', action='store_true', default=True, help='only attack examples that are classified correctly')
parser.add_argument('--randomize_attack', action='store_true', default=False, help='randomly choose an attack at each step')
parser.add_argument('--maximize_attack', action='store_true', default=False, help='choose the attack with maximum loss')

parser.add_argument('--seed', type=int, default=0, help='RNG seed')
parser.add_argument('--continue', default=False, action='store_true', help='continue previous training')
parser.add_argument('--keep_every', type=int, default=1, help='only keep a checkpoint every X epochs')

parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr_schedule', type=str, required=False, help='comma-separated list of epochs when learning rate should drop')
parser.add_argument('--clip_grad', type=float, default=1.0, help='clip gradients to this value')

parser.add_argument('--attack', type=str, action='append', help='attack(s) to harden against',
                    default=["FastLagrangePerceptualAttack(model, bound=0.5, num_iterations=10, lpips_model='alexnet_cifar')"])

parser.add_argument('--cnfg_dir', type=str, required=True)
parser.add_argument('--ckpt_dir', type=str, required=True)

parser.add_argument('--attack_type', type=str, required=True)
parser.add_argument('--attack_iters', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)

parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--frac', type=float, required=True)
parser.add_argument('--freq', type=int, required=False, default=15)

args        = parser.parse_args()

config_file = args.cnfg_dir
classifier  = TrainRobustClassifier(config_file, args)

classifier.configdata['ckpt']['dir']                = args.ckpt_dir
classifier.configdata['train_args']['results_dir']  = args.ckpt_dir
classifier.configdata['train_args']['attack_type']  = args.attack_type
classifier.configdata['train_args']['alpha']        = args.alpha
classifier.configdata['train_args']['attack_iters'] = args.attack_iters
classifier.configdata['train_args']['num_epochs']   = args.num_epochs
classifier.configdata['train_args']['print_args']   = ["val_loss", "val_acc", "tst_loss", "tst_acc", "time"]

classifier.configdata['dataset']['name']            = 'imagenet12' if args.dataset == 'imagenet' else 'cifar10'
classifier.configdata['dataset']['datadir']         = '/data/cephfs/punim0955/data/backdoor'

classifier.configdata['optimizer']['lr']           = args.lr
classifier.configdata['optimizer']['weight_decay'] = 2e-4
classifier.configdata['dss_strategy']['fraction']  = args.frac

classifier.configdata['dss_strategy']['select_every'] = args.freq
classifier.configdata['model']['architecture']        = 'ResNet50'
classifier.configdata['model']['numclasses']          = 12 if args.dataset == 'imagenet' else 10

classifier.train()

classifier.configdata['ckpt']['is_load'] = True
classifier.eval()
