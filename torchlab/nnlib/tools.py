import torch
import torch.nn as nn
from torch.autograd import Variable

def adjust_learning_rate2(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):

    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, learning_rate, epoches):
    """Sets the learning rate to the initial LR decayed by 10 every time at 70%, 90% of total epoches"""
    schedule = [0.7, 0.9]
    if float(epoch)/epoches < schedule[0]:
        stage = 0
    elif float(epoch)/epoches < schedule[1]:
        stage = 1
    else:
        stage = 2
    lr = learning_rate * (0.1 ** stage)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_varabile(arr, requires_grad=False, is_cuda=True):
    if type(arr) == np.ndarray:
        tensor = torch.from_numpy(arr)
    else:
        tensor = arr
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            nn.init.constant(m.bias.data, 0)
        # nn.init.normal(m.weight.data, mean=0, std=0.01)
        # nn.init.constant(m.bias.data, val=0)
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform(m.weight.data)
        
def init_with_pretrain(model, pretrained_dict):
    model_dict = model.state_dict()
    if pretrained_dict.keys()[0][0:7]=='module.':
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    elif pretrained_dict.keys()[0][0:7+6]=='model.module.':
        pretrained_dict = {k[7+6:]: v for k, v in pretrained_dict.items() if k[7+6:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

