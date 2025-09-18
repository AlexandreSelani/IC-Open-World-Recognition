import torch
import torch.nn.functional as F


def generater_input(inputs, targets, args, repeats=4, reduce=16):
    b, c, w, h = inputs.shape
    if b != args.bs:
        return inputs, targets
    g_data = inputs.repeat(repeats, 1, 1, 1)  # repeat examples 3 times [n*sampler, c]
    g_data = g_data[torch.randperm(g_data.size()[0])]
    g_data = g_data.view(-1, b // reduce, c, w, h)
    g_data = g_data.mean(dim=0, keepdim=False)
    g_label = args.train_class_num * torch.ones(b // reduce, dtype=targets.dtype)
    inputs = torch.cat([inputs, g_data], dim=0)
    targets = torch.cat([targets, g_label], dim=0)

    r = torch.randperm(inputs.size()[0])
    inputs = inputs[r]
    targets = targets[r]
    return inputs, targets

def generater_unknown(inputs, targets, args, repeats=4, reduce=16):
    b, c, w, h = inputs.shape

    number = b // reduce if b == args.bs else b

    g_data = inputs.repeat(repeats, 1, 1, 1)  # repeat examples 3 times [n*sampler, c]
    g_data = g_data[torch.randperm(g_data.size()[0])]
    g_data = g_data.view(-1, number, c, w, h)
    g_data = g_data.mean(dim=0, keepdim=False)
    return g_data



def generater_gap(gap,batchsize=64):
    # generated a random gap doesn't require gradient
    b, c = gap.size()
    mem = gap.clone().detach()
    mem = mem.view(-1)
    mem = mem[torch.randperm(mem.size()[0])]
    mem = mem.view([b, c])
    if batchsize < b:
        mem = mem[:batchsize]
    mem = mem.to(gap.device)
    return mem


def demo():
    n = 16
    c = 5
    inputs = torch.rand([n, 1, 3, 3])
    targets = torch.empty(n, dtype=torch.long).random_(c)
    generater_input(inputs, targets, n)

# demo()

def demo_gap():
    gap = torch.rand([3,6],requires_grad=True)
    generater_gap(gap)

demo_gap()
