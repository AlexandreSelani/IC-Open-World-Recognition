import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPLoss(nn.Module):
    def __init__(self, alpha=1):
        super(DFPLoss, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets):
        sim_fea2cen = net_out["sim_fea2cen"]
        loss_similarity = self.ce(sim_fea2cen, targets)

        dist_fea2cen = 0.5*(net_out["dis_fea2cen"])**2  #0.5*||d||^2
        batch_size, num_classes = dist_fea2cen.shape
        classes = torch.arange(num_classes, device=targets.device).long()
        labels = targets.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)
        loss_distance = self.alpha * (dist_within.sum()) / batch_size

        loss = loss_similarity + loss_distance

        return {
            "total": loss,
            "similarity": loss_similarity,
            "distance": loss_distance
        }


class DFPLoss2(nn.Module):
    def __init__(self, alpha=1, beta=1, theta=2):
        super(DFPLoss2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.theta = theta  # input fea in threshold

    def forward(self, net_out, targets):
        sim_fea2cen = net_out["sim_fea2cen"]
        dist_fea2cen = 0.5*(net_out["dis_fea2cen"])**2
        dist_gen2cen = 0.5*(net_out["dis_gen2cen"])**2
        dist_gen2ori = 0.5*(net_out["dis_gen2ori"])**2
        thresholds = net_out["thresholds"]  # [class_num]

        # classification loss for input data
        loss_similarity = self.ce(sim_fea2cen, targets)

        #  distance loss for input data
        batch_size, num_classes = dist_fea2cen.shape
        classes = torch.arange(num_classes, device=targets.device).long()
        labels = targets.unsqueeze(1).expand(batch_size, num_classes)  # [batch,class]
        mask = (labels.eq(classes.expand(batch_size, num_classes))).float()  # [batch,class]
        dist_within = dist_fea2cen * mask  # [batch,class] distance to centroids
        mask_in = (dist_within <= thresholds.unsqueeze(dim=0)).float()
        mask_out = (dist_within > thresholds.unsqueeze(dim=0)).float()
        # batch_size_in = (mask_in * mask).sum()
        # batch_size_out = (mask_out * mask).sum()
        # print(f"batch: {batch_size} / in {batch_size_in} / {batch_size_out}"
        #       f" ---- equal {(batch_size_in+batch_size_out)==batch_size}")
        loss_distance_in = (dist_within * mask_in).sum(dim=1, keepdim=False)
        loss_distance_in = self.alpha * (loss_distance_in.sum()) / batch_size
        loss_distance_out = (dist_within * mask_out).sum(dim=1, keepdim=False)
        loss_distance_out = self.alpha * self.theta* (loss_distance_out.sum()) /batch_size

        #  distance loss for generated data
        loss_generate = F.relu((2*thresholds.unsqueeze(dim=0) - dist_gen2cen), inplace=True)
        loss_generate = self.beta * (loss_generate.sum()) / (loss_generate.shape[0])


        loss = loss_similarity + loss_distance_in + loss_distance_out + loss_generate
        return {
            "total": loss,
            "similarity": loss_similarity,
            "distance_in": loss_distance_in,
            "distance_out": loss_distance_out,
            "generate": loss_generate
        }


def demo():
    n = 3
    c = 5
    dist_fea2cen = torch.rand([n, c + 1])
    dist_gen2cen = torch.rand([n, c + 1])

    label = torch.empty(3, dtype=torch.long).random_(5)
    print(label)
    loss = DFPLoss(1.)
    netout = {
        "sim_fea2cen": dist_gen2cen,
        "dis_fea2cen": dist_fea2cen
    }
    dist_loss = loss(netout, label)
    print(dist_loss['total'])
    print(dist_loss['similarity'])
    print(dist_loss['distance'])


def demo2():
    n = 3
    c = 5
    dist_fea2cen = torch.rand([n, c])
    dist_gen2cen = torch.rand([n, c])
    thresholds = torch.rand([c])
    dist_gen2ori = torch.rand([c, 1])


    label = torch.empty(3, dtype=torch.long).random_(5)
    # print(label)
    loss = DFPLoss2(1.)
    netout = {
        "sim_fea2cen": dist_gen2cen,
        "dis_fea2cen": dist_fea2cen,
        "dis_gen2cen": dist_gen2cen,
        "dis_gen2ori": dist_gen2ori,
        "thresholds": thresholds
    }
    dist_loss = loss(netout, label)
    # print(dist_loss['total'])
    # print(dist_loss['similarity'])


demo2()
for i in range(100):
    demo2()
