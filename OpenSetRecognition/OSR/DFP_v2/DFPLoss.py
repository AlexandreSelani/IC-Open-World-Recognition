import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPLoss(nn.Module):
    """Discriminant Feature Representation loss.

        # better to find a replacement that can combine these two terms.
        Loss = Sigmoid[Dis(x, gt)]+ beta*Sigmoid[-1/(class_num-1)*Sum_i(Dis(x,cls_i))]
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
        """

    def __init__(self, alpha=1, beta=1):
        super(DFPLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion_softamx = nn.CrossEntropyLoss()

    def forward(self, net_out, labels):
        # the distance is based on normalized features
        dist_fea2cen = net_out["dist_fea2cen"]  # [class_num, class_num+1]
        logits = net_out["logits"]

        batch_size, num_classes = dist_fea2cen.shape  # num_classes includes one  place-hold unknown class center
        classes = torch.arange(num_classes, device=labels.device).long()
        targets = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = targets.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)
        dist_between = F.relu(dist_within - dist_fea2cen, inplace=True)  # ensure within_distance greater others
        dist_between = dist_between.sum(dim=1, keepdim=False)
        dist_between = dist_between / (num_classes - 1.0)

        loss_within = (dist_within.sum()) / batch_size
        loss_between = (dist_between.sum()) / batch_size
        loss_classify = self.criterion_softamx(logits, labels)

        # reweight each loss
        loss_within = self.alpha * loss_within
        loss_between = self.beta * loss_between

        loss = loss_within + loss_between + loss_classify

        return {
            "total": loss,
            "within": loss_within,
            "between": loss_between,
            "classify": loss_classify
        }


class DFPLossGeneral(nn.Module):
    """Discriminant Feature Representation loss.

        # better to find a replacement that can combine these two terms.
        Loss = Sigmoid[Dis(x, gt)]+ beta*Sigmoid[-1/(class_num-1)*Sum_i(Dis(x,cls_i))]
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
        """

    def __init__(self, beta=1, sigma=1, gamma=1):
        super(DFPLossGeneral, self).__init__()
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def forward(self, net_out, labels):
        dist_fea2cen = net_out["dist_fea2cen"]
        dist_cen2cen = net_out["dist_cen2cen"]
        dist_gen2cen = net_out["dist_gen2cen"]
        # print(f"Generated example number is {dist_gen2cen.shape[0]} \n")
        # gen_label = dist_gen2cen.shape[1]*torch.ones(dist_gen2cen.shape[0],device=labels.device)
        # dist_fea2cen = torch.cat([dist_fea2cen,dist_gen2cen], dim=0)
        # labels = torch.cat([labels,gen_label], dim=0)

        batch_size, num_classes = dist_fea2cen.shape
        classes = torch.arange(num_classes, device=labels.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)

        # dist_between = (dist * (1 - mask.float()))
        dist_between = F.relu(dist_within - dist_fea2cen, inplace=True)  # ensure within_distance greater others
        dist_between = dist_between.sum(dim=1, keepdim=False)
        # dist_between = dist_between / (num_classes - 1.0)

        loss_within = (dist_within.sum()) / batch_size
        loss_between = self.beta * (dist_between.sum()) / batch_size

        loss_cen2cen = (dist_cen2cen.sum()) / (dist_cen2cen.shape[0] - 1)
        loss_cen2cen = loss_cen2cen / (dist_cen2cen.shape[0] - 1)
        loss_cen2cen = self.sigma * (1.0 - 0.5 * loss_cen2cen)

        #  calculate generated data loss
        batch_size, num_classes = dist_gen2cen.shape
        mask = classes.expand(batch_size, num_classes) == (num_classes - 1)
        dist_within_gen = (dist_gen2cen * mask.float()).sum(dim=1, keepdim=True)
        dist_between_gen = F.relu(dist_within_gen - dist_gen2cen, inplace=True)  # ensure within_distance greater others
        dist_between_gen = dist_between_gen.sum(dim=1, keepdim=False)
        # dist_between = dist_between / (num_classes - 1.0)
        loss_within_gen = self.gamma * (dist_within_gen.sum()) / batch_size
        loss_between_gen = self.gamma * self.beta * (dist_between_gen.sum()) / batch_size

        loss = loss_within + loss_between + loss_within_gen + loss_between_gen + loss_cen2cen

        return {
            "total": loss,
            "within": loss_within,
            "between": loss_between,
            "cen2cen": loss_cen2cen,
            "within_gen": loss_within_gen,
            "between_gen": loss_between_gen
        }


def demo():
    n = 3
    c = 5
    dist_fea2cen = torch.rand([n, c])
    dist_gen2cen = torch.rand([n, c])
    dist_cen2cen = torch.rand([c, c])
    label = torch.Tensor([1, 3, 2])
    loss = DFPLoss(1, 1)
    netout = {
        "dist_fea2cen": dist_fea2cen,
        "dist_cen2cen": dist_cen2cen,
        "dist_gen2cen": dist_gen2cen
    }
    dist_loss = loss(netout, label)
    print(dist_loss['total'])
    print(dist_loss['within'])
    print(dist_loss['between'])
    print(dist_loss['cen2cen'])

    loss2 = DFPLossGeneral(1, 1, 1)

    dist_loss = loss2(netout, label)
    print(dist_loss['total'])
    print(dist_loss['within'])
    print(dist_loss['between'])
    print(dist_loss['cen2cen'])
    print(dist_loss['within_gen'])
    print(dist_loss['between_gen'])

# demo()
