import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F


def plot_feature(net, plotloader, device, dirname, epoch=0, plot_class_num=10, maximum=500, plot_quality=150,
                 normalized=True):
    plot_features = []
    plot_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(plotloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            embed_fea = out["embed_fea"]
            if normalized:
                embed_fea = F.normalize(embed_fea, dim=1, p=2)
            try:
                embed_fea = embed_fea.data.cpu().numpy()
                targets = targets.data.cpu().numpy()
            except:
                embed_fea = embed_fea.data.numpy()
                targets = targets.data.numpy()

            plot_features.append(embed_fea)
            plot_labels.append(targets)

    plot_features = np.concatenate(plot_features, 0)
    plot_labels = np.concatenate(plot_labels, 0)

    net_dict = net.state_dict()
    centroids = net_dict['module.centroids'] if isinstance(net, nn.DataParallel) \
        else net_dict['centroids']
    if normalized:
        centroids = F.normalize(centroids, dim=1, p=2)

    try:
        centroids = centroids.data.cpu().numpy()
    except:
        centroids = centroids.data.numpy()
    # print(centroids)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(plot_class_num):
        features = plot_features[plot_labels == label_idx, :]
        maximum = min(maximum, len(features)) if maximum > 0 else len(features)
        plt.scatter(
            features[0:maximum, 0],
            features[0:maximum, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        # c=colors[label_idx],
        c='black',
        marker="*",
        s=5,
    )
    # currently only support 10 classes, for a good visualization.
    # change plot_class_num would lead to problems.
    legends = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.legend(legends[0:plot_class_num] + ['c'], loc='upper right')

    save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi=plot_quality)
    plt.close()


def plot_distance(net,
                  plotloader: torch.utils.data.DataLoader,
                  device: str,
                  args
                  ) -> dict:
    print("===> Calculating distances...")
    results = {i: {"distances": []} for i in range(args.train_class_num)}
    threshold_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(plotloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            dist_fea2cen = out["dist_fea2cen"]  # [n, class_num]


            for i in range(dist_fea2cen.shape[0]):
                label = targets[i]
                dist = dist_fea2cen[i, label]
                results[label.item()]["distances"].append(dist)

    for i in range(args.train_class_num):
        # print(f"The examples number in class {i} is {len(results[i]['distances'])}")
        cls_dist = results[i]['distances']  # distance list for each class
        cls_dist.sort()  # python sort function do not return anything.
        results[i]['distances'] = cls_dist
        cls_dist = cls_dist[:-(args.tail_number)]  # remove the tail examples.

        index = int(len(cls_dist) * (1 - args.p_value))
        threshold = cls_dist[index].item()
        threshold_list.append(threshold)
        # cls_dist = cls_dist / (max(cls_dist))  # normalized to 0-1, we consider min as 0.
        # min_distance = min(cls_dist)
        min_distance = min(cls_dist)
        max_distance = max(cls_dist)
        hist = torch.histc(torch.Tensor(cls_dist), bins=args.bins, min=min_distance, max=max_distance)
        results[i]['hist'] = hist
        results[i]['max'] = max_distance
        results[i]['min'] = min_distance
        results[i]['threshold'] = threshold
    unknown_threshold = threshold - threshold - 100.  # we set threshold for unknown to -100. (actually 0 is fine)
    threshold_list.append(unknown_threshold)
    results['thresholds'] = torch.Tensor(threshold_list)  # the threshold for unknown is 0.
    torch.save(results, os.path.join(args.checkpoint, 'distance.pkl'))
    print("===> Distance saved.")
    return results
