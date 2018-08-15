import pdb
#from sklearn.utils import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score


def plot_analysis(results, datasets_names, keys,  bins=None):

    num_dataset = len(datasets_names)
    f, axes = plt.subplots(len(results), num_dataset + 1, figsize=(5 * (num_dataset + 1), 5 * len(results)),
                           sharex='row', sharey='row')
    for i, value in enumerate(results):
        if len(keys) == 1:
            this_axis = axes[0]
            last_axis = axes[num_dataset]
        else:
            this_axis = axes[i, 0]
            last_axis = axes[i, num_dataset]
        if (bins is not None) and (keys[i] in bins.keys()):
            bin = bins[keys[i]]
            bin = np.linspace(*bin)
            this_axis.hist(value[:1000], bins=bin, alpha=0.5, label=datasets_names[0])
            last_axis.hist(value[:1000], bins=bin, alpha=0.3, label=datasets_names[0])
        else:
            this_axis.hist(value[:1000], alpha=0.5, label=datasets_names[0])
            last_axis.hist(value[:1000], alpha=0.3, label=datasets_names[0])

        this_axis.set_xlabel(keys[i] + " of " + datasets_names[0])
        for index in range(1, num_dataset):
            if len(keys) == 1:
                this_axis = axes[index]
            else:
                this_axis = axes[i, index]
            if (bins is not None) and (keys[i] in bins.keys()):
                this_axis.hist(value[1000 * index:1000 * (index + 1)], bins=bin, alpha=0.5, label=datasets_names[index])
                last_axis.hist(value[1000 * index:1000 * (index + 1)], bins=bin, alpha=0.3, label=datasets_names[index])
            else:
                this_axis.hist(value[1000 * index:1000 * (index + 1)], alpha=0.5, label=datasets_names[index])
                last_axis.hist(value[1000 * index:1000 * (index + 1)], alpha=0.3, label=datasets_names[index])

            this_axis.set_xlabel(keys[i] + " of " + datasets_names[index])

            truth = np.concatenate([np.ones(1000), np.zeros(1000)])
            predictions = np.concatenate([value[:1000], value[1000 * index:1000 * (index + 1)]])
            auroc_score = roc_auc_score(y_true=truth, y_score=predictions)
            ap_score = average_precision_score(y_true=truth, y_score=predictions)
            print(datasets_names[index], "using", keys[i], ",  AUROC: ", str(auroc_score)[:6], "  AP: ", str(ap_score)[:6])
        last_axis.set_xlabel(keys[i])
        last_axis.legend()

    f.savefig("stats")

