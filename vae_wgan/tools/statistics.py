import pdb
#from sklearn.utils import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score, f1_score
from tools.analysis import *


def FPRat95TPR(labels, predictions):
    fprs, tprs, thresholds = roc_curve(y_true=labels, y_score=predictions, drop_intermediate=False)
    for i in range(len(tprs)-1):
        if (tprs[i] < 0.95) and (tprs[i+1] >= 0.95):
            return fprs[i+1]

def F1score(labels, predictions, is_mean):
    test_dist = predictions[:1000]  #mnist/cifar10 test set distribution
    preds_normalized = predictions / (test_dist.max() - test_dist.min())
    test_normalized = preds_normalized[:1000]
    test_mean = test_normalized.mean()
    test_std = test_normalized.std()
    if is_mean:
        threshold = test_mean - 3*test_std
        masked = preds_normalized > threshold
    else:  # if var,
        threshold = test_mean + 3*test_std
        masked = preds_normalized < threshold
    f1 = f1_score(y_true=labels, y_pred=masked)
    return f1

def TFstatistics(labels, predictions):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(labels)):
        if labels[i] == 1 and predictions[i] == 1:
            TP += 1
        elif labels[i] == 0 and predictions[i] == 1:
            FP += 1
        elif labels[i] == 0 and predictions[i] == 0:
            TN += 1
        elif labels[i] == 1 and predictions[i] == 0:
            FN += 1
    return TP, FP, TN, FN

# calculates and prints # of true positives, false positives, etc
def print_stats(values, truth, thresholds, name):
    for threshold in thresholds:
        predictions = np.zeros(values.shape[0])
        predictions[np.argwhere(values > threshold)] = 1
        stats = TFstatistics(truth, predictions)
        print(name, str(threshold), ":  TP:", str(stats[0]), " FP:", str(stats[1]), " TN:", str(stats[2]), " FN:",
              str(stats[3]))


def analysis_helper(compare_datasets, expand_last_dim, noised_list, noise_type_list, show_adv_examples,
                   model_fn,  model_dir, which_model, adv_apply, keys, feature_shape=(28,28), each_size=1000):

    dataset_dic = {'mnist': tf.keras.datasets.mnist, 'fashion_mnist': tf.keras.datasets.fashion_mnist,
                   'cifar10': tf.keras.datasets.cifar10, 'cifar100': tf.keras.datasets.cifar100,
                   'ImageNet':'ImageNet', 'SVHN':'SVHN','notMNIST':'notMNIST', 'celebA':'celebA',
                   'normal_noise': 'normal_noise','uniform_noise': 'uniform_noise',
                   'credit_card_normal':'credit_card_normal','credit_card_anomalies':'credit_card_anomalies'}

    converted_datasets = []
    datasets_names = []
    for index, dataset in enumerate(compare_datasets):
        converted_datasets.append(dataset_dic[dataset])
        dataset_name = dataset
        if noised_list[index]:
            dataset_name = dataset_name + ' nsd by '+noise_type_list[index]
        datasets_names.append(dataset_name)

    input_fn = build_eval_multiple_datasets(converted_datasets, 100, expand_last_dim, noised_list,
                                            noise_type_list, feature_shape, each_size)

    results = fetch(input_fn, model_fn, model_dir, keys, which_model)

    if show_adv_examples is not None:
        adv_normal, adv_uniform = adversarial_fetch(get_eval_dataset(compare_datasets[0], 100), 100, model_fn,
                                                    model_dir, keys, which_model, adv_apply)

        if show_adv_examples == 'normal':
            datasets_names.append('adv normal noise')
            results = np.concatenate([results, adv_normal], axis=1)
        elif show_adv_examples == 'uniform':
            datasets_names.append('adv uniform noise')
            results = np.concatenate([results, adv_uniform], axis=1)

    return results, datasets_names

def plot_analysis(results, datasets_names, keys,  bins=None, each_size=1000):
    results = np.clip(results, -1e5, 1e5) # clip so that histograms work
    num_dataset = len(datasets_names)
    f, axes = plt.subplots(len(results), num_dataset + 1, figsize=(5 * (num_dataset + 1), 5 * len(results)),
                           sharex='row', sharey='row')
    for i, value in enumerate(results):  # iterate over values of each key
        if len(keys) == 1:
            this_axis = axes[0]
            last_axis = axes[num_dataset]
        else:
            this_axis = axes[i, 0]
            last_axis = axes[i, num_dataset]

        if (bins is not None) and (keys[i] in bins.keys()):
            bin = bins[keys[i]]
            bin = np.linspace(*bin)
            this_axis.hist(value[:each_size], bins=bin, alpha=0.5, label=datasets_names[0])
            last_axis.hist(value[:each_size], bins=bin, alpha=0.3, label=datasets_names[0])
        else:
            this_axis.hist(value[:each_size], alpha=0.5, label=datasets_names[0])
            last_axis.hist(value[:each_size], alpha=0.3, label=datasets_names[0])

        this_axis.set_xlabel(keys[i] + " of " + datasets_names[0])
        for index in range(1, num_dataset):
            if len(keys) == 1:
                this_axis = axes[index]
            else:
                this_axis = axes[i, index]
            if (bins is not None) and (keys[i] in bins.keys()):
                this_axis.hist(value[each_size * index:each_size * (index + 1)], bins=bin, alpha=0.5, label=datasets_names[index])
                last_axis.hist(value[each_size * index:each_size * (index + 1)], bins=bin, alpha=0.3, label=datasets_names[index])
            else:
                this_axis.hist(value[each_size * index:each_size * (index + 1)], alpha=0.5, label=datasets_names[index])
                last_axis.hist(value[each_size * index:each_size * (index + 1)], alpha=0.3, label=datasets_names[index])

            this_axis.set_xlabel(keys[i] + " of " + datasets_names[index])

            if 'var' in keys[i]:
                is_mean = False  # points above threshold are anomalous
                truth = np.concatenate([np.zeros(each_size), np.ones(each_size)])
            else:
                is_mean = True  # points below threshold are anomalous
                truth = np.concatenate([np.ones(each_size), np.zeros(each_size)])

            predictions = np.concatenate([value[:each_size], value[each_size * index:each_size * (index + 1)]])
            auroc_score = roc_auc_score(y_true=truth, y_score=predictions)
            ap_score = average_precision_score(y_true=truth, y_score=predictions)
            fpr_at_95tpr = FPRat95TPR(truth, predictions)
            f1score = F1score(truth, predictions, is_mean)
            print(datasets_names[index], " using ", keys[i], ",  AUROC: ", str(auroc_score)[:6], "  AP: ", str(ap_score)[:6],
                  " FPR@95%TPR: ", str(fpr_at_95tpr)[:5]," f1: ", str(f1score)[:5])
        last_axis.set_xlabel(keys[i])
        last_axis.legend()

    f.savefig("stats")

# for single model
def single_analysis(compare_datasets, expand_last_dim, noised_list, noise_type_list, show_adv_examples, model_fn,  model_dir, which_model,
                                                         adv_apply, keys, bins=None, feature_shape=(28,28), each_size=1000):


    results, datasets_names = analysis_helper(compare_datasets, expand_last_dim, noised_list, noise_type_list,
                                     show_adv_examples, model_fn,  model_dir, which_model, adv_apply, keys, feature_shape, each_size)

    # plot elbo/logit/logp(x) for each dataset
    # and calculate AUROC/AP scores for keys (threshold variables)
    plot_analysis(results, datasets_names, keys, bins, each_size)


