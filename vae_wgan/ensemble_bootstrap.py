import numpy as np
import pickle
import glob

data = []

for f in glob.glob("/home/bfc/experiments/fmnist_vae_mar12/scores*"):
  print(f)
  data.append(pickle.load(open(f, "rb")))

datasets = [d['dataset'] for d in data[0]]

for dataset in datasets:
  aurocs = []
  for e in data: # each ensemble result
    for d in e:
      if d['dataset'] == dataset and d['threshold_var'] == 'WAIC': # can also threshold on single_elbo
        aurocs.append(d['AUROC']) 
  print('{}: mean={}, std={}'.format(dataset, np.mean(aurocs), np.std(aurocs))) 
