# Uncertainty Estimation and Generative Models

Unsupervised Anomaly Detection

Ensemble of VAEs for detecting out of distribution (OOD) samples

## Variational Autoencoder

![alt text](https://github.com/hschoi1/rich_latent/tree/master/adv/model.jpg)

Take an input from the data space, map it into a distribution (posterior) in the latent space, sample from the posterior distribution, and decode it back to an output in the data space.

Unlike the regular VAE, we used Normalizing Flows for a richer prior.


## Out of Distribution Samples

These are reconstructed inputs for adversarially perturbed input (second row) and adversarially perturbed random noise (third row)
Both are created by the Fast Gradient Method.

![alt text](https://github.com/hschoi1/rich_latent/tree/master/adv/adversarial.png)

This is the latent space for normal digits

![alt text](https://github.com/hschoi1/rich_latent/tree/master/adv/normal.png)

This is the latent space for adversarially perturbed input.

![alt text](https://github.com/hschoi1/rich_latent/tree/master/adv/perturbed_input_latent.png)

This is the latent space for adversarially perturbed random noise.

![alt text](https://github.com/hschoi1/rich_latent/tree/master/adv/perturbed_random_noise_latent.png)


## Anomaly Detection

As a supervised classifier we used a single NN classifier, AUROC = 0.94 and AP Score = 0.94

For a unsupervised classifier, we want to use our ensemble model.

Anomalies are also out of distribution samples. However, ELBO is not enough.

Which threshold to use for unsupervised learning?
-> Ensemble Statistics

## Ensemble Learning
The ensemble statistics will serve as proper threshold variables

![alt text](https://github.com/hschoi1/rich_latent/tree/master/adv/iwae_vs_ensemble_var.png)

Using the ensemble variance of posterior means, we achieved AUROC = 0.927 and AP Score = 0.940

## Dependencies
Tensorflow Probability

## References
1.Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles

2.Joshua V.Dillon, Ian Langmore, Dustin Tran, Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt Hoffman, Rif A. Saurous. Tensorflow Distributions

3.Danilo Jimenez Rezende and Shakir Mohamed, Variational Inference with Normalizing Flows

## Code Citations
url: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

## Mentor
Eric Jang

## Dataset
Credit Card Fraud Detection from Kaggle

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

## Acknowledgements
This was supported by [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) which was organized by [TensorFlow Korea User Group](https://facebook.com/groups/TensorFlowKR/).
