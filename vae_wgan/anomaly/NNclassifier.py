import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import pdb
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(0)

def FPRat95TPR(labels, predictions):
    fprs, tprs, thresholds = roc_curve(y_true=labels, y_score=predictions, drop_intermediate=False)
    for i in range(len(tprs)-1):
        if (tprs[i] < 0.95) and (tprs[i+1] >= 0.95):
            return fprs[i+1]

def FPRat99TPR(labels, predictions):
    fprs, tprs, thresholds = roc_curve(y_true=labels, y_score=predictions, drop_intermediate=False)
    for i in range(len(tprs)-1):
        if (tprs[i] < 0.99) and (tprs[i+1] >= 0.99):
            return fprs[i+1]




df = pd.read_csv('creditcard.csv.zip',header=0, sep=',', quotechar='"')

normal = df[df.Class == 0] #(284315,31)
anormal = df[df.Class == 1] #(492,31)

X_index = df.columns[1:-2]
normal_features = normal[X_index]
normal_time_amount = normal[['Time', 'Amount']]
normal_time_amount = (normal_time_amount - normal_time_amount.mean()) / normal_time_amount.std()
normal_X = pd.concat([normal_features, normal_time_amount], axis=1)
normal_X = normal_X.as_matrix()
normal_X = normal_X.astype(np.float32)

anormal_features = anormal[X_index]
anormal_time_amount = anormal[['Time', 'Amount']]
anormal_time_amount = (anormal_time_amount - anormal_time_amount.mean()) / anormal_time_amount.std()
anormal_X = pd.concat([anormal_features, anormal_time_amount], axis=1)
anormal_X = anormal_X.as_matrix()
anormal_X = anormal_X.astype(np.float32)


n_anormal = anormal.shape[0]  # 492
np.random.shuffle(normal_X)
test_normal_X = normal_X[:100]
test_anormal_X = anormal_X[:100]

test_data = np.concatenate((np.column_stack((test_normal_X,np.zeros(test_normal_X.shape[0])))
                           ,np.column_stack((test_anormal_X, np.ones(test_anormal_X.shape[0])))))
train_normal_X = normal_X[100:]
train_anormal_X = anormal_X[100:]

train_data = np.concatenate((np.column_stack((train_normal_X, np.zeros(train_normal_X.shape[0])))
                            ,np.column_stack((train_anormal_X, np.ones(train_anormal_X.shape[0])))))
#pdb.set_trace()

np.random.shuffle(train_data)

def statistics(labels, predictions):
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
    return TP,FP,TN,FN

def get_batch(data_x, batch_size=100):
    batch_n = len(data_x) // batch_size
    for i in range(batch_n):
        batch_x = data_x[i * batch_size:(i + 1) * batch_size]
        yield batch_x


tf.reset_default_graph()
with tf.Graph().as_default() as g:
    input_layer = tf.placeholder(tf.float32, shape=[None, 30], name='x')
    output_layer = tf.placeholder(tf.int32, shape=[None], name='y')
    layer1 = tf.layers.dense(input_layer, units=512)
    dropout1 = tf.layers.dropout(layer1, rate=0.5)
    layer2 = tf.layers.dense(layer1, units=512)
    dropout2 = tf.layers.dropout(layer2, rate=0.5)

    # Output logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=1)
    logits = tf.reshape(logits, [-1])
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.greater(logits,0.5),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # In predictions, return the prediction value, do not modify

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(output_layer,logits=logits,weights=1 + output_layer*580))


    optimizer = tf.train.RMSPropOptimizer(0.00001)
    train_op = optimizer.minimize(loss)

    tf.summary.scalar('loss', loss)
    summ_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/train')
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    g.finalize()

# train and visualize
with tf.Session(graph=g) as sess:

    sess.run(init)
    train_writer.add_graph(g)
    global_step = []
    losses = []
    step = 0
    for epoch in range(10):
        # train
        print("epoch",str(epoch))
        np.random.shuffle(train_data)
        fetch_dict = {'loss': loss, 'train': train_op, 'summary': summ_op}
        for batch_x in get_batch(train_data):
            feed_dict = {input_layer:batch_x[:,:-1], output_layer:batch_x[:,-1]}
            result = sess.run(fetch_dict, feed_dict)
            step += 1
            # collect loss
            if step % 200 == 0:
                train_writer.add_summary(result['summary'], step)
                global_step.append(step)
                losses.append(result['loss'])
                print("loss at", step, ":", result['loss'])





    feed_dict = {input_layer: test_data[:, :-1], output_layer: test_data[:, -1]}
    result = sess.run({"predictions":predictions["classes"], "logits": logits}, feed_dict)
    ap_score = sklearn.metrics.average_precision_score(y_true=test_data[:,-1], y_score=result["logits"])
    auroc_score = roc_auc_score(y_true=test_data[:,-1], y_score=result["logits"])
    fpr95tpr = FPRat95TPR(test_data[:,-1],result["logits"])
    fpr99tpr = FPRat99TPR(test_data[:, -1], result["logits"])
    stats=statistics(test_data[:,-1],result["predictions"])

    print("TP:",str(stats[0])," FP:",str(stats[1])," TN:",str(stats[2])," FN:",str(stats[3]))
    print("AP score :",str(ap_score), "at iteration", str(step))
    print("AUROC score : ",str(auroc_score), "at iteration",str(step))
    print("FPR@95%TPR : ", str(fpr95tpr), "at iteration", str(step))
    print("FPR@99%TPR : ", str(fpr99tpr), "at iteration", str(step))





