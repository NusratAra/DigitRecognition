#pip install requests jupyter
import mxnet as mx


#mnist = mx.test_utils.get_mnist()
import numpy

trainf = open("bangla_data/train_images.txt", "r")
train =  list(trainf.read().split())


trainlf = open("bangla_data/train_labels.txt", "r")
trainl =  list(trainlf.read().split())

print trainl

testf = open("bangla_data/test_images.txt", "r")
test =  list(testf.read().split())

testlf = open("bangla_data/test_labels.txt", "r")
testl =  list(testlf.read().split())

batch_size = 784
train_iter = mx.io.NDArrayIter(train, trainl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(test, testl, batch_size)

print train_iter
print val_iter


data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
print conv1
print "conv1"
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
print "tanh1"
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
print "pool1"
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
print "conv2"
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
print "tanh2"
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
print "pool2"
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
print "first fullc layer"
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
print "second fullc"
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
print "softmax loss"


# # create a trainable module on GPU 0
# lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
# # train with the same
# lenet_model.fit(train_iter,
#                 eval_data=val_iter,
#                 optimizer='sgd',
#                 optimizer_params={'learning_rate':0.1},
#                 eval_metric='acc',
#                 batch_end_callback = mx.callback.Speedometer(batch_size, 100),
#                 num_epoch=1)
# print "training complete"
#
#
#
# test_iter = mx.io.NDArrayIter(test, None, batch_size)
# prob = lenet_model.predict(test_iter)
# test_iter = mx.io.NDArrayIter(test, testl, batch_size)
# # predict accuracy for lenet
# acc = mx.metric.Accuracy()
# lenet_model.score(test_iter, acc)
# print(acc)
# assert acc.get()[1] > 0.98
trainf.close()
trainlf.close()
testf.close()
testlf.close()