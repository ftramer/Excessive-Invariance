import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 128#64 # Model dimensionality
BATCH_SIZE = 100 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 2000000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

NOISY = False

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    label = None
    if noise is None:
        label = tf.random_uniform([n_samples],0,10,dtype=tf.int32)
        label = tf.one_hot(label, 10)
        noise = tf.random_normal([n_samples, 64])
        noise = tf.concat([label, noise], axis=1)
        
    output = lib.ops.linear.Linear('Generator.Input', 64+10, 4*4*4*DIM, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)    

    if NOISY:
        output += tf.random_normal((n_samples,1,28,28), stddev=.1)
    
    output = tf.nn.sigmoid(output)
    

    return tf.reshape(output, [-1, OUTPUT_DIM]), label

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])

    preds = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 10, output)
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1]), preds

if __name__ == "__main__":
    real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    labels_real = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    fake_data, labels_fake = Generator(BATCH_SIZE)
    
    disc_real, preds_real = Discriminator(real_data)
    disc_fake, preds_fake = Discriminator(fake_data)
    
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')
    
    if True:
        classifier_cost_real = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_real,
                                                                              logits=preds_real)
        classifier_cost_fake = tf.nn.softmax_cross_entropy_with_logits(labels=labels_fake,
                                                                       logits=preds_fake)
        classifier_cost = classifier_cost_real + classifier_cost_fake
        gen_cost = -tf.reduce_mean(disc_fake) + classifier_cost
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) + classifier_cost
    
        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty
    
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params)
    
        clip_disc_weights = None
    
    # For saving samples
    fixed_noise = np.random.normal(size=(128, 74))
    fixed_noise[:,:10] = 0
    for i in range(128):
        fixed_noise[i,i%10] = 1
    fixed_noise = tf.constant(fixed_noise.astype('float32'))
    fixed_noise_samples, _ = Generator(128, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        lib.save_images.save_images(
            samples.reshape((128, 28, 28)), 
            ("noisy-" if NOISY else "")+'mnist_acgan_samples_{0:09d}.png'.format(frame)
        )
    
    # Dataset iterator
    train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
    def inf_train_gen():
        while True:
            for images,targets in train_gen():
                yield images,targets
    
    saver = tf.train.Saver()
    
    # Train loop
    with tf.Session() as session:
    
        session.run(tf.initialize_all_variables())
    
        gen = inf_train_gen()
    
        for iteration in range(ITERS):
            start_time = time.time()
    
            if iteration > 0:
                _ = session.run(gen_train_op)
    
            if MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = CRITIC_ITERS
            for i in range(disc_iters):
                _data,_targets = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data,
                               labels_real: _targets}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)
    
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
    
            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
                dev_disc_costs = []
                for images,targets in dev_gen():
                    _dev_disc_cost, _creal, _cfake = session.run(
                        (disc_cost, classifier_cost_real, classifier_cost_fake), 
                        feed_dict={real_data: images,
                                   labels_real: targets}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                lib.plot.plot('dev classreal cost', np.mean(_creal))
                lib.plot.plot('dev classfake cost', np.mean(_cfake))
    
                generate_image(iteration, _data)
                saver.save(session, 'model/mnist-acgan-2'+("-noisy" if NOISY else ""))
    
            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()
    
            lib.plot.tick()
    
