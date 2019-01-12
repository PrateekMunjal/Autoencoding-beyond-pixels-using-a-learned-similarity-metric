import tensorflow as tf 
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt 

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('data/fashion/',one_hot=True);
mnist = input_data.read_data_sets('tmp/data/',one_hot=True);


tf.set_random_seed(10);

#tuning_knobs
encoder_learning_rate = 0.001;
decoder_learning_rate = 0.001;
discriminator_learning_rate = 0.001;

batch_size = 256;
n_epoch = 200;
z_dim = 32;

#model_params
n_inputs = 28*28; #as images are of 28 x 28 dimension
n_outputs = 10;

tfd = tf.contrib.distributions

X = tf.placeholder(tf.float32,[None,n_inputs]);
Y = tf.placeholder(tf.float32,[None,n_outputs]);
epoch_number = tf.placeholder(tf.float32,[]);
keep_prob = tf.placeholder(tf.float32,[]);

def prior_z(latent_dim):
        z_mean = tf.zeros(latent_dim);
        z_var = tf.ones(latent_dim);
        return tfd.MultivariateNormalDiag(z_mean,z_var);

#assumed noise distribution N(0,1)
def epsilon_distribution(latent_dim):
    eps_mean = tf.zeros(latent_dim);
    eps_var = tf.ones(latent_dim);
    return tfd.MultivariateNormalDiag(eps_mean,eps_var);

def encoder_dist(X,isTrainable=True,reuse=False,name='encoder'):
    
    with tf.variable_scope(name) as scope:  
        
        encoder_activations = {};

        if reuse:
            scope.reuse_variables();
        
        X = tf.reshape(X,[-1,28,28,1]);
        
        #28x28x1 --> means size of input before applying conv1
        conv1 = tf.layers.conv2d(X,filters=16,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse); 
        encoder_activations['enc_conv1_layer'] = conv1;
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        #14x14x16 
        conv2 = tf.layers.conv2d(conv1,filters=32,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse);
        encoder_activations['enc_conv2_layer'] = conv2;
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        #7x7x32
        conv3 = tf.layers.conv2d(conv2,filters=64,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=tf.nn.relu,trainable=isTrainable,reuse=reuse);
        encoder_activations['enc_conv3_layer'] = conv3;
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        #4x4x64
        conv3_flattened = tf.layers.flatten(conv3);

        z_mean = tf.layers.dense(conv3_flattened,z_dim,name='enc_mean',trainable=isTrainable,reuse=reuse);
        z_variance = tf.layers.dense(conv3_flattened,z_dim,activation=tf.nn.softplus,name='enc_variance',trainable=isTrainable,reuse=reuse);
        epsilon_val = epsilon_distribution(z_dim).sample(tf.shape(X)[0]);
        z_sample = tf.add(z_mean,tf.multiply(z_variance,epsilon_val));

        dist = tfd.MultivariateNormalDiag(z_mean,z_variance);

        return dist,z_sample,encoder_activations;

def decoder(z_sample,isTrainable=True,reuse=False,name='decoder'):
    print('------------------------');
    print('z_sample : ',z_sample);
    print('------------------------');
    with tf.variable_scope(name) as scope:
        
        decoder_activations = {};

        if reuse:
            scope.reuse_variables();

        z_sample = tf.layers.dense(z_sample,4*4*64,activation=tf.nn.tanh,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer');
        z_sample = tf.reshape(z_sample,[-1,4,4,64]);
        
        deconv1 = tf.layers.conv2d_transpose(z_sample,filters=32,kernel_size=[3,3],padding='SAME',activation=tf.nn.leaky_relu,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse); # 16x16
        decoder_activations['dec_deconv1_layer'] = deconv1;
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1');

        deconv2 = tf.layers.conv2d_transpose(deconv1,filters=16,kernel_size=[3,3],padding='SAME',activation=tf.nn.leaky_relu,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse); #32x32
        decoder_activations['dec_deconv2_layer'] = deconv2;
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2');

        deconv3 = tf.layers.conv2d_transpose(deconv2,filters=1,kernel_size=[3,3],padding='SAME',activation=tf.nn.leaky_relu,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse); #32x32
        decoder_activations['dec_deconv3_layer'] = deconv3;
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3');

        deconv3 = tf.layers.flatten(deconv3);
        deconv_fc = tf.layers.dense(deconv3,28*28,activation=tf.nn.sigmoid,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_last_layer');

        print('--------------------------');
        print('deconv3 : ',deconv3);
        print('--------------------------');
        deconv_fc_reshaped = tf.reshape(deconv_fc,[-1,784]);

        return deconv_fc_reshaped,decoder_activations;

def discriminator(X,isTrainable=True,reuse=False,name='discriminator'):
    with tf.variable_scope(name) as scope:
            
        discriminator_activations = {};

        if reuse:
            scope.reuse_variables();

        X = tf.reshape(X,[-1,28,28,1]);

        #28x28x1 --> means size of input before applying conv1
        conv1 = tf.layers.conv2d(X,filters=16,kernel_size=[3,3],padding='SAME',strides=(2,2),name='dis_conv1_layer',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse); 
        discriminator_activations['dis_conv1_layer'] = conv1;
        #conv1 = tf.layers.dropout(conv1,rate=keep_prob,training = isTrainable);
        #conv1 = tf.layers.batch_normalization(conv1);
        #14x14x16 
        conv2 = tf.layers.conv2d(conv1,filters=32,kernel_size=[3,3],padding='SAME',strides=(2,2),name='dis_conv2_layer',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse);
        discriminator_activations['dis_conv2_layer'] = conv2;
        #conv2 = tf.layers.dropout(conv2,rate=keep_prob,training = isTrainable);
        #conv2 = tf.layers.batch_normalization(conv2);
        #7x7x32
        conv3 = tf.layers.conv2d(conv2,filters=64,kernel_size=[3,3],padding='SAME',strides=(2,2),name='dis_conv3_layer',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse);
        discriminator_activations['dis_conv3_layer'] = conv3;
        #conv3 = tf.layers.dropout(conv3,rate=keep_prob,training = isTrainable);
        #conv3 = tf.layers.batch_normalization(conv3);
        #4x4x64
        conv3_flattened = tf.layers.flatten(conv3);
        l_th_layer_representation = tf.layers.flatten(conv3_flattened);

        output_disc = tf.layers.dense(conv3_flattened,1,activation=tf.nn.sigmoid,name='dis_fc_layer',trainable=isTrainable,reuse=reuse);
        return l_th_layer_representation,output_disc;

posterior_dist,z_sample,encoder_activations = encoder_dist(X);
prior_dist = prior_z(z_dim);

generated_sample = prior_dist.sample(batch_size);
print('========================');
print('** z_sample : ',z_sample);
print('========================');
reconstructed_x_tilde,x_tilde_decoder_activations = decoder(z_sample);
test_reconstruction,_ = decoder(z_sample,isTrainable=False,reuse=True);
reconstructed_x_dash,x_dash_decoder_activations = decoder(generated_sample,reuse=True);

true_x_l_th_layer_representation,Dis_X = discriminator(X);
x_tilde_l_th_layer_representation,Dis_x_tilde = discriminator(reconstructed_x_tilde,reuse=True);
x_dash_l_th_layer_representation,Dis_x_dash = discriminator(reconstructed_x_dash,reuse=True);

#loss functions :)
ae_loss = tf.reduce_mean(tf.pow(X- reconstructed_x_tilde,2));

gan_loss = tf.reduce_mean(tf.add(tf.add(tf.log(Dis_X),tf.log(1-Dis_x_tilde)),tf.log(1-Dis_x_dash)));
#gan_loss = tf.reduce_mean(tf.add(tf.log(Dis_X),tf.log(1-Dis_x_dash)));
gan_loss = -1 * gan_loss; #bcoz we need to maximize above loss function, henceforth it is same as minimizing negation of it.

#dis_l_layer_loss = tf.reduce_mean(tf.pow(x_tilde_l_th_layer_representation - x_dash_l_th_layer_representation,2));
dis_l_layer_loss = tf.reduce_mean(tf.pow(x_tilde_l_th_layer_representation - true_x_l_th_layer_representation,2));
kl_loss = tf.reduce_mean(tfd.kl_divergence(posterior_dist,prior_dist));
#kl_loss = tf.clip_by_value(kl_loss,0.0,0.30,name='clipped_kl_loss'); 

encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder');
decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder');
discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator');
#
gamma1 = 100;#50/(epoch_number+1);

decoder_loss = gamma1*dis_l_layer_loss + tf.reduce_mean(- tf.log(Dis_x_tilde) - tf.log(Dis_x_dash));
#decoder_loss = gamma*dis_l_layer_loss +tf.reduce_mean(- tf.log(Dis_x_dash));
#human_loss_fn = tf.reduce_mean(tf.pow(X- reconstructed_x_tilde,2)); # in paper gan_loss is added with negative sign but we have already negated it above --> so we have added gan_loss with positive sign
#human_weightage = 50.0/784;
#decoder_loss += human_weightage*human_loss_fn;


discriminator_loss = gan_loss;
kl_weightage = 1/(z_dim);#0.0005;#1.0/(batch_size);

gamma2 = 100;
#kl_weightage = 1.0 / (1.0 + tf.exp(-epoch_number/3+5));
encoder_loss = kl_weightage*kl_loss + gamma2*dis_l_layer_loss;
#encoder_loss = kl_weightage*kl_loss + gamma*dis_l_layer_loss;
#encoder_loss = dis_l_layer_loss;

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):

    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001,beta1=0.5);
    autoencoder_gradsVars = autoencoder_optimizer.compute_gradients(ae_loss, encoder_params+decoder_params);
    autoencoder_train_optimizer = autoencoder_optimizer.apply_gradients(autoencoder_gradsVars);

    encoder_optimizer = tf.train.AdamOptimizer(learning_rate = encoder_learning_rate,beta1=0.5);
    encoder_gradsVars = encoder_optimizer.compute_gradients(encoder_loss, encoder_params);
    encoder_train_optimizer = encoder_optimizer.apply_gradients(encoder_gradsVars);

    decoder_optimizer = tf.train.AdamOptimizer(learning_rate = decoder_learning_rate,beta1=0.5,beta2=0.999);
    decoder_gradsVars = decoder_optimizer.compute_gradients(decoder_loss, decoder_params);
    decoder_train_optimizer = decoder_optimizer.apply_gradients(decoder_gradsVars);

    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = discriminator_learning_rate,beta1=0.5);
    discriminator_gradsVars = discriminator_optimizer.compute_gradients(discriminator_loss, discriminator_params);
    discriminator_train_optimizer = discriminator_optimizer.apply_gradients(discriminator_gradsVars);
'''
epoch_dis_loss = 0.0;
epoch_dis_loss += batch_dis_loss;

epoch_dec_loss = 0.0;
epoch_dec_loss += batch_dec_loss;

epoch_enc_loss = 0.0;
epoch_enc_loss += batch_enc_loss;
'''
##TENSORBOARD
tf.summary.scalar("kl_loss ",kl_weightage*kl_loss);
tf.summary.scalar("Discriminator_Lth_layer_loss in Encoder ",gamma1*dis_l_layer_loss);
tf.summary.scalar("Discriminator_Lth_layer_loss in Decoder ",gamma2*dis_l_layer_loss);
tf.summary.scalar("encoder_loss",encoder_loss);
tf.summary.scalar("decoder_loss",decoder_loss);
tf.summary.scalar("discriminator_loss",discriminator_loss);
#tf.summary.scalar("human_loss_fn",human_weightage*human_loss_fn);

for g,v in encoder_gradsVars:    
    tf.summary.histogram(v.name,v)
    tf.summary.histogram(v.name+str('grad'),g)

for g,v in decoder_gradsVars:    
    tf.summary.histogram(v.name,v)
    tf.summary.histogram(v.name+str('grad'),g)

for g,v in discriminator_gradsVars:    
    tf.summary.histogram(v.name,v)
    tf.summary.histogram(v.name+str('grad'),g)

merged_all = tf.summary.merge_all();
log_directory = 'VAE-GAN-dir';
model_directory='VAE-GAN-model_dir';

if not os.path.exists(log_directory):
    os.makedirs(log_directory);
if not os.path.exists(model_directory):
    os.makedirs(model_directory); 

op_dirs = ['op-real','op-gen','op-recons'];
for dir_name in op_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name);

def train():
    n_batches = mnist.train.num_examples/batch_size;
    n_batches = int(n_batches);
    print('n_batches : ',n_batches);
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        temp_batch = 1; #for plotting
        #for tensorboard
        saver = tf.train.Saver();
        writer = tf.summary.FileWriter(log_directory,sess.graph);
        iterations = 0;

        # for epoch in range(15):
        #   for batch in range(n_batches):
        #       dead_neurons_prob = 0.0;
        #       X_batch,Y_batch = mnist.train.next_batch(batch_size);
        #       fd = {X:X_batch,Y:Y_batch,epoch_number:epoch,keep_prob:dead_neurons_prob};
        #       #Train Autoencoder 
        #       k=1;
        #       for i in range(k):
        #           #X_batch,Y_batch = mnist.train.next_batch(batch_size);
        #           #fd = {X:X_batch,Y:Y_batch,epoch_number:epoch};
        #           _,ae_loss_= sess.run([autoencoder_train_optimizer,ae_loss],feed_dict = fd);
        #   print('AT epoch #',epoch,' AE-loss : ',ae_loss_);

        # print('-------------------------------');
        # print('-----------Autoencoder Trained------------');
        # print('-------------------------------');

        for epoch in range(n_epoch):
            for batch in range(n_batches):
                iterations += 1;
                dead_neurons_prob = 0.0;
                X_batch,Y_batch = mnist.train.next_batch(batch_size);
                fd = {X:X_batch,Y:Y_batch,epoch_number:epoch,keep_prob:dead_neurons_prob};
                #Train Discriminator 
                k=1;
                for i in range(k):
                    #X_batch,Y_batch = mnist.train.next_batch(batch_size);
                    #fd = {X:X_batch,Y:Y_batch,epoch_number:epoch};
                    _,dis_loss= sess.run([discriminator_train_optimizer,discriminator_loss],feed_dict = fd);

                #Train Encoder
                
                j=1;
                for i in range(j):
                    #X_batch,Y_batch = mnist.train.next_batch(batch_size);
                    #fd = {X:X_batch,Y:Y_batch,epoch_number:epoch};
                    #_,enc_loss,kl_div_loss,merged = sess.run([encoder_train_optimizer,encoder_loss,kl_loss,merged_all],feed_dict = fd);
                    _,enc_loss = sess.run([encoder_train_optimizer,encoder_loss],feed_dict = fd);

                #Train Decoder
                #X_batch,Y_batch = mnist.train.next_batch(batch_size);
                #fd = {X:X_batch,Y:Y_batch,epoch_number:epoch};
                m=1;
                for i in range(m):
                    #X_batch,Y_batch = mnist.train.next_batch(batch_size);
                    #fd = {X:X_batch,Y:Y_batch,epoch_number:epoch};
                    _,dec_loss,kl_div_loss,merged = sess.run([decoder_train_optimizer,decoder_loss,kl_loss,merged_all],feed_dict = fd);
                    #_,dec_loss = sess.run([decoder_train_optimizer,decoder_loss],feed_dict = fd);

                
                
                #_dis_loss,_enc_loss,_dec_loss,merged = sess.run([epoch_dis_loss,epoch_enc_loss,epoch_dec_loss,merged_all],feed_dict = {X:X_batch,Y:Y_batch,epoch_number:epoch,batch_dis_loss : dis_loss,batch_dec_loss:dec_loss,batch_enc_loss:enc_loss});
                #merged = sess.run(merged_all);
                if(iterations%20==0):
                    writer.add_summary(merged,iterations);

                if(batch%200 == 0):
                    print('Batch #',batch,' done!');

            if(epoch%5==0):
                n = 5;
                #dead_neurons_prob = 0.7;
                reconstructed = np.empty((28*n,28*n));
                original = np.empty((28*n,28*n));

                for i in range(n):
                    
                    batch_X,_ = mnist.test.next_batch(n);
                    recons = sess.run(test_reconstruction,feed_dict={X:batch_X});
                    #print ('recons : ',recons.shape);
                    recons = np.reshape(recons,[-1,784]);
                    #print ('recons : ',recons.shape);

                    for j in range(n):
                            original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_X[j].reshape([28, 28]);

                    for j in range(n):
                        reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

                print("Original Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original, origin="upper", cmap="gray");
                plt.savefig('op-real/original_new_vae_'+str(epoch)+'.png');

                print("Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed, origin="upper", cmap="gray");
                plt.savefig('op-recons/reconstructed_new_vae'+str(epoch)+'.png');

                n=5;
                reconstructed = np.empty((28*n,28*n));
                for i in range(n):
                    sample = tf.random_normal([n,z_dim]);
                    recons = sess.run(test_reconstruction,feed_dict={z_sample:sample.eval()});
                    recons = np.reshape(recons,[-1,784]);

                    for j in range(n):
                        reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

                print("Generated Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed, origin="upper", cmap="gray");
                plt.title('Epoch '+str(epoch));
                plt.savefig('op-gen/gen-img-'+str(epoch)+'.png');
                plt.close();
                        
                #writer.add_summary(merged,epoch);#*n_batches + batch);
            print('=== Epoch #',epoch,' completed! ===');
            print('encoder_loss : ',enc_loss,' decoder_loss : ',dec_loss,' discriminator/gan_loss : ',dis_loss,' kl-div loss : ',kl_div_loss);
            if (epoch % 2) == 0:
                save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
                print("At epoch #",epoch," Model is saved at path: ",save_path);

        print('----------------------------------');
        print ('Training Phase Completed');
        print('----------------------------------');

        n = 5;
        
        reconstructed = np.empty((28*n,28*n));
        original = np.empty((28*n,28*n));

        for i in range(n):
            
            batch_X,_ = mnist.test.next_batch(n);
            recons = sess.run(test_reconstruction,feed_dict={X:batch_X});
            print ('recons : ',recons.shape);
            recons = np.reshape(recons,[-1,784]);
            print ('recons : ',recons.shape);

            for j in range(n):
                    original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_X[j].reshape([28, 28]);

            for j in range(n):
                reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

        print("Original Images");
        plt.figure(figsize=(n, n));
        plt.imshow(original, origin="upper", cmap="gray");
        plt.savefig('original_new_vae.png');

        print("Reconstructed Images");
        plt.figure(figsize=(n, n));
        plt.imshow(reconstructed, origin="upper", cmap="gray");
        plt.savefig('reconstructed_new_vae.png');


def sampleImages(model_num=n_epoch-2):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        #saver = tf.train.Saver();

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        '''
        for var in params:
            print (var.name+"\t");
        '''
        string = model_directory+'/model_'+str(model_num); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded");
        
        #saver = tf.train.Saver();

        n = 10000;

        generate_dir = 'simpleMNIST-generations';

        if not os.path.exists(generate_dir):
            os.makedirs(generate_dir);
        
        samplesFromPrior = tf.random_normal([n,z_dim]).eval();
        for i in range(n):
            #sample = tf.random_normal([1,z_dim]);
            sample = samplesFromPrior[[i],:];
            #print('sample.shape : ',sample.shape);
            recons = sess.run(test_reconstruction,feed_dict={z_sample:sample});
            plt.imshow(np.reshape(recons,[28,28]), interpolation="nearest", cmap="gray");
            #plt.title('Generated Image');
            plt.savefig(generate_dir+'/gen-img-'+str(i)+'.png');
            if(i%100==0):
                print('gen-img-'+str(i)+'.png done !!');
            plt.close();        

def generateImages():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        #saver = tf.train.Saver();

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        for var in params:
            print (var.name+"\t");

        string = model_directory+'/model_'+str(198); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded");
        
        #saver = tf.train.Saver();

        n = 5;
        
        reconstructed = np.empty((28*n,28*n));
        original = np.empty((28*n,28*n));

        for i in range(n):
            
            #batch_X,_ = mnist.test.next_batch(n);
            batch_X = mnist.test.images[i*n:i*n+n];
            recons = sess.run(test_reconstruction,feed_dict={X:batch_X});
            #print ('recons : ',recons.shape);
            recons = np.reshape(recons,[-1,784]);
            #print ('recons : ',recons.shape);

            for j in range(n):
                    original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_X[j].reshape([28, 28]);

            for j in range(n):
                reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

        print("Original Images");
        plt.figure(figsize=(n, n));
        plt.imshow(original, origin="upper", cmap="gray");
        plt.savefig('orig-img.png');

        print("Reconstructed Images");
        plt.figure(figsize=(n, n));
        plt.imshow(reconstructed, origin="upper", cmap="gray");
        plt.savefig('recons-img.png');


        n=10;
        reconstructed = np.empty((28*n,28*n));
        for i in range(n):
            sample = tf.random_normal([n,z_dim]);
            recons = sess.run(test_reconstruction,feed_dict={z_sample:sample.eval()});
            recons = np.reshape(recons,[-1,784]);

            for j in range(n):
                reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

        print("Generated Images");
        plt.figure(figsize=(n, n));
        plt.imshow(reconstructed, origin="upper", cmap="gray");
        plt.title('Epoch ');
        plt.savefig('gen-img.png');
        plt.close();
        # sample = tf.random_normal([1,z_dim]);
        # recons = sess.run(test_reconstruction,feed_dict={z_sample:sample.eval()});
        # plt.imshow(np.reshape(recons,[28,28]), interpolation="nearest", cmap="gray");
        # plt.title('Generated Image');
        # plt.savefig('gen-img.png');


recons_output_directory = 'reconstructions/'
orig_output_directory = 'original/'

if not os.path.exists(recons_output_directory):
    os.makedirs(recons_output_directory); 
if not os.path.exists(orig_output_directory):
    os.makedirs(orig_output_directory);

def generateReconstructedImages(num_images=10000,model_number=198,n=1):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        #model_directory = 'myVAE-GAN-model_dir_save_when_62_epoch_is_running';
        string = model_directory+'/model_'+str(model_number); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded successfully from ",string);

        n_batches = int(1.0*num_images/n);
        print('Total n_batches : ',n_batches);
        start_image_number = 0;
        for batch in range(n_batches):
            start_image_number = batch*n;
            stop_image_number = start_image_number + n;
            print('start_image_number : ',start_image_number);
            print('stop_image_number : ',stop_image_number);
            batch_X = mnist.test.images[batch*n:batch*n+n];
            #print('batch_X.shape : ',batch_X.shape);
            #sys.exit(0);


            recons = sess.run(test_reconstruction,feed_dict={X:batch_X});
            recons = np.reshape(recons,[-1,784]);

            _start_image_number = start_image_number;
            for i in range(recons.shape[0]):
                #if i%50==0:
                #    print('Generated image #',i);
                plt.figure(figsize=(0.28, 0.28))
                plt.axis('off');
                plt.imshow(recons[i].reshape([28,28]), origin="upper",interpolation='nearest', cmap="gray",aspect='auto');
                plt.savefig(recons_output_directory+str(_start_image_number+1).zfill(6)+'.jpg');
                _start_image_number += 1;
                plt.close();

            _start_image_number = start_image_number;
            for i in range(batch_X.shape[0]):
                #if i%50==0:
                #    print('Generated image #',i);
                plt.figure(figsize=(0.28, 0.28))
                plt.axis('off');
                plt.imshow(batch_X[i].reshape([28,28]), origin="upper",interpolation='nearest', cmap="gray",aspect='auto');
                plt.savefig(orig_output_directory+str(start_image_number+1).zfill(6)+'.jpg');
                start_image_number += 1;
                plt.close();

            print("Batch #",batch," done !!");

#generateReconstructedImages(n=100);


        

train();
#generateImages();
#sampleImages();





        