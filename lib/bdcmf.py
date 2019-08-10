import numpy as np
import lib.utils as utils
import tensorflow as tf
import sys
import math
import scipy
import scipy.io
import logging

class Params:
    """Parameters for DMF
    """
    def __init__(self):
        self.a = 1
        self.b = 0.01
        self.c = 1
        self.d = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 10
        self.lambda_r = 1
        self.lambda_q = 0.1
        self.lambda_g = 0.1
        self.max_iter = 10
        self.M = 300

        # for updating W and b
        self.lr = 0.001
        self.batch_size = 64
        self.n_epochs = 10

class BDCMF:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, 
        dims, activations, n_z=50, loss_type='cross-entropy', lr=0.1, 
        wd=1e-4, dropout=0.1, random_seed=0, print_step=50, verbose=True):
        self.m_num_users = num_users
        self.m_num_items = num_items
        self.m_num_factors = num_factors

        self.m_U = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.m_G = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)

        self.input_dim = input_dim
        self.dims = dims
        self.activations = activations
        self.lr = lr
        self.params = params
        self.print_step = print_step
        self.verbose = verbose
        self.loss_type = loss_type
        self.n_z = n_z
        self.weights = []
        self.reg_loss = 0

        self.x0 = tf.placeholder(tf.float32, [None, self.input_dim[0]], name='x0')
        self.x1 = tf.placeholder(tf.float32, [None, self.input_dim[1]], name='x1')
        self.v = tf.placeholder(tf.float32, [None, self.m_num_factors])

        x_recon = self.forward(self.x0,self.x1) 

        # loss
        # reconstruction loss
        if loss_type == 'rmse':
            self.gen_loss = tf.reduce_mean(tf.square(tf.sub(self.x0, x_recon)))
        elif loss_type == 'cross-entropy':
            x_recon = tf.nn.sigmoid(x_recon, name='x_recon')
            # self.gen_loss = -tf.reduce_mean(self.x * tf.log(tf.maximum(x_recon, 1e-10)) 
            #     + (1-self.x)*tf.log(tf.maximum(1-x_recon, 1e-10)))
            self.gen_loss = -tf.reduce_mean(tf.reduce_sum(self.x0 * tf.log(tf.maximum(x_recon, 1e-10)) 
                + (1-self.x0) * tf.log(tf.maximum(1 - x_recon, 1e-10)),1))

        self.latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq)
            - self.z_log_sigma_sq - 1, 1))
        self.v_loss = 1.0*params.lambda_v/params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v - self.z), 1))

        self.loss = self.gen_loss + self.latent_loss + self.v_loss + 2e-4*self.reg_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Initializing the tensor flow variables
        self.saver = tf.train.Saver(self.weights)
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def forward(self, x0, x1):
        W11 = tf.get_variable("W11", [self.input_dim[1], self.dims[0]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b11 = tf.get_variable("b11", [self.dims[0]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        with tf.variable_scope("inference"):
            rec = {'W1': tf.get_variable("W1", [self.input_dim[0], self.dims[0]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b1': tf.get_variable("b1", [self.dims[0]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W11': tf.get_variable("W11", [self.input_dim[1], self.dims[0]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b11': tf.get_variable("b11", [self.dims[0]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W2': tf.get_variable("W2", [self.dims[0], self.dims[1]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2': tf.get_variable("b2", [self.dims[1]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_mean': tf.get_variable("W_z_mean", [self.dims[1], self.n_z], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_mean': tf.get_variable("b_z_mean", [self.n_z], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[1], self.n_z], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        self.weights += [rec['W1'], rec['b1'],rec['W11'],rec['b11'], rec['W2'], rec['b2'], rec['W_z_mean'],
                        rec['b_z_mean'], rec['W_z_log_sigma'], rec['b_z_log_sigma']]
        self.reg_loss += tf.nn.l2_loss(rec['W1']) + tf.nn.l2_loss(rec['W2']) + tf.nn.l2_loss(rec['W11'])
        h1 = self.activate(
            tf.matmul(x0, rec['W1']) + rec['b1'] + tf.matmul(x1,rec['W11']) + rec['b11'], self.activations[0])
        h2 = self.activate(
            tf.matmul(h1, rec['W2']) + rec['b2'], self.activations[1])
        self.z_mean = tf.matmul(h2, rec['W_z_mean']) + rec['b_z_mean']
        self.z_log_sigma_sq = tf.matmul(h2, rec['W_z_log_sigma']) + rec['b_z_log_sigma']

        eps = tf.random_normal((self.params.batch_size, self.n_z), 0, 1, 
            seed=0, dtype=tf.float32)
        self.z = self.z_mean + tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)) * eps

        with tf.variable_scope("generation"):
            gen = {'W2': tf.get_variable("W2", [self.n_z, self.dims[1]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2': tf.get_variable("b2", [self.dims[1]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W1': tf.transpose(rec['W2']),
                'b1': rec['b1'],
                'W_x': tf.transpose(rec['W1']),
                'b_x': tf.get_variable("b_x", [self.input_dim[0]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}

        self.weights += [gen['W2'], gen['b2'], gen['b_x']]
        self.reg_loss += tf.nn.l2_loss(gen['W1']) + tf.nn.l2_loss(gen['W_x'])
        h2 = self.activate(
            tf.matmul(self.z, gen['W2']) + gen['b2'], self.activations[1])
        h1 = self.activate(
            tf.matmul(h2, gen['W1']) + gen['b1'], self.activations[0])
        x_recon = tf.matmul(h1, gen['W_x']) + gen['b_x']

        return x_recon

    def cdl_estimate(self, data_x0, data_x1, num_iter):
        for i in range(num_iter):
            b_x0, ids = utils.get_batch(data_x0, self.params.batch_size)
            b_x1 = data_x1[ids,:]
            _, l, gen_loss, v_loss = self.sess.run((self.optimizer, self.loss, self.gen_loss, self.v_loss),
             feed_dict={self.x0: b_x0,self.x1:b_x1, self.v: self.m_V[ids, :]})
            # Display logs per epoch step
            if i % self.print_step == 0 and self.verbose:
                print("Iter:", '%04d' % (i+1), \
                      "loss=", "{:.5f}".format(l), \
                      "genloss=", "{:.5f}".format(gen_loss), \
                      "vloss=", "{:.5f}".format(v_loss))
        return gen_loss

    def transform(self, data_x0, data_x1):
        data_en,latent_loss = self.sess.run([self.z_mean, self.latent_loss], feed_dict={self.x0: data_x0, self.x1: data_x1})
        return data_en,latent_loss

    def update(self, users, friend, items, test_users, test_items, params):
        """
        update u,v,g and return likelihood
        users: list of list
        """
        min_iter = 1
        a_minus_b = params.a - params.b
        c_minus_d = params.c - params.d
        converge = 1.0
        likelihood_origin = 0.0
        likelihood = -math.exp(20)
        it = 0
        while ((it < params.max_iter and converge > 1e-6) or it < min_iter):
            likelihood_origin = likelihood
            likelihood = 0
            # update U
            # VV^T for v_j that has at least one user liked
            ids = np.array([len(x) for x in items]) > 0
            v = self.m_V[ids]
            VVT = np.dot(v.T, v)
            XX = VVT * params.b + np.eye(self.m_num_factors) * params.lambda_u  


            ids1 =np.array([len(x) for x in friend])>0
            g = self.m_G[ids1]
            GGT = np.dot(g.T,g)
            GG = GGT * params.lambda_q * params.c


            for i in range(self.m_num_users):
                item_ids = users[i]
                friend_ids=friend[i]
                n = len(item_ids)
                m=len(friend_ids)
                if n > 0:
                    A = np.copy(XX)
                    B = np.copy(GG)
                    A += np.dot(self.m_V[item_ids, :].T, self.m_V[item_ids,:]) * a_minus_b
                    B += np.dot(self.m_G[friend_ids,:].T, self.m_G[friend_ids,:]) * c_minus_d * params.lambda_q
                    A += B
                    x = params.a * np.sum(self.m_V[item_ids, :], axis=0)
                    y= params.c * params.lambda_q * np.sum(self.m_G[friend_ids,:],axis = 0)
                    x += y
                    self.m_U[i, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * params.lambda_u * np.sum(self.m_U[i]*self.m_U[i])

            # upadate {G}
            ids=np.array([len(x) for x in users]) > 0
            u=self.m_U[ids]
            XX=np.dot(u.T,u)* params.d * params.lambda_q
            for k in range(self.m_num_users):
               friend_ids = friend[k]
               m=len(friend_ids)
               if(m>0):
                   A = np.copy(XX)
                   A += np.dot(self.m_U[friend_ids,:].T,self.m_U[friend_ids,:]) * c_minus_d * params.lambda_q
                   B = np.copy(A)
                   A += np.eye(self.m_num_factors) * params.lambda_g
                   x = params.c * np.sum(self.m_U[friend_ids,:],axis=0)
                   self.m_G[i,:] = scipy.linalg.solve(A,x)

                   # likelihood += -0.5 * m * params.c
                   # likelihood += params.c * np.sum(np.dot(self.m_U[friend_ids, :], self.m_G[k,:][:, np.newaxis]),axis=0)
                   # likelihood += -0.5 * self.m_G[k,:].dot(B).dot(self.m_G[k,:][:,np.newaxis])

                   likelihood += -0.5 * m * params.c * params.lambda_q
                   likelihood += params.c * params.lambda_q * np.sum(np.dot(self.m_U[friend_ids, :], self.m_G[k,:][:, np.newaxis]),axis=0)
                   likelihood += -0.5 * self.m_G[k,:].dot(B).dot(self.m_G[k,:][:,np.newaxis])
                   likelihood += -0.5 * params.lambda_g * np.sum(self.m_G[k,:] * self.m_G[k:])



            # update V
            ids = np.array([len(x) for x in users]) > 0
            u = self.m_U[ids]
            XX = np.dot(u.T, u) * params.b
            for j in range(self.m_num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m>0 :
                    A = np.copy(XX)
                    A += np.dot(self.m_U[user_ids,:].T, self.m_U[user_ids,:])*a_minus_b
                    B = np.copy(A)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.a * np.sum(self.m_U[user_ids, :], axis=0) + params.lambda_v * self.m_theta[j,:]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * m * params.a
                    likelihood += params.a * np.sum(np.dot(self.m_U[user_ids, :], self.m_V[j,:][:, np.newaxis]),axis=0)
                    likelihood += -0.5 * self.m_V[j,:].dot(B).dot(self.m_V[j,:][:,np.newaxis])

                    ep = self.m_V[j,:] - self.m_theta[j,:]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep)
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.lambda_v * self.m_theta[j,:]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)

                    ep = self.m_V[j,:] - self.m_theta[j,:]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep)
            it += 1
            converge = abs(1.0*(likelihood - likelihood_origin)/likelihood_origin)

            if self.verbose:
                if likelihood < likelihood_origin:
                    print("likelihood is decreasing!")

                print("[iter=%04d], likelihood=%.5f, converge=%.10f" % (it, likelihood, converge))

        return likelihood

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def run(self, users, items, friend, test_users, test_items, data_x0, data_x1, params):
        self.m_theta[:], latent_loss = self.transform(data_x0, data_x1)
        self.m_V[:] = self.m_theta
        n = data_x0.shape[0]
        elbo_list = []
        for epoch in range(params.n_epochs):
            # num_iter = int(n / params.batch_size)
            num_iter = 1
            # gen_loss = self.cdl_estimate(data_x, params.cdl_max_iter)
            gen_loss = self.cdl_estimate(data_x0, data_x1, num_iter)
            self.m_theta[:],latent_loss = self.transform(data_x0, data_x1)
            likelihood = self.update(users, friend, items, test_users, test_items, params)
            elbo = likelihood - gen_loss * n - latent_loss * n
            elbo_list.append(elbo)
            logging.info("[#epoch=%06d], elbo=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
                epoch, elbo, likelihood - gen_loss * n, gen_loss))
            self.save_model(weight_path="model/bdcmf", bdcmf_path="model/matrix/bdcmf_{}".format(epoch))

        loss_list = np.array(elbo_list)
        np.savetxt('model/elbo_list.txt',elbo_list)



    def save_model(self, weight_path, bdcmf_path=None):
        # self.saver.save(self.sess, weight_path)
        # logging.info("Weights saved at " + weight_path)
        if bdcmf_path is not None:
            scipy.io.savemat(bdcmf_path,{"m_U": self.m_U, "m_V": self.m_V, "m_theta": self.m_theta})
            logging.info("bdcmf saved at " + bdcmf_path)

    def load_model(self, weight_path, pmf_path=None):
        logging.info("Loading weights from " + weight_path)
        self.saver.restore(self.sess, weight_path)
        if pmf_path is not None:
            logging.info("Loading pmf data from " + pmf_path)
            data = scipy.io.loadmat(pmf_path)
            self.m_U[:] = data["m_U"]
            self.m_V[:] = data["m_V"]
            self.m_theta[:] = data["m_theta"]

