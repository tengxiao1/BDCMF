from lib.bdcmf import *
import numpy as np
import tensorflow as tf
import scipy.io
from lib.utils import *

np.random.seed(1)
tf.set_random_seed(1)
init_logging("bdcmf.log")

def load_data(data_dir):
  data = {}
  print('load data')
  if(data_dir == "data/lastfm/"):
    variables = scipy.io.loadmat(data_dir + "content.mat")
    data["content0"] = variables['X0']
    data["content1"] = variables['X1']


    data["train_users"] = load_rating(data_dir + "train_users_45.dat")
    data["train_items"] = load_rating(data_dir + "train_items_45.dat")
    data["test_users"] = load_rating(data_dir + "test_users_45.dat")
    data["test_items"] = load_rating(data_dir + "test_items_45.dat")
    data["friend"] = load_rating(data_dir + "social.dat")

  elif(data_dir == "data/delicious/"):
    content = load_rating(data_dir + "content.dat")
    data["train_users"] = load_rating(data_dir + "train_users_50.dat")
    data["train_items"] = load_rating(data_dir + "train_items_50.dat")
    data["test_users"] = load_rating(data_dir + "test_users_50.dat")
    data["test_items"] = load_rating(data_dir + "test_items_50.dat")
    data["friend"] = load_rating(data_dir + "social.dat")
    X0 = np.zeros((69226,53388))
    for i in range(len(content)):
      for j in range(len(content[i])):
        X0[i,content[i][j]] = 1
    X1 = np.zeros((69226,1867))
    for i in range(len(data['train_items'])):
      for j in range(len(data['train_items'][i])):
        X1[i,data['train_items'][i][j]] = 1
    data['content0'] = X0
    data['content1'] = X1

  print('done')
  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a[0]==0:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr

params = Params()
params.lambda_u = 0.1
params.lambda_v = 0.1
params.lambda_r = 1
params.lambda_q = 1
params.lambda_g = 1
params.a = 1
params.b = 0.01
params.c = 1
params.d = 0.01
params.M = 300
params.n_epochs = 100
params.max_iter = 1


print('lambda_v:',params.lambda_v)
print('lambda_q:',params.lambda_q)
data = load_data("data/lastfm/")
num_factors = 50
print('num_factors:',num_factors)
model = BDCMF(num_users=1892, num_items=17632, num_factors=num_factors, params=params, 
    input_dim= [11946,1892], dims=[200, 100], n_z=num_factors, activations=['sigmoid', 'sigmoid'], 
    loss_type='cross-entropy', lr=0.001, random_seed=0, print_step=10, verbose=False)
# model.load_model(weight_path="model/pretrain")
model.run(data["train_users"], data["train_items"], data['friend'], data["test_users"], data["test_items"],
  data["content0"],data['content1'], params)
model.save_model(weight_path="model/bdcmf", bdcmf_path="model/bdcmf")
