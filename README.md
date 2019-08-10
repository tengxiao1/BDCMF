# BDCMF

The is the code for"Bayesian Deep Collaborative Matrix Factorization(BDCMF)" (under review). It consists of two parts: a Matlab component and a Python component. 

Requirements:

    Python 3.4
    Tensorflow 1.7
    Matlab
    
To run "train.py", it generates two '.mat' files: U.mat and V.mat which are latent factors of users and items, respectively. Then you can use the two files to evaluate recommendation performance on test data.

#Baselines     
For strong baseline CDL, we use the code provided the author. You can dowlond from https://github.com/js05212/CDL    
For PossMF-CS, you can dowlond from https://github.com/zehsilva/poissonmf_cs   
For NCF, you can dowlond from https://github.com/hexiangnan/neural_collaborative_filtering    
    

