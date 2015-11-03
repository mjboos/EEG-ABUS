# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:32:41 2014

@author: moritz
"""

#get the data as np_array of the form chan*time*epochs

#re-test for gammas/pnulls

#import re
import os
import pystan
import pandas as pd
import numpy as np
#this is the model

hierarchic_model = """
data {
int<lower=0> npb;
int<lower=0> cases;
int<lower=0> N[npb,cases];
int<lower=0> y[npb,cases];
real X[cases];
}
parameters {
real<lower=0> mu_gam;
real<lower=0> tau_gam;
real mu_pz;
real<lower=0> tau_pz;
real<lower=0> gam[npb];
real pzero[npb];
}
model {

mu_gam ~ normal(1,10) T[0,];
mu_pz ~ normal(0,2);
gam ~ normal(mu_gam,tau_gam);
pzero ~ normal(mu_pz,tau_pz);


for (i in 1:cases)
{
for (j in 1:npb)
y[j,i] ~ binomial_logit(N[j,i],X[i]*gam[j] + (1-gam[j])*pzero[j]);
}
}
"""

#list the files, assuming a folder structure of cwd/Data/...

#there are 16 vps
npb = 16
#and 14 (12 without 1st ball) different cases for the binomial distribution
cases = 5
path = '/home/mboos/Work/Bayesian Updating/Old Data/'

freq_urn_cc = np.loadtxt('/home/mboos/Work/Bayesian Updating/Old Data/freq_urn_cc.txt')
rare_urn_cc = np.loadtxt(path+'rare_urn_cc.txt')
freq_urn_uc = np.loadtxt(path+'freq_urn_uc.txt')
rare_urn_uc = np.loadtxt(path+'rare_urn_uc.txt')
freq_urn_uu = np.loadtxt(path+'freq_urn_uu.txt')
rare_urn_uu = np.loadtxt(path+'rare_urn_uu.txt')
freq_urn_cu = np.loadtxt(path+'freq_urn_cu.txt')
rare_urn_cu = np.loadtxt(path+'rare_urn_cu.txt')

N_cc = freq_urn_cc + rare_urn_cc
N_uc = freq_urn_uc + rare_urn_uc
N_cu = freq_urn_cu + rare_urn_cu
N_uu = freq_urn_uu + rare_urn_uu


#also, we need the actual posterior probabilities as logits


prior = 0.3
likelihood = 0.9

posterior = np.array([ (prior*likelihood**i*(1-likelihood)**(4-i)) / ((prior*likelihood**i*(1-likelihood)**(4-i))+((1-prior)*(1-likelihood)**i*likelihood**(4-i))) for i in xrange(5) ])
lo_x = np.log(posterior/(1-posterior))
#and again drop first ball/2 cases


#%%
#ONLY FOR 3 BALLS AND MODEL WITH NON-UNIQUE POSTERIORS
#create the data-structure
data_cc = {"npb" : npb,"cases":cases,"N":N_cc.astype(int),"y":rare_urn_cc.astype(int),"X":lo_x}
#fit now
fit = pystan.stan(model_code=hierarchic_model, data=data_cc,iter=40000, chains=20)
#%%
#MODEL WITH ONLY UNIQUE POSTERIORS 
#try to compress X, so same values are added
#first round
data_uc = {"npb" : npb,"cases":cases,"N":N_uc.astype(int),"y":rare_urn_uc.astype(int),"X":lo_x}

fit = pystan.stan(model_code=hierarchic_model, data=data_uc,iter=40000, chains=20)
