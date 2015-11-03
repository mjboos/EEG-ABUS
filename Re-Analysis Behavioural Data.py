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

mu_gam ~ normal(1,1) T[0,];
mu_pz ~ normal(0,10);
gam ~ normal(mu_gam,tau_gam);
pzero ~ normal(mu_pz,tau_pz);


for (i in 1:cases)
{
for (j in 1:npb)
y[j,i] ~ binomial_logit(N[j,i],X[i]*gam[j] + (1-gam[j])*pzero[j]);
}
}
"""

hierarchic_model_kolossa = """
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
real<lower=0> gam[npb];
}
model {

mu_gam ~ normal(1,1) T[0,];
gam ~ normal(mu_gam,tau_gam);


for (i in 1:cases)
{
for (j in 1:npb)
y[j,i] ~ binomial_logit(N[j,i],X[i]*gam[j]);
}
}
"""

hierarchic_model_kolossa2 = """
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
real<lower=0> gam[npb];
}
model {

mu_gam ~ normal(1,1) T[0,];
gam ~ normal(mu_gam,tau_gam);


for (i in 1:cases)
{
for (j in 1:npb)
y[j,i] ~ binomial_logit(N[j,i],X[i]*gam[j]);
}
}
"""

#list the files, assuming a folder structure of cwd/Data/...
files = os.listdir("Data/newdata/")

#now get all the data for exp80
pattern = "80exp"

exp80_y = pd.DataFrame({0:14*[0]})
exp80_n = pd.DataFrame({0:14*[0]})

for fn in files:
    if pattern in fn:
        tmp = pd.read_table("Data/newdata/"+fn,sep=" ",header=None)
        exp80_y[int(fn[5:7])] = tmp[0]
        exp80_n[int(fn[5:7])] = tmp[1]
       
#now sort them
exp80_n.sort_index(axis=1,inplace=True)
exp80_y.sort_index(axis=1,inplace=True)

#for easier use convert to array/CHANGE THIS LATER

exp80_n = exp80_n.as_matrix() 
exp80_y = exp80_y.as_matrix()

#and also drop the first column

exp80_n = exp80_n[:,1:]
exp80_y = exp80_y[:,1:]

#pop first 2 cases, so no first ball
exp80_n = exp80_n[2:,:]
exp80_y = exp80_y[2:,:]
#03a or 03?
#now you can start re-analysing the data
#there are 16 vps
npb = 16
#and 14 (12 without 1st ball) different cases for the binomial distribution
cases = 12

#also, we need the actual posterior probabilities as logits


prior = 0.2
likelihood = 0.7

x = np.array([(prior*(likelihood**j)*((1-likelihood)**(i-j)))/(((1-prior)*((likelihood**(i-j))*((1-likelihood)**j)))+(prior*(likelihood**j)*((1-likelihood)**(i-j)))) for i in xrange(1,5) for j in xrange(i+1)])
lo_x = np.log(x/(1-x))
#and again drop first ball/2 cases
lo_x = lo_x[2:]

#transpose so rows are vps and columns are cases
exp80_n = exp80_n.T
exp80_y = exp80_y.T

#%%
#ONLY FOR 3 BALLS AND MODEL WITH NON-UNIQUE POSTERIORS
#create the data-structure
data_exp80 = {"npb" : npb,"cases":cases,"N":exp80_n,"y":exp80_y,"X":lo_x}
#fit now
fit = pystan.stan(model_code=hierarchic_model_kolossa, data=data_exp80,
                 iter=10000, chains=10)
#%%
#MODEL WITH ONLY UNIQUE POSTERIORS 
#try to compress X, so same values are added
#first round

lo_x = around(lo_x,3)

#now add n and y with the same X (with cool comprehension)

exp80_n = array( [ sum(exp80_n[:,lo_x==lx],1) for lx in unique(lo_x)]).T
exp80_y = array( [ sum(exp80_y[:,lo_x==lx],1) for lx in unique(lo_x)]).T

#update cases
cases = exp80_n.shape[1]

data_exp80 = {"npb" : npb,"cases":cases,"N":exp80_n,"y":exp80_y,"X":unique(lo_x)}



#%%
fit = pystan.stan(model_code=hierarchic_model, data=data_exp80,
                 iter=40000, chains=20)
#%%