# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
"""
Created on Wed Jan 15 20:32:41 2014

@author: moritz
"""



#import re
from helper_functions
import matplotlib.colors as colors
from functools import partial
import os
import pystan
import pandas as pd
from math import fsum
from scipy.io import loadmat
import scipy.stats
from sklearn.decomposition import FastICA
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV,RidgeCV,Ridge
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVR,NuSVR
from scipy.fftpack import fft
#change this model


chan_list = np.array(["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FCz","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","TP9","TP10","Oz","O2","O1"])

#%%
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

#create a supervised PCA class
class UnivariateCorrelationThreshold(BaseEstimator):
    """ ... """
    #assumes standardized values
    
    def __init__(self,correlation_threshold=0.2):
        self.correlation_threshold = correlation_threshold
    
    def fit(self,X,y,n_jobs = 1):
        linreg = LinearRegression()
        self.correlations = np.array([linreg.fit(x[:,None],y).coef_ for x in X.T])
        
    def transform(self,X):
        try:
            self.correlations
        except NameError:
            print("Need to fit first")
        return X[:,(np.abs(self.correlations) >= self.correlation_threshold)[:,0] ]


uct = UnivariateCorrelationThreshold()
scaler = StandardScaler()
pca = PCA()
linreg = LinearRegression()
supervisedPCA = Pipeline([('scaler',scaler),('correlation threshold',uct),('PCA',pca),('Linear Regression',linreg)])

sPCA_params = {'uct__correlation_threshold' : [0.2,0.3,0.4,0.5],'pca__n_components' : [3,5,7,10,15,20]}

#%%
#very simple model with most extreme case: average the amplitude over the period 0-500 ms (or similar)
#then regress with log-odds or something similar

eeg_model_for_one = """
data {

int<lower=0> n_ep;
int<lower=0> n_chan;
int<lower=0> n_bin;
//more efficient?
matrix[n_chan,n_ep] dats[n_bin];
vector[n_ep] X;
}
parameters {
//more efficient?
matrix[n_chan,n_bin] p_vec;

//define variance
real<lower=0,upper=1000> sigma;
}
model {

//prior here, probably normal
for (i in 1:n_bin)
    p_vec[i] ~ normal(0,10);

for (i in 1:n_chan)
    for (j in 1:n_bin)
        dats[j,i] ~ normal(X*p_vec[i,j],sigma);

}
"""
#model needs to be improved
eeg_model_for_all = """
data {

int<lower=0> n_ep;
int<lower=0> n_chan;
int<lower=0> n_bin;
int<lower=0> n_pb;
//more efficient?
real dats[n_chan,n_bin,n_ep];
int<lower=1,upper=n_pb> pbs[n_ep]; 
vector[n_ep] X;
}
parameters {
//more efficient?
real p_vec[n_chan,n_bin,n_pb];
//define variance
real<lower=0,upper=1000> sigma;
real h_mu[n_chan,n_bin];
real<lower=0> h_sigma;

}
model {

for (i in 1:n_bin)
    for (j in 1:n_chan)
        h_mu[j,i] ~normal(0,10);


//prior here, probably normal
for (i in 1:n_bin)
    for (j in 1:n_chan)    
        p_vec[j,i] ~ normal(h_mu[j,i],h_sigma);


for (i in 1:n_chan)
    for (j in 1:n_bin)
        for (p in 1:n_ep)
            dats[i,j,p] ~ normal(X[p]*p_vec[i,j,pbs[p]],sigma); //very inefficient

}
"""
#%%

eeg_model_all_reversed = """
data {

int<lower=0> n_ep;
int<lower=0> n_chan;
int<lower=0> n_bin;
int<lower=0> n_pb;
//more efficient?
matrix[n_chan,n_bin] amplitudes[n_ep];
int<lower=1,upper=n_pb> pbs[n_ep]; 
real X[n_ep];
}
parameters {
//more efficient?
row_vector[n_chan] theta_chan[n_pb];
vector[n_bin] theta_time[n_pb];

//define variance
real<lower=0,upper=1000> sigma;


}
model {



//prior here, probably normal
for (i in 1:n_pb)
{
    theta_chan[i] ~ normal(0,15);
    theta_time[i] ~ normal(0,15);
}

//you can make it even more efficient
for (p in 1:n_ep)
    X[p] ~ normal(theta_chan[pbs[p]]*amplitudes[p]*theta_time[pbs[p]],sigma);

}
"""

#%%

eeg_model_all_reversed_std = """
data {

int<lower=0> n_ep;
int<lower=0> n_chan;
int<lower=0> n_bin;
int<lower=0> n_pb;
//more efficient?
matrix[n_chan,n_bin] amplitudes[n_ep];
int<lower=1,upper=n_pb> pbs[n_ep]; 
real X[n_ep];
}
parameters {
//more efficient?
row_vector[n_chan] theta_chan[n_pb];
vector[n_bin] theta_time[n_pb];
real intercept[n_pb];


//define variance
real<lower=0> sigma_std;


}
model {

sigma_std ~ cauchy(0,5);

//prior here, probably normal
for (i in 1:n_pb)
{
    theta_chan[i] ~ normal(0,10);
    theta_time[i] ~ normal(0,10);
}
intercept ~ normal(0,10);

//you can make it more efficient

for (p in 1:n_ep)
    X[p] ~ normal(intercept[pbs[p]]+theta_chan[pbs[p]]*amplitudes[p]*theta_time[pbs[p]],sigma_std);

}
"""

#%%
#features are
#trials * (chan*bin)
#non-hierarchic, non pb difference

eeg_model_reversed_ft_std = """
data {

int<lower=0> n_ep;
int<lower=0> n_ft;
matrix[n_ep,n_ft] amplitudes;
vector[n_ep] X;
}
parameters {
vector[n_ft] theta;

//define variance
real<lower=0> sigma_std;


}
model {

sigma_std ~ cauchy(0,5);

theta ~ normal(0,15);

X ~ normal(amplitudes*theta,sigma_std);

}
"""

#%%
#features are
#trials * (chan*bin)
#non-hierarchic, non pb difference

eeg_model_kld_reversed_ft_std = """
data {

int<lower=0> n_ep;
int<lower=0> n_ft;
matrix[n_ep,n_ft] amplitudes;
vector[n_ep] X;
}
parameters {
vector[n_ft] theta;

//define variance
real<lower=0> sigma_std;


}
model {

sigma_std ~ cauchy(0,5);

theta ~ normal(0,15);

X ~ normal(amplitudes*theta,sigma_std);

}
"""

#%%
#EEG-model for encoding (hierarchical) with grand average ERP
eeg_model_for_all = """
data {


int<lower=0> n_chan;
int<lower=0> n_bin;
int<lower=0> n_pb;
//more efficient?
real dats[n_chan,n_bin,n_ep];
int<lower=1,upper=n_pb> pbs[n_ep]; 
vector[n_ep] X;
}
parameters {
//more efficient?
real p_vec[n_chan,n_bin,n_pb];
//define variance
real<lower=0,upper=1000> sigma;
real h_mu[n_chan,n_bin];
real<lower=0> h_sigma;

}
model {

for (i in 1:n_bin)
    for (j in 1:n_chan)
        h_mu[j,i] ~normal(0,10);


//prior here, probably normal
for (i in 1:n_bin)
    for (j in 1:n_chan)    
        p_vec[j,i] ~ normal(h_mu[j,i],h_sigma);


for (i in 1:n_chan)
    for (j in 1:n_bin)
    dats[i,j,p] ~ normal(X[p]*p_vec[i,j,pbs[p]],sigma); //very inefficient

}
"""
#%%
#maybe do some visual inspection first
#link data to log-odds


likelihood = { 1 : 0.3, 2: 0.7 }
prior = 0.5

f_ep = get_failed_epochs("TSVP01_50exp_Raw Data_markers.txt")
bclass = get_bclass("TSVP01_50exp_Raw Data_markers.txt",prior,likelihood)

#example data
e50test = loadmat("/home/mboos/Work/Bayesian Updating/Data EEG/VP01_exp50_epochs.mat")
e50test["ep_data"] = e50test["ep_data"][:30,:,:]

#%%
#Re-Reference by Average per Timepoint
time_means = np.mean(e50test["ep_data"],axis=0)
rr_data = e50test["ep_data"] - time_means

#%%
#test: average over n datapoints, split in k groups
#uses rereferenced dataplot
#strip the first 200 ms --> 200/4 = 50 datapoints

rr_data = rr_data[:,50:,:]
k = 29
e50_mean = np.array([ np.mean(ar,axis=1) for ar in np.array_split(rr_data,k,axis=1)]).swapaxes(0,1)

#%%
#construct log-odds difference array

bc_diff = get_bclass_diff("TSVP01_50exp_Raw Data_markers.txt",likelihood)



#%%
#only use the _sequences_ that have no failed epochs
#new bclass, e50_mean objects

e50_mean = e50_mean[:,:,(f_ep[bclass[:,0]>0])==1]
bclass = bclass[f_ep==1,:]

#%%


data_e50_01 = { "n_ep" : e50_mean.shape[2],"n_bin" : e50_mean.shape[1] , "n_chan" : 30, "dats" : e50_mean.swapaxes(0,1), "X" : bclass[:,1]}
#fit = pystan.stan(model_code=eeg_model_for_one, data=data_e50_01,
#                 iter=1000, chains=5)
        
#%%
#get mean and sd for electrode for timepoint over all epochs
    
z_tran = (e50_mean - np.mean(e50_mean,axis=2,keepdims=True)) / np.std(e50_mean,axis=2,keepdims=True)

#%%
#fit with z-transformed amplitude data

data_e50_zt = { "n_ep" : z_tran.shape[2],"n_bin" : z_tran.shape[1] , "n_chan" : 30, "dats" : z_tran.swapaxes(0,1), "X" : np.round(bclass[:,1],decimals=2)}
fit = pystan.stan(model_code=eeg_model_for_one, data=data_e50_zt,iter=5000, chains=5)



#%%
#######################################
#NOW FOR ALL PARTICIPANTS
########################################
likelihood = { 1 : 0.3, 2: 0.7 }
prior = 0.2

path = "/home/mboos/Work/Bayesian Updating/Data/"
path_mat = "/home/mboos/Work/Bayesian Updating/Data EEG/"

files = os.listdir(path)
mat_files = os.listdir(path_mat)

pattern_TS = "80exp"
pattern = "exp80"

#uncomment and run when .mat files exist
#how many bins
nbin = 48 #25 ms


bc_dict = dict()
mat_dict = dict()

#%%
#strip VEOH,HEOG electrodes
#thoroughly test this

#bc_dict has only one column in an entry

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        bclass = get_bclass(fn,prior,likelihood)
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(bclass[:,0]>0):
            continue
        mat_dict[fn[4:6]] = curr[:,:,(f_ep[bclass[:,0]>0])==1]
        bc_dict[fn[4:6]] = bclass[f_ep==1,1]

        
#%%
#without first two balls
#strip VEOH,HEOG electrodes

#maybe test this again

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        bclass = get_bclass(fn,prior,likelihood)
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(bclass[:,0]>0):
            continue
        if f_ep.size % 4 != 0:
            print "error at " + fn
            break
        mat_dict[fn[4:6]] = curr[:,:,np.logical_and((f_ep[bclass[:,0]>0])==1,np.tile([0,0,1,1],f_ep.size/4)[bclass[:,0]>0]==1)]
        bc_dict[fn[4:6]] = bclass[np.logical_and(f_ep==1,np.tile([0,0,1,1],f_ep.size/4)==1),1]
        
        
#%%
#get the KULLBACK-LEIBLER-DIVERGENCE
#KLD for last 2 balls
kld_dict = dict()

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        kld = kld_vec(fn,prior,likelihood)
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(kld[:,0]>0):
            continue
        if f_ep.size % 4 != 0:
            print "error at " + fn
            break
        mat_dict[fn[4:6]] = curr[:,:,np.logical_and((f_ep[kld[:,0]>0])==1,np.tile([0,0,1,1],f_ep.size/4)[kld[:,0]>0]==1)]
        kld_dict[fn[4:6]] = kld[np.logical_and(f_ep==1,np.tile([0,0,1,1],f_ep.size/4)==1),1]
        
        
#KLD for all balls
#STILL TODO

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        kld = kld_vec(fn,prior,likelihood)
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(kld[:,0]>0):
            continue
        if f_ep.size % 4 != 0:
            print "error at " + fn
            break
        mat_dict[fn[4:6]] = curr[:,:,np.logical_and((f_ep[kld[:,0]>0])==1,np.tile([0,0,1,1],f_ep.size/4)[kld[:,0]>0]==1)]
        kld_dict[fn[4:6]] = kld[np.logical_and(f_ep==1,np.tile([0,0,1,1],f_ep.size/4)==1),1]
        
 
#%%
gammas_exp80 = np.array([2.9,2.5,2.0,2.2,2.5,1.7,3.8,3.1,3.8,2.6,3.0,3.9,1.9,3.9,1.7,1.6]).T
#pnull in log-odds
pnull_exp80 = np.array([-1.5,-1.6,-1.9,0.2,-0.8,-1.8,-1.0,0.8,-1.6,-2.4,-2.5,-1.0,-2.4,-1.0,-0.2,-1.5]).T
#now you can transform the bc_dict values for every person

#%%
for k in bc_dict.keys():
    bc_dict[k] = bc_dict[k]*gammas_exp80[int(k)-1] + pnull_exp80[int(k)-1]*(1-gammas_exp80[int(k)-1])

#%%
#prepare data, n_ep, n_chan, n_bin, n_pb
n_ep_list = [mat_dict[k].shape[2] for k in sorted(mat_dict.keys())]
data_all_e80 = { "n_ep" : sum(n_ep_list),"n_pb" : 15, "n_bin" : nbin,"n_chan" : 30, "dats" : np.concatenate([mat_dict[m] for m in sorted(mat_dict.keys())],axis=2),"pbs" : np.repeat(np.arange(1,16),n_ep_list),"X" : np.concatenate([bc_dict[k] for k in sorted(bc_dict.keys())])}
#%%
#kills whole process
#fit = pystan.stan(model_code=eeg_model_for_all, data=data_all_e80,iter=500, chains=2)

#%%
#for P3a at 250-450ms
#use time bins 6-9 inkl.
#use appropriate channels, for now only use Fz (5.) and FCz (14.)

data_dict  = { m : mat_dict[m][[7,8,9,10,12,13,14],10:18,:] for m in mat_dict.keys()}



#%%
#prepare data, n_ep, n_chan, n_bin, n_pb
#change this so it's only part of the data
n_ep_list = [data_dict[k].shape[2] for k in sorted(data_dict.keys())]
data_all_e80 = { "n_ep" : sum(n_ep_list),"n_pb" : 15, "n_bin" : 8,"n_chan" : 7, "dats" : np.concatenate([data_dict[m] for m in sorted(data_dict.keys())],axis=2),"pbs" : np.repeat(np.arange(1,16),n_ep_list),"X" : np.concatenate([kld_dict[k] for k in sorted(kld_dict.keys())])}

#%%

fit = pystan.stan(model_code=eeg_model_for_all, data=data_all_e80,iter=1000, chains=10)



#%%


#%%
#data for reverse model

n_ep_list = [data_dict[k].shape[2] for k in sorted(data_dict.keys())]
data_reverse_e80 = { "n_ep" : sum(n_ep_list),"n_pb" : 15, "n_bin" : 8,"n_chan" : 2, "amplitudes" : np.concatenate([data_dict[m] for m in sorted(data_dict.keys())],axis=2).swapaxes(0,2).swapaxes(1,2),"pbs" : np.repeat(np.arange(1,16),n_ep_list),"X" : np.concatenate([bc_dict[k] for k in sorted(bc_dict.keys())])}

#%%
#standardize data for reversed model
#normalize over TIMEPOINTS, so mean of timepoints of a channel/epoch is 0 and sd is 1

data_reverse_e80["amplitudes"] = (data_reverse_e80["amplitudes"] - np.mean(data_reverse_e80["amplitudes"],keepdims=True,axis=2))/np.std(data_reverse_e80["amplitudes"],axis=2,keepdims=True)

data_reverse_e80["X"] = (data_reverse_e80["X"] - np.mean(data_reverse_e80["X"]))/np.std(data_reverse_e80["X"])


#%%
#fit reverse model
fit = pystan.stan(model_code=eeg_model_all_reversed_std, data=data_reverse_e80,iter=500, chains=2)

#%%
######################################


#for P3a at 250-450ms
#use time bins 6-9 inkl.
#try to use more channels

data_dict  = { m : mat_dict[m][[7,8,9,10,12,13,14],10:18,:] for m in mat_dict.keys()}

#%%

#reverse model with feature vector

n_ep_list = [data_dict[k].shape[2] for k in sorted(data_dict.keys())]

data_reverse_ft_e80 = { "n_ep" : sum(n_ep_list),"n_ft" : 7*8, "amplitudes" : np.vstack([data_dict[m][...,i].flatten() for m in sorted(data_dict.keys()) for i in xrange(data_dict[m].shape[-1])]),"X" : np.concatenate([kld_dict[k] for k in sorted(kld_dict.keys())])}

#%%
#normalize features

data_reverse_ft_e80["amplitudes"] = (data_reverse_ft_e80["amplitudes"] - np.mean(data_reverse_ft_e80["amplitudes"],keepdims=True,axis=0))/np.std(data_reverse_ft_e80["amplitudes"],axis=0,keepdims=True)

#add ones for intercept and add 1 to n_ft
data_reverse_ft_e80["amplitudes"] = np.hstack([ np.ones((data_reverse_ft_e80["n_ep"],1)), data_reverse_ft_e80["amplitudes"]])

data_reverse_ft_e80["n_ft"] += 1

#uncomment to standardize the log-odds posteriors
#data_reverse_ft_e80["X"] = (data_reverse_ft_e80["X"] - np.mean(data_reverse_ft_e80["X"]))/np.std(data_reverse_ft_e80["X"])

#%%
#fit reverse model with feature vector

fit = pystan.stan(model_code=eeg_model_reversed_ft_std, data=data_reverse_ft_e80,iter=1000, chains=10)


#%%
#get posterior specific ERPs
#chan x posterior x timebin

ERP_avr_dict = {k : np.array([ get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],c) for c in xrange(mat_dict[k].shape[0])]).swapaxes(1,2) for k in mat_dict.keys() }

#%%
#construct data_dict with chan X times X posteriors, construct similarly post_dict with posterior values

data_avr_dict  = { m : ERP_avr_dict[m][[7,8,9,10,12,13,14],10:18,:] for m in ERP_avr_dict.keys()}

post_dict = { k : np.unique(np.round(bc_dict[k],2)) for k in bc_dict.keys()}

n_ep_list = [data_avr_dict[k].shape[2] for k in sorted(data_avr_dict.keys())]

data_reverse_ft_avr_e80 = { "n_ep" : sum(n_ep_list),"n_ft" : 7*8, "amplitudes" : np.vstack([data_avr_dict[m][...,i].flatten() for m in sorted(data_avr_dict.keys()) for i in xrange(data_avr_dict[m].shape[-1])]),"X" : np.concatenate([post_dict[k] for k in sorted(post_dict.keys())])}

#%%

#normalize features averaged model

data_reverse_ft_avr_e80["amplitudes"] = (data_reverse_ft_avr_e80["amplitudes"] - np.mean(data_reverse_ft_avr_e80["amplitudes"],keepdims=True,axis=0))/np.std(data_reverse_ft_avr_e80["amplitudes"],axis=0,keepdims=True)

#add ones for intercept and add 1 to n_ft
data_reverse_ft_avr_e80["amplitudes"] = np.hstack([ np.ones((data_reverse_ft_avr_e80["n_ep"],1)), data_reverse_ft_avr_e80["amplitudes"]])

data_reverse_ft_avr_e80["n_ft"] += 1

#uncomment to standardize the log-odds posteriors
#data_reverse_ft_avr_e80["X"] = (data_reverse_ft_avr_e80["X"] - np.mean(data_reverse_ft_avr_e80["X"]))/np.std(data_reverse_ft_avr_e80["X"])

#%%
#averaged fit

fit = pystan.stan(model_code=eeg_model_reversed_ft_std, data=data_reverse_ft_avr_e80,iter=1000, chains=10)





#%%
#test for correlations

c_list = []

for i in xrange(data_reverse_ft_e80["amplitudes"].shape[1]):
    c_list.append(scipy.stats.pearsonr(data_reverse_ft_e80["X"],data_reverse_ft_e80["amplitudes"][:,i]))
    


#%%
#plot data by posterior probability for certain channel, x time, y amplitude, several curves
#plot_ERPs_per_posterior(e50_mean,bclass[:,1],chan=4)
#show()

#%%
#get average ERP per post per pb

figure(1)
for k in mat_dict.keys():
    avr_ERP_p_Post = get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],chan=4)
    
    for sp,n in enumerate(np.unique(np.round(bc_dict[k],decimals=2))):
        subplot(3,3,sp+1)
        plot(np.linspace(0,1000,avr_ERP_p_Post.shape[1]),avr_ERP_p_Post[sp,:])
        title("Posterior ="+str(n))

#%%

figure(1)
for sp,n in enumerate(np.unique(np.round(bclass[:,1],decimals=2))):
    plot(np.linspace(0,1200,avr_ERP_p_Post.shape[1]),avr_ERP_p_Post[sp,:])
    title("Posterior ="+str(n))


#%%

#########################################################
#   INTERVAL FEATURE EXTRACTION                         #
#                                                       #
#########################################################
# do interval feature extraction for the relevant channels and timebins
# use the model with no individual differences ft_std

#IFE_dict = { m : np.vstack([np.concatenate([interval_feature_extraction(mat_dict[m][i,10:18,j]).flatten() for i in xrange(mat_dict[m].shape[0])]) for j in xrange(mat_dict[m].shape[2])]) for m in sorted(mat_dict.keys()) }


#too many features


#%%
#ICA features

#try ICA to extract components per person, cluster them, and use them as features
#keep all datapoints after the event

#######################################
#NOW FOR ALL PARTICIPANTS
#FOR ICA OR FFT
########################################
likelihood = { 1 : 0.3, 2: 0.7 }
prior = 0.2

path = "/home/mboos/Work/Bayesian Updating/Data/"
path_mat = "/home/mboos/Work/Bayesian Updating/Data EEG/"

files = os.listdir(path)
mat_files = os.listdir(path_mat)

pattern_TS = "80exp"
pattern = "exp80"

bc_dict = dict()
mat_dict = dict()
brar_dict = dict()

#%%
#strip VEOH,HEOG electrodes




for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        bclass = get_bclass(fn,prior,likelihood)
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,25:,:])
        if curr.shape[2] != sum(bclass[:,0]>0):
            continue
        mat_dict[fn[4:6]] = curr[:,:,(f_ep[bclass[:,0]>0])==1]
        bc_dict[fn[4:6]] = bclass[f_ep==1,1]

        

#%%
#for one person
#create an ICA instance
#for testing purposes get the mixing matrix/scalp map of two persons



#test_set_1 = np.reshape(np.swapaxes(mat_dict["01"],1,2),(30,-1))
#test_set_2 = np.reshape(np.swapaxes(mat_dict["02"],1,2),(30,-1))
#
#ica.fit(test_set_1.T)
#m1 = ica.mixing_
#ica.fit(test_set_2.T)
#m2 = ica.mixing_
#norm1 = np.apply_along_axis(np.linalg.norm,0,m1)
#norm2 = np.apply_along_axis(np.linalg.norm,0,m2)
#nm1 = np.apply_along_axis(lambda x : x/norm1,1,m1)
#nm2 = np.apply_along_axis(lambda x : x/norm2,1,m2)
#sim_mat = np.dot(nm1.T,nm2)
##use ICA for every epoch and all datapoints
#now test this for every person

################################
#NOW do an ICA for every participant

#%%
ica = FastICA()

#ica_dict = dict()
#for k in sorted(mat_dict.keys()):
#	ica.fit(np.reshape(np.swapaxes(mat_dict["01"],1,2),(30,-1)).T)
#	ica_dict[k] = ica.mixing_
#
##now think about a good way to compare them	
#do an ICA over all pbs
all_sources = ica.fit_transform(np.hstack([np.reshape(np.swapaxes(mat_dict[k],1,2),(30,-1)) for k in sorted(mat_dict.keys())]).T)
all_pb_mixing = ica.mixing_


#now order the ICs of the individuals according to their similarity to the mixing matrix of all pbs
#but first: you can try to do an analysis with the ICs of all persons

#maybe check ICs first
#plot the mean ICs grouped by different values of the posterior
#first round them up to 2 decimals

for k in bc_dict.keys():
    bc_dict[k] = np.round(bc_dict[k],decimals=2)

#%%

sizes_list = [ mat_dict[m].shape[2] for m in sorted(mat_dict.keys()) ]
pb_epoch_index = [ sum(sizes_list[:i])*250 for i in xrange(1,len(sizes_list))]

pb_epoch_index = [ sum(sizes_list[:i])*75 for i in xrange(1,len(sizes_list))]
#now get the activations back into original form

all_sources = all_sources.T
pb_list_of_sources = np.hsplit(all_sources,pb_epoch_index)

source_dict = { k : np.reshape(pb_list_of_sources[i],(30,-1,250)) for i,k in enumerate(sorted(mat_dict.keys())) }

source_dict = { k : np.reshape(pb_list_of_sources[i],(30,-1,75)) for i,k in enumerate(sorted(mat_dict.keys())) }
#now get the means

mean_source_per_post = dict()

#creates a dictionary of source activations per posterior probability

for p in np.unique(bc_dict["01"]):
    mean_sources = np.ones((1,250))
    for i in xrange(source_dict["01"].shape[0]):
        mean_sources = np.vstack([mean_sources,np.mean(np.vstack([source_dict[k][i,bc_dict[k]==p] for k in sorted(source_dict.keys()) if p in np.unique(bc_dict[k])]),axis=0)])
    mean_source_per_post[p] = mean_sources[1:,:]

#now do the same for kullback-leibler-divergence

#now order the components by their explained variance
#compute predicted EEG timeseries (column of mixing matrix of that component * its activations) 
#and cross-correlate with the real timeseries

correlations = np.empty((30))
for i in xrange(all_pb_mixing.shape[1]):
    predicted_ts = np.dot(np.reshape(all_pb_mixing[:,i],(all_pb_mixing[:,i].shape[0],1)),np.reshape(all_sources[i,:],(1,all_sources[i,:].shape[0])))
    correlations[i] = np.corrcoef(np.vstack([predicted_ts.flatten(),np.hstack([np.reshape(np.swapaxes(mat_dict[k],1,2),(30,-1)) for k in sorted(mat_dict.keys())]).flatten()]))[0,1]

corr_idx_sorted = np.argsort(correlations)

#%%

kbin_source_per_post = { s : k_bin_average(mean_source_per_post[s][:150],20) for s in mean_source_per_post.keys() }

#plots the averaged sources for the first 10 components with the highest explained variance for each posterior (different colors)
#for i in xrange(10):
#    plt.subplot(2,5,i+1)
#    for j,k in enumerate(sorted(kbin_source_per_post.keys())):
#        plt.plot(kbin_source_per_post[k][list(reversed(corr_idx_sorted))[i],:],['r','g','b','c','m','y','k','0.25','0.75'][j])
#
#
#for i,k in enumerate(sorted(kbin_source_per_post.keys())):
#    plt.subplot(3,3,i+1)
#    plt.title(str(k))
#    for j in xrange(5):
#        plt.plot(kbin_source_per_post[k][list(reversed(corr_idx_sorted))[j],:],['r','g','b','c','m','y','k','0.25','0.75'][j])
#
#for i in xrange(3):
#    plt.subplot(3,1,i+1)
#    for j,k in enumerate(sorted(kbin_source_per_post.keys())):
#        plt.plot(kbin_source_per_post[k][[12,7,6][i],:],['r','g','b','c','m','y','k','0.25','0.75'][j],label=str(k))
#




#%%
#TODO: Kullback-Leibler-Divergence re-check

#now test this with the simple reverse ft model
#averages and uses only the data from 100 to 500 ms
#now each entry is again components x times x epochs
source_d_data = { k : k_bin_average(np.swapaxes(source_dict[k],1,2),40)[list(reversed(corr_idx_sorted))[:10],4:20,:] for k in source_dict.keys() }
n_ep_list = sizes_list

#use this for the normal posterior model
source_data = { "n_ep" : sum(n_ep_list),"n_ft" : 16*10, "amplitudes" : np.vstack([source_d_data[m][...,i].flatten() for m in sorted(source_d_data.keys()) for i in xrange(source_d_data[m].shape[-1])]),"X" : np.concatenate([bc_dict[k] for k in sorted(bc_dict.keys())])}

source_data["amplitudes"] = np.hstack([np.ones((source_data["amplitudes"].shape[0],1)),source_data["amplitudes"]])
source_data["n_ft"] += 1
#%%

#start model
#fit = pystan.stan(model_code=eeg_model_reversed_ft_std, data=source_data,iter=500, chains=5)

#Kullback Leibler Divergence
nbin = 1#doesnt matter, curr isnt used, only to ensure that we have an estimate of the number of epochs 
kld_dict = dict()

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        kld = kld_vec(fn,prior,likelihood)
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(kld[:,0]>0):
            continue
        if f_ep.size % 4 != 0:
            print "error at " + fn
            break
        kld_dict[fn[4:6]] = kld[f_ep==1,1]


#%%
nbin = 1#doesnt matter, curr isnt used, only to ensure that we have an estimate of the number of epochs 
predictive_surprise_dict = dict()

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        predsurp = predictive_surprise_vec(fn,likelihood)
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(predsurp[:,0]>0):
            continue
        if f_ep.size % 4 != 0:
            print "error at " + fn
            break
        predictive_surprise_dict[fn[4:6]] = predsurp[f_ep==1,1]


#%%
#for distorted KLDs

nbin = 1#doesnt matter, curr isnt used, only to ensure that we have an estimate of the number of epochs 
kld_dict = dict()

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        kld = kld_vec_distort(fn,prior,likelihood,gammas_exp80[int(fn[4:6])-1],pnull_exp80[int(fn[4:6])-1])
        #re-reference and average over k bins
        #also think about better way to ensure the result is unique!
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(kld[:,0]>0):
            continue
        if f_ep.size % 4 != 0:
            print "error at " + fn
            break
        kld_dict[fn[4:6]] = kld[f_ep==1,1]


#%%
source_data = { "n_ep" : sum(n_ep_list),"n_ft" : 16*10, "amplitudes" : np.vstack([source_d_data[m][...,i].flatten() for m in sorted(source_d_data.keys()) for i in xrange(source_d_data[m].shape[-1])]),"X" : np.concatenate([kld_dict[k] for k in sorted(kld_dict.keys())])}

#now standardize them
source_data["amplitudes"] = (source_data["amplitudes"] - np.mean(source_data["amplitudes"],axis=0,keepdims=True))/np.std(source_data["amplitudes"],axis=0,keepdims=True) 
#now add intercept term
#source_data["amplitudes"] = np.hstack([np.ones((source_data["amplitudes"].shape[0],1)),source_data["amplitudes"]])
#source_data["n_ft"] += 1
#standardizing KLDs
source_data["X"] = (source_data["X"] - np.mean(source_data["X"]))/np.std(source_data["X"])
fit = pystan.stan(model_code=eeg_model_kld_reversed_ft_std , data=source_data,iter=500, chains=5)

#this is for testing the model per sklearn procedures
#getting the data in the form trials x features
skl_data = np.vstack([ np.swapaxes(source_dict[k],1,2)[...,i].flatten() for k in sorted(source_dict.keys()) for i in xrange(source_dict[k].shape[1])])
klds = np.concatenate([kld_dict[k] for k in sorted(kld_dict.keys())])


#%%
#frequency representation
#compute fft and stack positive frequencies as features
allfreqdata = [ np.reshape(fft(mat_dict[a][:,50:100,:],axis=1)[:,:25,:],(-1,mat_dict[a].shape[-1]),order='F') for a in sorted(mat_dict.keys())]
allfreqdata = np.hstack(allfreqdata)
allfreqdata = np.swapaxes(allfreqdata,0,1)

#%%
#sklearn
#which machine learning technique would make sense here?
#first try kernel pca
alldata = [ np.reshape(mat_dict[a],(-1,mat_dict[a].shape[-1]),order='F') for a in sorted(mat_dict.keys()) ]
alldata = np.hstack(alldata)
alldata = np.swapaxes(alldata,0,1)

brar = np.concatenate([brar_dict[a] for a in sorted(brar_dict.keys())]) -1

#log-odds
Y = np.concatenate([bc_dict[a] for a in sorted(bc_dict.keys())])

#klds
Y = np.concatenate([kld_dict[a] for a in sorted(kld_dict.keys())])


Yscaler = StandardScaler()

newY = Yscaler.fit_transform(Y)

scaler = StandardScaler()

#times
alldata = scaler.fit_transform(alldata)


#or for frequencies
allfreqdata = scaler.fit_transform(allfreqdata)

#alldata w/ ball rare/non-rare information
alldata = np.hstack((alldata,brar[:,None]))


#times
X_train,X_test,Y_train,Y_test = train_test_split(alldata,newY,test_size=0.3)

#frequencies
X_train,X_test,Y_train,Y_test = train_test_split(allfreqdata,newY,test_size=0.3)

svr = SVR(cache_size=1500)
svr_params = { 'C' : [1e-2,1,1e2] , 'epsilon' : [1e-3,1e-2,1e-1]  }

#fit without transforms 0.009
#fit with kld 0.017

#test with newy hier. interc.
#takes looong

enet_cv = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],max_iter=2000)
enet_cv.fit(X_tr_new,Y_train)

rcv = RidgeCV(alphas=[1e-2,1e-1,1,10])
#rcv.fit(X_train,Y_train)

svr_gs = GridSearchCV(svr,svr_params,verbose=1,n_jobs=-1)
#svr_gs.fit(X_train,Y_train)


#%%
#visualization of posterior ERPs averaged over Pbs and epochs
#for chan Fz
posteriors = np.unique(np.round(bc_dict["01"],decimals=2))

avr_ERP_p_post_list = [get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],chan=4) for k in sorted(mat_dict.keys())] 

ERP_p_post = np.array([np.mean([pb[n,:] for pb in avr_ERP_p_post_list if pb.shape[0] > n ],axis=0) for n,posterior in enumerate(posteriors) ])

#%%
#plot it
coolwarm = plt.get_cmap('coolwarm')
nm = colors.Normalize(vmin=-5,vmax=5)
scm_cm = plt.cm.ScalarMappable(norm=nm,cmap=coolwarm)

plt.figure()
plt.hold(True)
for i,post in enumerate(posteriors):
    cval = (post-np.min(posteriors))/(np.max(posteriors)-np.min(posteriors))
    plot(np.linspace(0,1000,ERP_p_post.shape[1]),ERP_p_post[i,:],c=scm_cm.to_rgba(post))
plt.xlabel('Time in ms')
plt.ylabel('Amplitude in micro Volt')
plt.title('Average ERPs for different posteriors in log-odds form for channel ' + chan_list[j])
plt.legend(map(lambda x : "lo: " + str(x) + " p: " + str(np.round(logist(x),decimals=3))[1:],posteriors))
savefig('posterior_ERPs_'+chan_list[j]+'.png')
plt.close()
#%%
#creating a lot of plots for average ERPs per post
#for channels

#now from -100 ms and with N per posterior
interesting_ones = ['Fz','FCz','Cz','Pz','Oz']
posteriors = np.unique(np.round(bc_dict["01"],decimals=2))
posterior_N = [ np.sum(np.sum(np.round(bc_dict[k],decimals=2) == p) for k in bc_dict.keys()) for p in posteriors  ]


for channel in interesting_ones:
    avr_ERP_p_post_list = [get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],chan=np.where(chan_list==channel)[0]) for k in sorted(mat_dict.keys())] 
    ERP_p_post = np.array([np.mean([pb[n,:] for pb in avr_ERP_p_post_list if pb.shape[0] > n ],axis=0) for n,posterior in enumerate(posteriors) ])
    
    
    #plot it
    coolwarm = plt.get_cmap('coolwarm')
    nm = colors.Normalize(vmin=-5,vmax=5)
    scm_cm = plt.cm.ScalarMappable(norm=nm,cmap=coolwarm)
    
    fig = plt.figure()
    fig.set_size_inches(12,10)
    plt.hold(True)
    for i,post in enumerate(posteriors):
        #cval = (post-np.min(posteriors))/(np.max(posteriors)-np.min(posteriors))
        plot(np.linspace(-100,1000,ERP_p_post.shape[1]),ERP_p_post[i,:],c=scm_cm.to_rgba(post))
    plt.xlim(-100,1000)
    plt.xlabel('Time in ms')
    plt.ylabel('Amplitude in micro Volt')
    plt.title('Average ERPs for different posteriors in log-odds form for channel ' + channel)
    plt.legend(map(lambda x : "lo: " + str(x[0]) + " p: " + str(np.round(logist(x[0]),decimals=3))[1:] + " N="+str(x[1]),zip(posteriors,posterior_N)))
    savefig('new_posterior_ERPs_'+channel+'.pdf')
    plt.close()
    
#%%
#creating plots for kld
#expects kld_dict
#IS KLD WRONG? LOOKS LIKE IT

interesting_ones = ['Fz','FCz','Cz','Pz','Oz']
klds = np.unique(np.round(kld_dict["01"],decimals=3))
kld_dict = { k : np.round(kld_dict[k],decimals=3) for k in sorted(kld_dict.keys())} 
kld_N = [ np.sum(np.sum(np.round(kld_dict[k],decimals=3) == p) for k in kld_dict.keys()) for p in klds  ]


for channel in interesting_ones:
    avr_ERP_p_kld_list = [ np.array([ np.mean(mat_dict[k][np.where(chan_list==channel)[0],:,kld_dict[k]==i],axis=0) for i in np.unique(kld_dict[k]) ]) for k in sorted(mat_dict.keys())] 
    ERP_p_kld = np.array([np.mean([pb[n,:] for pb in avr_ERP_p_kld_list if pb.shape[0] > n ],axis=0) for n,kld in enumerate(klds) ])
    
    
    #plot it
    coolwarm = plt.get_cmap('coolwarm')
    nm = colors.Normalize(vmin=0,vmax=0.1)
    scm_cm = plt.cm.ScalarMappable(norm=nm,cmap=coolwarm)
    
    fig = plt.figure()
    fig.set_size_inches(12,10)
    plt.hold(True)
    for i,kld in enumerate(klds):
        plot(np.linspace(-100,1000,ERP_p_kld.shape[1]),ERP_p_kld[i,:],c=scm_cm.to_rgba(kld))
    plt.xlim(-100,1000)
    plt.xlabel('Time in ms')
    plt.ylabel('Amplitude in micro Volt')
    plt.title('Average ERPs for different posteriors in log-odds form for channel ' + channel)
    plt.legend(map(lambda x : "kld: " + str(x[0]) + " N=" + str(x[1]),zip(klds,kld_N)))
    savefig('new_kld_ERPs_'+channel+'.pdf')
    plt.close()
    
    
#%%
#plots for distorted probabilities

for k in bc_dict.keys():
    bc_dict[k] = np.round(bc_dict[k],decimals=2)

for channel in interesting_ones:
    avr_ERP_p_post_list = [get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],chan=np.where(chan_list==channel)[0]) for k in sorted(mat_dict.keys())] 
    #ERP_p_post = np.array([np.mean([pb[n,:] for pb in avr_ERP_p_post_list if pb.shape[0] > n ],axis=0) for n,posterior in enumerate(posteriors) ])
    
    
    #plot it
    coolwarm = plt.get_cmap('coolwarm')
    nm = colors.Normalize(vmin=-15,vmax=10)
    scm_cm = plt.cm.ScalarMappable(norm=nm,cmap=coolwarm)
    
    fig = plt.figure()
    fig.set_size_inches(12,10)
    plt.hold(True)
    for j,ERP_p_post in enumerate(avr_ERP_p_post_list):
        for i,post in enumerate(np.unique(bc_dict[list(sorted(bc_dict.keys()))[j]])):
            #cval = (post-np.min(posteriors))/(np.max(posteriors)-np.min(posteriors))
            plot(np.linspace(-100,1000,ERP_p_post.shape[1]),ERP_p_post[i,:],c=scm_cm.to_rgba(post))
    plt.xlim(-100,1000)
    plt.xlabel('Time in ms')
    plt.ylabel('Amplitude in micro Volt')
    plt.title('Average ERPs for different posteriors in log-odds form for channel ' + channel)
    savefig('individual_posterior_ERPs_'+channel+'.pdf')
    plt.close()
    

#%%
#without gammas applied
#but distorts/averages
#now from -100 ms and with N per posterior
interesting_ones = ['Fz','FCz','Cz','Pz','Oz']
posteriors = np.unique(np.round(bc_dict["01"],decimals=2))
#posterior_N = [ np.sum(np.sum(np.round(bc_dict[k],decimals=2) == np.unique(np.round(bc_dict[k],decimals=2))[p]) for k in bc_dict.keys() if p < len(np.unique(np.round(bc_dict[k],decimals=2)))) for p in xrange(len(posteriors))  ]
posterior_N = [ np.sum(np.sum(np.round(bc_dict[k],decimals=2) == p) for k in bc_dict.keys()) for p in posteriors  ]

#weigh by distorted probability of pb
avr_disto_posteriors = [ np.round(np.sum(np.sum(np.round(bc_dict[k],decimals=2) == p)*distort_logodds(p,gammas_exp80[i],pnull_exp80[i]) for i,k in enumerate(sorted(mat_dict.keys())))/p_N,decimals=2) for p,p_N in zip(posteriors,posterior_N)] 

for channel in interesting_ones:
    avr_ERP_p_post_list = [get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],chan=np.where(chan_list==channel)[0]) for k in sorted(mat_dict.keys())] 
    ERP_p_post = np.array([np.mean([pb[n,:] for pb in avr_ERP_p_post_list if pb.shape[0] > n ],axis=0) for n,posterior in enumerate(posteriors) ])
    
    
    #plot it
    coolwarm = plt.get_cmap('coolwarm')
    nm = colors.Normalize(vmin=-12,vmax=12)
    scm_cm = plt.cm.ScalarMappable(norm=nm,cmap=coolwarm)
    
    fig = plt.figure()
    fig.set_size_inches(12,10)
    plt.hold(True)
    for i,post in enumerate(avr_disto_posteriors):
        #cval = (post-np.min(posteriors))/(np.max(posteriors)-np.min(posteriors))
        plot(np.linspace(-100,1000,ERP_p_post.shape[1]),ERP_p_post[i,:],c=scm_cm.to_rgba(post))
    plt.xlim(-100,1000)
    plt.xlabel('Time in ms')
    plt.ylabel('Amplitude in micro Volt')
    plt.title('Average ERPs for different posteriors in log-odds form for channel ' + channel)
    plt.legend(map(lambda x : "lo: " + str(x[0]) + " p: " + str(np.round(logist(x[0]),decimals=3))[1:] + " N="+str(x[1]),zip(avr_disto_posteriors,posterior_N)))
    savefig('weighted_posterior_ERPs_'+channel+'.pdf')
    plt.close()
    
#%%
#create X,y representation of data using grand average
interesting_ones = ['Fz','FCz','Cz','Pz','Oz']
#list of participants, each entry the concatenated timecourses from 0 to 800 ms (4 ms sampling, no time binning)
avr_ERP_p_post_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],chan=np.where(chan_list==channel)[0])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]

#%%
#get rid of n < 4 values
avr_ERP_p_post_list = [ERPs[[nr for nr,p in enumerate(np.unique(np.round(bc_dict[sorted(bc_dict.keys())[nerp]],decimals=2))) if np.sum(np.round(bc_dict[sorted(bc_dict.keys())[nerp]],decimals=2) == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_post_list)]

#create corresponding y
y = np.concatenate([[p for p in np.unique(np.round(bc_dict[k],decimals=2)) if np.sum(np.round(bc_dict[k],decimals=2) == p) >= 4  ] for k in sorted(bc_dict.keys())])


#%%
#now concatenate on axis of posteriors
X = np.concatenate(avr_ERP_p_post_list,axis=0)

#create corresponding y
y = np.concatenate([np.unique(np.round(bc_dict[k],decimals=2))  for k in sorted(bc_dict.keys())])


scaler = StandardScaler()

#times
X = scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1)

#enet_cv = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],max_iter=2000)

#ridge = Ridge(alpha=1.0).fit(X_train,Y_train)

#%%
ralpha = 0.000001

coefs = np.reshape([ Ridge(alpha=ralpha).fit(Y_train[:,None],xnow).coef_ for xnow in X_train.T],(5,200))


#%%
rcv = RidgeCV(alphas=[1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2])
rcv.fit(X_train,Y_train)
coefs = np.reshape(rcv.coef_,(5,200))

#%%
#visualize

plt.imshow(coefs,aspect='auto',interpolation='nearest')
xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
yticks([0,1,2,3,4],interesting_ones)
plt.colorbar()

#%%
#FOR KLDs or DISTORTED KLDs
#create X,y representation of data using grand average
interesting_ones = ['Fz','FCz','Cz','Pz','Oz']
#list of participants, each entry the concatenated timecourses from 0 to 800 ms (4 ms sampling, no time binning)
avr_ERP_p_kld_list = [ np.concatenate([np.array([ np.mean(mat_dict[k][np.where(chan_list==channel)[0],:,kld_dict[k]==i],axis=0) for i in np.unique(kld_dict[k]) ])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())] 

#avr_ERP_p_kld_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],kld_dict[k],chan=np.where(chan_list==channel)[0])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]
#%%
#get rid of n < 4 values
avr_ERP_p_kld_list = [ERPs[[nr for nr,p in enumerate(np.unique(kld_dict[sorted(kld_dict.keys())[nerp]])) if np.sum(kld_dict[sorted(kld_dict.keys())[nerp]] == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_kld_list)]



#%%
#now concatenate on axis of posteriors
X = np.concatenate(avr_ERP_p_kld_list,axis=0)

#create corresponding y
y = np.concatenate([[p for p in np.unique(kld_dict[k]) if np.sum(kld_dict[k]== p) >= 4] for k in sorted(kld_dict.keys())])


scaler = StandardScaler()

#times
X = scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1)
