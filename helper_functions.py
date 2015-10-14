# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:54:09 2015

@author: mboos
"""

#helper functions for EEG-analysis



#import re
import matplotlib.colors as colors
from functools import partial
import os

import pandas as pd
from math import fsum

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#change this model

path = "/home/mboos/Work/Bayesian Updating/"

#lppd for old model
#def lppd(fit_obj,X,Y):
#    """Computes Log Posterior predictive density using a stan fit_obj and a data object.
#    Uses a simple linear model for likelihood"""
#    #X and Y in same form as in stan model
#    #p_vec = ...n
#    #sigma = ...
#    lppd = 0
#    #now get the ideal values for each pb,epoch,chan,bin
#    for pb in xrange(n_pb):
#        for ep in xrange(n_ep):
#            for t in xrange(n_bin):
#                for c in xrange(chan_bin):
#                    lppd += np.mean(norm.pdf(Y[pb,ep,t,c],loc=X[pb,ep]*p_vec[t,c],scale=sigma))
#    
#    return lppd

#TODO: visualize the likelihoods and data, either a lot of likelihoods vs data or just a couple
#TODO: kick out outliers
def plot_IC_KLD(fit_obj,sources,klds):
    """Expects a stan fit object, the matrix of sources as given to the KLD model and the Kullback-Leibler-Divergences. Creates plots for the 6 components with the highest prediction"""
    theta_means = np.mean(fit_obj.get_posterior_mean()[:160,:],axis=1)
    #theta_w = np.argsort([ np.sum(source_times_weights) for source_times_weights in np.split(np.abs(theta_means),10)])[::-1]
    theta_w = np.argsort(np.abs(theta_means))[::-1]
    for i,idx in enumerate(theta_w[:6]):
        plt.subplot(2,3,i+1)
        plt.plot(klds,sources[:,idx],"o")
        plt.title("source: "+str(idx/16)+" timepoint:"+str(idx%16*25+100))


def get_residuals_for_model(fit_obj,Y,X):
    """Returns a n x 1 vector of residuals for the reverse_ft_std model with mean of parameter values from fit_obj, and data from Y and X
    :param: fit_obj: a stan-fit object from the reverse_ft_std model
    :param: Y: vector of length n of the posteriors or KLDs to predict
    :param: X: matrix of dimension n x p of the channels or ICs to predict with
    :return: vector of length n"""

    theta_means = np.mean(fit_obj.get_posterior_mean()[:X.shape[1],:],axis=1)
    predictions = np.dot(X,theta_means)
    return Y-predictions


def distort_logodds(lo,gamma,pnull):
    return np.array(lo)*gamma + pnull*(1-gamma)
   
def logodds(p):
    return np.log(p/(1-p))




def plot_for_components(mean_source_per_post,clist,nbin=20):
    kbin_source_per_post = { key : k_bin_average(mean_source_per_post[key],nbin) for key in mean_source_per_post.keys() }
    if len(clist) < 4:
        f,splots = plt.subplots(len(clist),1,sharex=True,sharey=True)
    else:
        rows = ([i for i in [2,3,4] if len(clist) % i == 0]+[2 if len(clist) < 8 else 3])[0]
        cols = int(np.ceil(float(len(clist))/rows))
        f,splots = plt.subplots(rows,cols)
    f.tight_layout()
    for i,ax in enumerate(splots.flatten()[:len(clist)]):
        for j,k in enumerate(sorted(kbin_source_per_post.keys())):
            ax.plot(kbin_source_per_post[k][clist[i],:],['r','g','b','c','m','y','k','0.25','0.75'][j],label=k)
    f.legend(*splots.flatten()[0].get_legend_handles_labels(),loc=4)
    
def predictive_surprise_vec(filename,likelihood,path = "/home/mboos/Work/Bayesian Updating/Data/"):
    """ fill in """
    with open(path+filename) as bc_file:
        #second array expression maps the event rare/freq 2/1 to its log2 likelihood and sums them up
        return np.array([ (int(line.strip("\n").split(" ")[-1]),-np.log2(likelihood[map(abs,map(int,line.strip("\n").split(" ")))[-1]])) for line in bc_file])


logist = lambda x : 1/(1+np.exp(-x))

kld_helper = lambda x : (logist(fsum(x[:-1])),logist(fsum(x)))


def channel_weights(w,chan_list = ["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FCz","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","TP9","TP10","Oz","O2","O1"]):
    for s in map(": ".join,zip(chan_list,np.array_str(w).strip("[]").split())):
        print s
	

def discrete_kld(dist1,dist2):
    """ returns the kullback leibler divergence of the two distributions over the second- one"""
    eps = 1e-20
    return np.log(eps+dist2/(eps+dist1))*dist2 + np.log(eps+(1-dist2)/(1-dist1+eps))*(1-dist2)
    

def kld_vec(filename,prior,likelihood,path = "/home/mboos/Work/Bayesian Updating/Data/"):
    """ fill in """
    with open(path+filename) as bc_file:
        #second array expression maps the event rare/freq 2/1 to its log likelihood ratio and sums them up with the log prior ratio
        return np.array([ (int(line.strip("\n").split(" ")[-1]),discrete_kld(*kld_helper([np.log(prior/(1-prior))]+map(lambda x : np.log(likelihood[x]/(1-likelihood[x])),map(abs,map(int,line.strip("\n").split(" "))))))) for line in bc_file])

def kld_vec_distort(filename,prior,likelihood,gamma,pnull,path = "/home/mboos/Work/Bayesian Updating/Data/"):
    """ fill in """
    with open(path+filename) as bc_file:
        #second array expression maps the event rare/freq 2/1 to its log likelihood ratio and sums them up with the log prior ratio
        return np.array([ (int(line.strip("\n").split(" ")[-1]),discrete_kld(*kld_helper(distort_logodds([np.log(prior/(1-prior))]+map(lambda x : np.log(likelihood[x]/(1-likelihood[x])),map(abs,map(int,line.strip("\n").split(" ")))),gamma,pnull)))) for line in bc_file])


def prob_dist(p,gamma,pnull):
    return logist(distort_logodds(logodds(p),gamma,pnull))


def interval_feature_extraction(samples,time=1):
    """Returns interval features for all the samples, mean amplitude, std and covariance with timevariable
    as a 3 x number of intervals array"""
    
    intervals = array([ array([np.mean(samples[:2**i]),np.std(samples[:2**i]),(1/(2**i))*np.dot(samples[:2**i],np.arange(time,time+2**i).T)-np.mean(samples[:2**i])*((time+2**i-1)/2)]) for i in xrange(1,int(np.trunc(np.log2(len(samples))))+1) ]).T
    
    if len(samples) == 2:
        return intervals
    else:
        return np.append(intervals,interval_feature_extraction(samples[1:],time+1),axis=1)

def vis_pred(fit_obj,X,Y,n_trials):
    """Plots Predictions on y Axis against actual Posteriors on X Axis"""
    theta,sigma = np.split(np.mean(fit_obj.get_posterior_mean()[:-1,:],axis=1),[57])
    figure()    
    plot(Y,np.dot(X,theta),"bo")
    ca = gca()
    ca.set_autoscale_on(False)
    ca.plot(ca.get_xlim(),ca.get_xlim(),"r")

def vis_res(fit_obj,X,Y,n_trials):
    """Plots (1) Residuals on Y against data on X, (2) Residuals on Y against normal with mean 0 and variance sigma"""
    theta,sigma = np.split(np.mean(fit_obj.get_posterior_mean()[:-1,:],axis=1),[57])
    residuals = (Y-np.dot(X,theta))
    figure(1)
    subplot(211)
    plot(Y,residuals,"bo")
    subplot(212)
    scipy.stats.probplot(residuals,sparams=(0,sigma))

def lppd(fit_obj,X,Y,n_trials):
    """Computes the log posterior predictive density using a stan fit object and data Y (the log-odds) and Y (a matrix trials X features)"""
    pars = fit_obj.extract(pars=["theta","sigma_std"],permuted=True)
    theta = pars["theta"]
    sigma_std = pars["sigma_std"]
    lppd = 0
    for i in xrange(n_trials):
        lppd += np.log(np.mean(scipy.stats.norm.pdf(Y[i],loc=np.dot(theta,X[i].T),scale=sigma_std)))
    return lppd

def lppd_spec(fit_obj,X,Y,n_trials):
    """Computes the log posterior predictive density using a stan fit object and data Y (the log-odds) and Y (a matrix trials X features)"""
    pars = fit_obj.extract(pars=["theta","sigma_std"],permuted=True)
    theta = pars["theta"]
    sigma_std = pars["sigma_std"]
    lppd = 0
    for i in xrange(int(np.round(n_trials/5,0))):
        ll = np.log(np.mean(scipy.stats.norm.pdf(Y[i],loc=np.dot(theta,X[i].T),scale=sigma_std)))
        print "Posterior: {0} LL: {1}".format(logist(Y[i]),ll)
        lppd += ll
    return lppd

def waic(fit_obj,X,Y,n_trials):
    """Computes the log posterior predictive density using a stan fit object and data Y (the log-odds) and Y (a matrix trials X features)"""
    pars = fit_obj.extract(pars=["theta","sigma_std"],permuted=True)
    theta = pars["theta"]
    sigma_std = pars["sigma_std"]
    pd = 0
    for i in xrange(n_trials):
        pd += np.var(np.log(scipy.stats.norm.pdf(Y[i],loc=np.dot(theta,X[i].T),scale=sigma_std)))
    return lppd(fit_obj,X,Y,n_trials) - pd


def get_bclass(filename,prior,likelihood,path = "/home/mboos/Work/Bayesian Updating/Data/"):
    """Returns N x 2 array the first column consisting of 1/2 the second of the posterior_probability
    likelihood needs to be a dict with likelihood values for 1/2"""
    with open(path+filename) as bc_file:
        #second array expression maps the event rare/freq 2/1 to its log likelihood ratio and sums them up with the log prior ratio
        return np.array([ (int(line.strip("\n").split(" ")[-1]),np.log(prior/(1-prior))+fsum(map(lambda x : np.log(likelihood[x]/(1-likelihood[x])),map(abs,map(int,line.strip("\n").split(" ")))))) for line in bc_file])

def get_bclass_list(filename,prior,likelihood,path = "/home/mboos/Work/Bayesian Updating/Data/"):
    """Returns N x 2 array the first column consisting of 1/2 the second of the posterior_probability
    likelihood needs to be a dict with likelihood values for 1/2"""
    with open(path+filename) as bc_file:
        #second array expression maps the event rare/freq 2/1 to its log likelihood ratio and sums them up with the log prior ratio
        return [ [np.log(prior/(1-prior))]+map(lambda x : np.log(likelihood[x]/(1-likelihood[x])),map(abs,map(int,line.strip("\n").split(" ")))) for line in bc_file]

def get_bclass_diff(filename,likelihood,path = "/home/mboos/Work/Bayesian Updating/Data/"):
    """Returns N x 2 array, first column consisting of 1/2 the second of the log-odds likelihood of that ball"""
    lo = lambda x : np.log(likelihood[x]/(1-likelihood[x]))
    with open(path+filename) as bc_file:
        return np.array([ (int(line.strip("\n").split(" ")[-1]),lo(map(abs,map(int,line.strip("\n").split(" ")))[-1])) for line in bc_file])

    
    
def get_failed_epochs(filename,path = "/home/mboos/Work/Bayesian Updating/Data/"):
    """Returns an array with 0 everywhere where an epoch failed in the sequence"""
    with open(path+filename) as ep_file:
        return np.array([ 0 if "-" in line else 1 for line in ep_file])   
        
def rereference(dmat):
    """returns ndarray of the data re-referenced with the average over all electrodes, argument dmat in form chan x time x epoch"""
    return dmat - np.mean(dmat,axis=0)
    
def k_bin_average(dmat,k):
    """returns ndarray with k timebins (average of timevalues), argument dmat in form chan x time x epoch"""
    return np.array([ np.mean(ar,axis=1) for ar in np.array_split(dmat,k,axis=1)]).swapaxes(0,1)
    
def plot_ERPs_per_posterior(dmat,ep_post_u,chan=4):
    """plots each ERP for each unique posterior"""
    ep_posterior = np.round(ep_post_u,decimals=2) 
    for sp,n in enumerate(np.unique(ep_posterior)):
        for i in np.where(ep_posterior==n)[0]:
            subplot(3,3,sp+1)
            plot(np.linspace(0,1200,dmat.shape[1]),dmat[chan,:,i],"b")
            title("Posterior = "+str(logist(n)))

def get_average_ERPs_per_posterior(dmat,ep_post_u,chan=4):
    """Returns an array M x N with posteriors M and time points N"""
    ep_posterior = np.round(ep_post_u,decimals=2)
    return np.array([ np.mean(dmat[chan,:,ep_posterior==n],axis=0) for n in np.unique(ep_posterior)])

def get_average_ERPs_per_pooled_posteriors(dmat,ep_post_u,cut_off_list,chan=4):
    """Computes the ERP for clustering/pooling of posteriors whose cut-off points are specified in cut_off_list
    IN: ....
    cut_off_list    -       each item in cut off list is the transition point between posterior classes, first and last item are transition points from first to second, respective from second-to-last to last"""
    smaller_than = [dmat[chan,:,:] < cut_off for cut_off in cut_off_list]
    def iadd(x,y):
        x+y
    
    return np.array([ np.mean(dmat[chan,:,np.logical_and(ep_post_u>=cut_off_list[n-1],ep_post_u < cut_off_list[n]) ],axis=0) for n in xrange(1,len(cut_off_list))])    

     
