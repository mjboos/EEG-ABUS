
# coding: utf-8

# # EEG Data Analysis for Bayesian Updating

# First import the data for only one condition (explicit with a prior probability of .8)

# In[2]:

from helper_functions import *
import sys
import pystan
import mne
#import re
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
from sklearn.linear_model import ElasticNetCV,RidgeCV,Ridge,BayesianRidge
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVR,NuSVR
from scipy.fftpack import fft



def get_coefficients(encoding,X,y,normalize=True,actfilt=True):
    '''Returns coefficients:
    Paramters:
    encoding    -   Encoding (True) or Decoding (False)
    X           -   EEG Timeseries
    y           -   Probability representation
    '''
    if encoding:
        coefs = np.reshape([BayesianRidge(normalize=normalize).fit(y[:,None],xnow).coef_ for xnow in X.T],(5,200))
    else:
        if actfilt:
            yscaler = StandardScaler()
            xscaler = StandardScaler()
            Xcov = np.cov(X,rowvar=0)
            y = yscaler.fit_transform(y)
            X = xscaler.fit_transform(X)
            coefs = BayesianRidge(normalize=False).fit(X,y).coef_
            coefs = np.reshape(Xcov.dot(coefs),(5,200))
        else:
            coefs = np.reshape(BayesianRidge(normalize=normalize).fit(X,y).coef_,(5,200))
    return coefs



chan_list = np.array(["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2",\
        "FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz",\
        "P4","P8","TP9","TP10","Oz","O2","O1"])
interesting_ones = ['Fz','FC2','Cz','Pz','Oz']

encoding = {'True':True,'False':False}[sys.argv[1]]

#can be normal, kolossa, kolossa_p0, kolossa_hier
model_type = sys.argv[2]

if len(sys.argv) > 3:
    output_dir = sys.argv[3] + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    output_dir = './'

#These can and should later be estimated in an overall model
gammas_exp80 = np.array([2.9,2.5,2.0,2.2,2.5,1.7,3.8,3.1,3.8,2.6,3.0,3.9,1.9,3.9,1.7,1.6]).T
#pnull in log-odds
pnull_exp80 = np.array([-1.5,-1.6,-1.9,0.2,-0.8,-1.8,-1.0,0.8,-1.6,-2.4,-2.5,-1.0,-2.4,-1.0,-0.2,-1.5]).T
#now you can transform the bc_dict values for every person



# In[4]:

#alternative for kolossa model
#These can and should later be estimated in an overall model
gammas_exp80_kolossa = np.array([0.74 for t in xrange(16)]).T
#pnull in log-odds
pnull_exp80_kolossa = np.array([0 for t in xrange(16)]).T
#now you can transform the bc_dict values for every person


# In[3]:

#alternative for kolossa model
gammas_exp80_kolossa_hier = np.array([0.75,0.74,0.72,0.76,0.74,0.74,0.74,0.81,0.77,0.75,0.71,0.73,0.73,0.72,0.71,0.74])
#pnull in log-odds
pnull_exp80_kolossa_hier = np.array([0 for t in xrange(16)]).T
#now you can transform the bc_dict values for every person


# In[155]:

#alternative for kolossa model
#These can and should later be estimated in an overall model
gammas_exp80_kolossa_p0 = np.array([0.88 for t in xrange(16)]).T
#pnull in log-odds
pnull_exp80_kolossa_p0 = np.array([4.0 for t in xrange(16)]).T
#now you can transform the bc_dict values for every person

if model_type == 'kolossa_p0':
    gammas_exp80_kolossa = gammas_exp80_kolossa_p0
    pnull_exp80_kolossa = pnull_exp80_kolossa_p0
elif model_type == 'kolossa_hier':
    gammas_exp80_kolossa = gammas_exp80_kolossa_hier
    pnull_exp80_kolossa = pnull_exp80_kolossa_hier

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

#100ms BEFORE stimulus presentation


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
        brar_dict[fn[4:6]] = bclass[f_ep==1,0]

        


# ### For last two balls

# In[2]:
# ## Different probability representations

# In[5]:

#Kullback-Leibler divergence
nbin = 1#doesnt matter, curr isnt used, only to ensure that we have an estimate of the number of epochs 
kld_dict = dict()
f_ep_dict = dict()

for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep_dict[fn[4:6]] = get_failed_epochs(fn)
        f_ep = f_ep_dict[fn[4:6]]
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

#Distorted log-odds
dbc_dict = {}
for k in bc_dict.keys():
    dbc_dict[k] = bc_dict[k]*gammas_exp80[int(k)-1] + pnull_exp80[int(k)-1]*(1-gammas_exp80[int(k)-1])

        
#Predictive Surprise
nbin = 1 #doesnt matter, curr isnt used, only to ensure that we have an estimate of the number of epochs 
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



#distorted Kullback-Leibler divergence
nbin = 1 #doesnt matter, curr isnt used, only to ensure that we have an estimate of the number of epochs 
kld_dist_dict = dict()

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
        kld_dist_dict[fn[4:6]] = kld[f_ep==1,1]

#distorted log-odds as in Kolossa et al.
nbin = 1
kdist_dict = dict()
blist_dict = dict()

#first get this as list representation
for fn in files:
    if fn.startswith("TS") and pattern_TS in fn:
        f_ep = get_failed_epochs(fn)
        bclass = get_bclass_list(fn,prior,likelihood)
        bcl1 = get_bclass(fn,prior,likelihood)
        curr = k_bin_average(rereference(loadmat(path_mat+filter(lambda x : fn[4:6] in x and pattern in x and "epochs" in x,mat_files)[0])["EEGdata"][:30,50:,:]),nbin)
        if curr.shape[2] != sum(bcl1[:,0]>0):
            continue
        brar_dict[fn[4:6]] = bcl1[:,0]
        blist_dict[fn[4:6]] = bclass

 
for pb in sorted(blist_dict.keys()):
    for i,e in enumerate(blist_dict[pb]):
        blist_dict[pb][i][0] = e[0] + e[1]
        blist_dict[pb][i].pop(1)
        blist_dict[pb][i] = [0 for t in xrange(4-len(blist_dict[pb][i]))] + blist_dict[pb][i]

for key in sorted(blist_dict.keys()):
    kdist_dict[key] = np.array(blist_dict[key])
    kdist_dict[key] = kdist_dict[key].dot(np.array([gammas_exp80_kolossa[int(key)-1]**i for i in reversed(xrange(1,5))]))


kld_kdist_dict = dict()
for key in sorted(kdist_dict.keys()):
    kld_kdist_dict[key] = np.concatenate([np.array([discrete_kld(*logist(row)) for row in rolling_window(np.append(gammas_exp80_kolossa[int(key)-1]*np.log(prior/(1-prior)),e),2)]) for e in np.reshape(kdist_dict[key],(50,4))])
    
for key in kdist_dict.keys():
    kdist_dict[key] = kdist_dict[key][f_ep_dict[key]==1]
    kld_kdist_dict[key] = kld_kdist_dict[key][f_ep_dict[key]==1]



interesting_ones = ['Fz','FC2','Cz','Pz','Oz']


if model_type == 'normal':
    avr_ERP_p_post_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],bc_dict[k],chan=np.where(chan_list==channel)[0])[:,25:-50] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]

    #%%
    #get rid of n < 4 values
    avr_ERP_p_post_list = [ERPs[[nr for nr,p in enumerate(np.unique(np.round(bc_dict[sorted(bc_dict.keys())[nerp]],decimals=2))) if np.sum(np.round(bc_dict[sorted(bc_dict.keys())[nerp]],decimals=2) == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_post_list)]

    #create corresponding y
    y = np.concatenate([[p for p in np.unique(np.round(bc_dict[k],decimals=2)) if np.sum(np.round(bc_dict[k],decimals=2) == p) >= 4  ] for k in sorted(bc_dict.keys())])
    #now concatenate on axis of posteriors
    X = np.concatenate(avr_ERP_p_post_list,axis=0)

    y_st = np.concatenate([bc_dict[key] for key in sorted(bc_dict.keys())])
    X_st = np.vstack([ np.reshape(np.swapaxes(np.swapaxes(mat_dict[key][:,25:-50,:],0,2),1,2)[:,[np.where(chan_list==channel)[0][0] for channel in interesting_ones],:],(-1,1000)) for key in sorted(mat_dict.keys()) ])

    ralpha = 0.000001


    # ### Correlations/Regression for ERPs with undistorted (log-odds) probabilites

    # In[138]:

    normalize = True

    coefs = get_coefficients(encoding,X,y)
    
    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('{} coefficients w/ undistorted probabilities  '.format({True:'Encoding',False:'Decoding'}[encoding]))

    plt.savefig(output_dir+'avr_ERP_unverzerrte_lo_probs.pdf')
    plt.clf() 


    # ### Single-Trial Regressions/Correlations with undistorted (log-odds) probabilities

    # In[6]:

    normalize = True
    save = False

    coefs = get_coefficients(encoding,X_st,y_st)
    

    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('Single Trial {} coefficients w/ undistorted probabilities  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'singletrial_ERP_unverzerrte_lo_probs.pdf')
    plt.clf()

    # In[16]:

    # In[6]:

    normalize = True

    coefs = get_coefficients(encoding,X_st,y_st)
    
    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('Single Trial {} coefficients w/ undistorted probabilities  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'avr_ERP_unverzerrte_lo_probs.pdf')

    plt.clf()

    # ### Correlations with distorted (log-odds) probabilities

    # In[140]:

    avr_ERP_p_post_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],dbc_dict[k],chan=np.where(chan_list==channel)[0])[:,25:-50] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]

    #%%
    #get rid of n < 4 values
    avr_ERP_p_post_list = [ERPs[[nr for nr,p in enumerate(np.unique(np.round(dbc_dict[sorted(dbc_dict.keys())[nerp]],decimals=2))) if np.sum(np.round(dbc_dict[sorted(dbc_dict.keys())[nerp]],decimals=2) == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_post_list)]

    #create corresponding y
    y = np.concatenate([[p for p in np.unique(np.round(dbc_dict[k],decimals=2)) if np.sum(np.round(dbc_dict[k],decimals=2) == p) >= 4  ] for k in sorted(dbc_dict.keys())])
    #now concatenate on axis of posteriors
    X = np.concatenate(avr_ERP_p_post_list,axis=0)


    y_st = np.concatenate([dbc_dict[key] for key in sorted(dbc_dict.keys())])
    X_st = np.vstack([ np.reshape(np.swapaxes(np.swapaxes(mat_dict[key][:,25:-50,:],0,2),1,2)[:,[np.where(chan_list==channel)[0][0] for channel in interesting_ones],:],(-1,1000)) for key in sorted(mat_dict.keys()) ])


    ralpha = 0.000001


    # ### Correlations/Regressions for average ERPs with distorted probabilities

    # In[141]:

    normalize = True

    coefs = get_coefficients(encoding,X,y)
    

    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('{} coefficients w/ distorted probabilities  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'avr_ERP_verzerrte_lo_probs.pdf')


    plt.clf()
    # ### Single Trial Correlations/Regression with distorted probabilities

    # In[142]:

    normalize = True

    coefs = get_coefficients(encoding,X_st,y_st)

    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('Single-Trial {} coefficients w/ distorted probabilities  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'singletrial_verzerrte_lo_probs.pdf')


    plt.clf()

    #list of participants, each entry the concatenated timecourses from 0 to 800 ms (4 ms sampling, no time binning)
    avr_ERP_p_kld_list = [ np.concatenate([np.array([ np.mean(mat_dict[k][np.where(chan_list==channel)[0],:,kld_dict[k]==i],axis=0) for i in np.unique(kld_dict[k]) ])[:,25:-50] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())] 

    #avr_ERP_p_kld_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],kld_dict[k],chan=np.where(chan_list==channel)[0])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]
    #%%
    #get rid of n < 4 values
    avr_ERP_p_kld_list = [ERPs[[nr for nr,p in enumerate(np.unique(kld_dict[sorted(kld_dict.keys())[nerp]])) if np.sum(kld_dict[sorted(kld_dict.keys())[nerp]] == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_kld_list)]


    #%%
    #now concatenate on axis of posteriors
    X = np.concatenate(avr_ERP_p_kld_list,axis=0)

    #create corresponding y
    y = np.concatenate([[p for p in np.unique(kld_dict[k]) if np.sum(kld_dict[k]== p) >= 4] for k in sorted(kld_dict.keys())])


    y_st = np.concatenate([kld_dict[key] for key in sorted(kld_dict.keys())])
    X_st = np.vstack([ np.reshape(np.swapaxes(np.swapaxes(mat_dict[key][:,25:-50,:],0,2),1,2)[:,[np.where(chan_list==channel)[0][0] for channel in interesting_ones],:],(-1,1000)) for key in sorted(mat_dict.keys()) ])



    # ### Correlation/Regression coefficients for undistorted KLDs over average ERPs

    # In[147]:

    normalize = True

    coefs = get_coefficients(encoding,X,y)
    

    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('{} coefficients w/ undistorted KLDs  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'avr_ERP_unverzerrte_KLDs.pdf')


    plt.clf()
    # ### Single Trial Correlation/Regression coefficients for undistorted KLDs

    # In[148]:

    normalize = True


    coefs = get_coefficients(encoding,X_st,y_st)

    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('Single-Trial {} coefficients w/ undistorted KLDs  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'singletrial_unverzerrte_KLDs.pdf')


    plt.clf()
    # ### Correlation with Distorted KLDs

    # In[7]:

    #list of participants, each entry the concatenated timecourses from 0 to 800 ms (4 ms sampling, no time binning)
    avr_ERP_p_kld_list = [ np.concatenate([np.array([ np.mean(mat_dict[k][np.where(chan_list==channel)[0],:,kld_dist_dict[k]==i],axis=0) for i in np.unique(kld_dist_dict[k]) ])[:,25:-50] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())] 

    #avr_ERP_p_kld_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],kld_dict[k],chan=np.where(chan_list==channel)[0])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]
    #%%
    #get rid of n < 4 values
    avr_ERP_p_kld_list = [ERPs[[nr for nr,p in enumerate(np.unique(kld_dist_dict[sorted(kld_dist_dict.keys())[nerp]])) if np.sum(kld_dist_dict[sorted(kld_dist_dict.keys())[nerp]] == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_kld_list)]


    #%%
    #now concatenate on axis of posteriors
    X = np.concatenate(avr_ERP_p_kld_list,axis=0)

    #create corresponding y
    y = np.concatenate([[p for p in np.unique(kld_dist_dict[k]) if np.sum(kld_dist_dict[k]== p) >= 4] for k in sorted(kld_dist_dict.keys())])


    y_st = np.concatenate([kld_dist_dict[key] for key in sorted(kld_dist_dict.keys())])
    X_st = np.vstack([ np.reshape(np.swapaxes(np.swapaxes(mat_dict[key][:,25:-50,:],0,2),1,2)[:,[np.where(chan_list==channel)[0][0] for channel in interesting_ones],:],(-1,1000)) for key in sorted(mat_dict.keys()) ])



    # ### Correlation/Regression Coefficients for distorted KLDs  

    # In[150]:

    normalize = True

    coefs = get_coefficients(encoding,X,y)
    

    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('{} coefficients w/ distorted KLDs  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'avr_ERP_verzerrte_KLDs.pdf')

    plt.clf()

    # ### Single Trial Correlation/Regression Coefficients for distorted KLDs

    # In[151]:

    normalize = True

    coefs = get_coefficients(encoding,X_st,y_st)
    
    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('Single-Trial {} coefficients w/ distorted KLDs  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'singletrial_verzerrte_KLDs.pdf')


    plt.clf()

# ### Correlations with distorted Log-Odds (as in Kolossa et al.)

# In[157]:
if model_type in ['kolossa','kolossa_p0','kolossa_hier']:
    avr_ERP_p_kdist_list = [ np.concatenate([np.array([ np.mean(mat_dict[k][np.where(chan_list==channel)[0],:,kdist_dict[k]==i],axis=0) for i in np.unique(kdist_dict[k]) ])[:,25:-50] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())] 

    #avr_ERP_p_kld_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],kld_dict[k],chan=np.where(chan_list==channel)[0])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]
    #%%
    #get rid of n < 4 values
    avr_ERP_p_kdist_list = [ERPs[[nr for nr,p in enumerate(np.unique(kdist_dict[sorted(kdist_dict.keys())[nerp]])) if np.sum(kdist_dict[sorted(kdist_dict.keys())[nerp]] == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_kdist_list)]


    #%%
    #now concatenate on axis of posteriors
    X = np.concatenate(avr_ERP_p_kdist_list,axis=0)

    #create corresponding y
    y = np.concatenate([[p for p in np.unique(kdist_dict[k]) if np.sum(kdist_dict[k]== p) >= 4] for k in sorted(kdist_dict.keys())])


    y_st = np.concatenate([kdist_dict[key] for key in sorted(kdist_dict.keys())])
    X_st = np.vstack([ np.reshape(np.swapaxes(np.swapaxes(mat_dict[key][:,25:-50,:],0,2),1,2)[:,[np.where(chan_list==channel)[0][0] for channel in interesting_ones],:],(-1,1000)) for key in sorted(mat_dict.keys()) ])



    # ### Kolossa et al. distorted probabilities for average ERPs

    # In[15]:

    normalize = True

    coefs = get_coefficients(encoding,X,y)


    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('{} coefficients w/ distorted probabilities'.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'avr_ERP_verzerrte_lo_probabilities_{}.pdf'.format(model_type))

    plt.clf()

    # ### Kolossa et al. distorted probabilities for Single Trial ERPs

    # In[14]:

    normalize = True

    coefs = get_coefficients(encoding,X_st,y_st)
    
    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('Single-Trial {} coefficients w/ distorted probabilities for model:  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'singletrial_verzerrte_lo_probs_{}.pdf'.format(model_type))


    plt.clf()
    # ### Correlations with KL-divergences

    # In[146]:

    # ### Distorted KLDs as in Kolossa et al.

    # In[10]:

    #list of participants, each entry the concatenated timecourses from 0 to 800 ms (4 ms sampling, no time binning)
    avr_ERP_p_kld_kdist_list = [ np.concatenate([np.array([ np.mean(mat_dict[k][np.where(chan_list==channel)[0],:,kld_kdist_dict[k]==i],axis=0) for i in np.unique(kld_kdist_dict[k]) ])[:,25:-50] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())] 

    #avr_ERP_p_kld_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],kld_dict[k],chan=np.where(chan_list==channel)[0])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]
    #%%
    #get rid of n < 4 values
    avr_ERP_p_kld_kdist_list = [ERPs[[nr for nr,p in enumerate(np.unique(kld_kdist_dict[sorted(kld_kdist_dict.keys())[nerp]])) if np.sum(kld_kdist_dict[sorted(kld_kdist_dict.keys())[nerp]] == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_kld_kdist_list)]


    #%%
    #now concatenate on axis of posteriors
    X = np.concatenate(avr_ERP_p_kld_kdist_list,axis=0)

    #create corresponding y
    y = np.concatenate([[p for p in np.unique(kld_kdist_dict[k]) if np.sum(kld_kdist_dict[k]== p) >= 4] for k in sorted(kld_kdist_dict.keys())])


    y_st = np.concatenate([kld_kdist_dict[key] for key in sorted(kld_kdist_dict.keys())])
    X_st = np.vstack([ np.reshape(np.swapaxes(np.swapaxes(mat_dict[key][:,25:-50,:],0,2),1,2)[:,[np.where(chan_list==channel)[0][0] for channel in interesting_ones],:],(-1,1000)) for key in sorted(mat_dict.keys()) ])



    # ### Correlation/Regression coefficients for distorted KLDs as in Kolossa et al. over average ERPs

    # In[12]:

    normalize = True

    coefs = get_coefficients(encoding,X_st,y_st)
    

    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('{} coefficients w/ distorted KLDs for model:  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'avr_ERP_verzerrte_KLDs_{}.pdf'.format(model_type))


    plt.clf()
    # ### Single-Trial Correlation/Regression coefficients for distorted KLDs as in Kolossa et al. 

    # In[13]:

    normalize = True

    coefs = get_coefficients(encoding,X_st,y_st)
   
    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('Single-Trial {} coefficients w/ distorted KLDs for model:  '.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'singletrial_verzerrte_KLDs_{}.pdf'.format(model_type))


    plt.clf()
# ### Predictive Surprise

# In[7]:
if model_type == 'predictive_surprise':
    #list of participants, each entry the concatenated timecourses from 0 to 800 ms (4 ms sampling, no time binning)
    avr_ERP_p_predictive_list = [ np.concatenate([np.array([ np.mean(mat_dict[k][np.where(chan_list==channel)[0],:,predictive_surprise_dict[k]==i],axis=0) for i in np.unique(predictive_surprise_dict[k]) ])[:,25:-50] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())] 

    #avr_ERP_p_kld_list = [np.concatenate([get_average_ERPs_per_posterior(mat_dict[k],kld_dict[k],chan=np.where(chan_list==channel)[0])[:,50:] for channel in interesting_ones],axis=1) for k in sorted(mat_dict.keys())]
    #%%
    #get rid of n < 4 values
    avr_ERP_p_predictive_list = [ERPs[[nr for nr,p in enumerate(np.unique(predictive_surprise_dict[sorted(predictive_surprise_dict.keys())[nerp]])) if np.sum(predictive_surprise_dict[sorted(predictive_surprise_dict.keys())[nerp]] == p) >= 4 ],:] for nerp,ERPs in enumerate(avr_ERP_p_predictive_list)]


    #%%
    #now concatenate on axis of posteriors
    X = np.concatenate(avr_ERP_p_predictive_list,axis=0)

    #create corresponding y
    y = np.concatenate([[p for p in np.unique(predictive_surprise_dict[k]) if np.sum(predictive_surprise_dict[k]== p) >= 4] for k in sorted(predictive_surprise_dict.keys())])


    # In[10]:

    
    coefs = get_coefficients(encoding,X,y)


    plt.imshow(coefs,aspect='auto',interpolation='nearest')
    plt.xticks([0,50,100,150,200],['0 ms','200 ms','400 ms','600 ms','800 ms'])
    plt.yticks([0,1,2,3,4],interesting_ones)
    plt.colorbar()
    plt.xlabel('ERP timecourse')
    plt.ylabel('Channels')
    plt.title('{} coefficients for predictive surprise ERP timecourse'.format({True:'Encoding',False:'Decoding'}[encoding]))
    plt.savefig(output_dir+'average_predictive_surprise.pdf')


    plt.clf()
