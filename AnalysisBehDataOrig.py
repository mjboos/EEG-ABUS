
# coding: utf-8

# # Modelling of Behavioral Data with distorted probabilities

# In[2]:

#import re
import pystan
import pandas as pd
import numpy as np
from collections import defaultdict
import pylab as plt
import seaborn as sns
from scipy.stats import norm
import joblib

def log_odds(x):
    return np.log(x/(1-x))

def logist(x):
    return 1/(1+np.exp(-x))

def computeWAIC(gam,pnull,y,N,x):
    '''Returns WAIC for extracted gam, pnull from binomial model'''
    from scipy.stats import binom
    pD = 0
    ll = 0
    lppd = 0
    for i in xrange(5):
        for j in xrange(16):
            ll = binom.pmf(y[j,i], N[j,i],
                           logist(x[i]*gam[:,j] + (1-gam[:,j])*log_odds(pnull[:,j])))
            pD = pD + np.var(np.log(ll))
            lppd = lppd + np.log(np.mean(ll))
    WAIC = (-2*(lppd-pD))
    return WAIC

def computeWAIC_components(gam,pnull,y,N,x):
    '''Returns WAIC for extracted gam, pnull from binomial model'''
    from scipy.stats import binom
    pD = 0
    ll = 0
    lppd = 0
    lppd_pD = []
    for i in xrange(5):
        for j in xrange(16):
            ll = binom.pmf(y[j,i], N[j,i],
                           logist(x[i]*gam[:,j] + (1-gam[:,j])*log_odds(pnull[:,j])))
            pD = np.var(np.log(ll))
            lppd = np.log(np.mean(ll))
            lppd_pD.append(lppd-pD)
    return np.array(lppd_pD)


def computeLPPD(gam,pnull,y,N,x):
    '''Returns log-posterior predictive density (LPPD) for extracted gam,pnull from binomial model'''
    from scipy.stats import binom
    ll = 0
    lppd = 0
    for i in xrange(5):
        ll = binom.pmf(y[i], N[i], 
                       logist(x[i]*gam + (1-gam)*log_odds(pnull)))
        lppd = lppd + np.log(np.mean(ll))
    return lppd

def computeLPPD_components(gam,pnull,y,N,x):
    '''Returns log-posterior predictive density (LPPD) for extracted gam,pnull from binomial model'''
    from scipy.stats import binom
    ll = 0
    lppd = []
    for i in xrange(5):
        ll = binom.pmf(y[i], N[i], 
                       logist(x[i]*gam + (1-gam)*log_odds(pnull)))
        lppd.append(np.log(np.mean(ll)))
    return np.array(lppd)


def get_pred(x,gam,pnull):
    return logist(gam*log_odds(x) + (1-gam)*log_odds(pnull)) 


sns.set_style('white')


# In[3]:

nopooling_model = '''
data {
int<lower=0> npb;
int<lower=0> cases;
int<lower=0> N[npb,cases];
int<lower=0> y[npb,cases];
real X[cases];
}
parameters {


real<lower=0,upper=1> pzero[npb];
real<lower=0> gam[npb];
}

model {
for (j in 1:npb)
{
gam[j] ~ normal(1,1) T[0,];
pzero[j] ~ beta(1,1);
}
for (i in 1:cases)
{
for (j in 1:npb)
y[j,i] ~ binomial_logit(N[j,i],X[i]*gam[j] + (1-gam[j])*log(pzero[j]/(1-pzero[j])));
}
}
'''

fullpooling_model = '''
data {
int<lower=0> npb;
int<lower=0> cases;
int<lower=0> N[npb,cases];
int<lower=0> y[npb,cases];
real X[cases];
}
parameters {
real<lower=0,upper=1> pzero;
real<lower=0> gam;
}

model {

gam ~ normal(1,1) T[0,];

pzero ~ beta(1,1);


for (i in 1:cases)
{
for (j in 1:npb)
y[j,i] ~ binomial_logit(N[j,i],X[i]*gam + (1-gam)*log(pzero/(1-pzero)));
}
}
'''

hierarchic_model = '''
data {
int<lower=0> npb;
int<lower=0> cases;
int<lower=0> N[npb,cases];
int<lower=0> y[npb,cases];
real X[cases];
}
parameters {

real<lower=0,upper=1> phi;
real<lower=0.1> lambda;
real<lower=0> mu_gam;
real<lower=0> tau_gam;


real<lower=0,upper=1> pzero[npb];
real<lower=0> gam[npb];
}

model {
real alpha;
real beta;
phi ~ beta(1,1);
lambda ~ pareto(0.1,1.5);


alpha <- lambda * phi;
beta <- lambda * (1 - phi);

mu_gam ~ normal(1,1) T[0,];
gam ~ normal(mu_gam,tau_gam);

pzero ~ beta(alpha,beta);


for (i in 1:cases)
{
for (j in 1:npb)
y[j,i] ~ binomial_logit(N[j,i],X[i]*gam[j] + (1-gam[j])*log(pzero[j]/(1-pzero[j])));
}
}
'''


# In[4]:


#there are 16 vps
npb = 16
#and 14 (12 without 1st ball) different cases for the binomial distribution
cases = 5
path = '/home/mboos/Work/Bayesian Updating/Old Data/'

freq_urn_cc = np.loadtxt(path+'freq_urn_cc.txt')
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


# In[5]:


prior_values = [0.1,0.3]
likelihood_values = [0.9,0.7]

lo_xs = list()

for prior in prior_values:
    for likelihood in likelihood_values:
        posterior = np.array([ (prior*likelihood**i*(1-likelihood)**(4-i))                               / ((prior*likelihood**i*(1-likelihood)**(4-i))+((1-prior)*(1-likelihood)**i*likelihood**(4-i))) for i in xrange(5) ])
        lo_xs.append(np.log(posterior/(1-posterior)))

#create the data-structures
data_cc = {"npb" : npb,"cases":cases,"N":N_cc.astype(int),"y":rare_urn_cc.astype(int),"X":lo_xs[0]}
data_cu = {"npb" : npb,"cases":cases,"N":N_cu.astype(int),"y":rare_urn_cu.astype(int),"X":lo_xs[1]}
data_uc = {"npb" : npb,"cases":cases,"N":N_uc.astype(int),"y":rare_urn_uc.astype(int),"X":lo_xs[2]}
data_uu = {"npb" : npb,"cases":cases,"N":N_uu.astype(int),"y":rare_urn_uu.astype(int),"X":lo_xs[3]}


# In[5]:

#create fit objects
fit_parpool = pystan.stan(model_code=hierarchic_model,data=data_cc)
fit_fullpool = pystan.stan(model_code=fullpooling_model,data=data_cc)
fit_nopool = pystan.stan(model_code=nopooling_model,data=data_cc)


# In[36]:

#fit now
WAIC = defaultdict(dict)

for modeltype, modelfit in zip (['full pooling','no pooling','partial pooling'],[fit_fullpool,fit_nopool,fit_parpool]):
    for condition, data in zip(['cc','uc','cu','uu'],[data_cc,data_uc,data_cu,data_uu]):
        fit = pystan.stan(fit=modelfit, data=data,iter=10000, chains=20)
        extr = fit.extract(['gam','pzero'])
        with open('paper_fitdumps/fitdump_{}_{}.txt'.format(modeltype,condition),'w+') as fh:
            print >> fh, fit
        if modeltype == 'full pooling':
            WAIC[modeltype][condition] = computeWAIC_components(np.tile(extr['gam'][:,None],data['npb'])                                         ,np.tile(extr['pzero'][:,None],data['npb']),data['y'],data['N'],data['X'])
        else:
            WAIC[modeltype][condition] = computeWAIC_components(extr['gam'],extr['pzero'],data['y'],data['N'],data['X'])

joblib.dump(WAIC, 'waic_components.pkl', compress=3)

# In[14]:

#computes -2 * leave-one-out cross-validation of the different models

LOCV_LPPD = defaultdict(dict)

for modeltype, modelfit in zip (['full pooling','partial pooling'], [fit_fullpool,fit_parpool]):
    for condition, data in zip(['cc','uc','cu','uu'], [data_cc,data_uc,data_cu,data_uu]):
        lo_lppd = []
        for leftout_subj in xrange(16):
            lo_data = data.copy()
            lo_data['npb'] = 15
            test_y = lo_data['y'][leftout_subj,:]
            test_N = lo_data['N'][leftout_subj,:]
            lo_data['y'] = np.delete(lo_data['y'],obj=leftout_subj,axis=0)
            lo_data['N'] = np.delete(lo_data['N'],obj=leftout_subj,axis=0)
            fit = pystan.stan(fit=modelfit, data=data,iter=10000, chains=20)
            if modeltype == 'full pooling':
                extr = fit.extract(['gam','pzero'])
                lo_lppd.append(computeLPPD_components(extr['gam'],extr['pzero'],test_y,test_N,data['X']))
            else:
                extr = fit.extract(['mu_gam','phi','tau_gam','lambda'])
                gam = np.random.normal(extr['mu_gam'],extr['tau_gam'])
                alpha = extr['phi'] * extr['lambda']
                beta = extr['lambda'] * (1-extr['phi'])
                pnull = np.random.beta(alpha,beta)
                lo_lppd.append(computeLPPD_components(gam,pnull,test_y,test_N,data['X']))
        LOCV_LPPD[modeltype][condition] = -2*np.array(lo_lppd)

joblib.dump(LOCV_LPPD,'locv_components.pkl',compress=3)



