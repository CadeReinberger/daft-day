import numpy as np
from scipy.special import lambertw

## QUARTERBACK MODEL
qb_params = np.load('adp_points_models/qb_model.npy')
[qba, qbb, qbc] = list(qb_params)

def qb(adp):
    return qba * (adp ** qbb) * np.exp(qbc * adp)

## RUNNING BACK MODEL
rb_params = np.load('adp_points_models/rb_model.npy')
[rba, rbb, rbc] = list(rb_params)

def rb(adp):
    return rba * (adp ** rbb) * np.exp(rbc * adp)

## WIDE RECEIVER MODEL
wr_params = np.load('adp_points_models/wr_model.npy')
[wra, wrb, wrc] = list(wr_params)

def wr(adp):
    return wra * (adp ** wrb) * np.exp(wrc * adp)

## TIGHT END MODEL
te_params = np.load('adp_points_models/te_model.npy')
[tea, teb, tec] = list(te_params)

def te(adp):
    return tea * (adp ** teb) * np.exp(tec * adp)


'''
We can invert this function algebraically. As it turns out, the inverse to 
the transmonomial 
            p = a*x^b*exp(c*x)
is given by 
            x = (b/c)*W((c/b)*(p/a)^(1/b))
where W is the principle branch of the Lambert W function. 
'''

## INVERSE QUARTERBACK MODEL
def bq(points):
    return (qbb / qbc) * lambertw((qbc / qbb) * ((points / qba) ** (1 / qbb)))

## INVERSE RUNNINGBACK MODEL
def br(points):
    return (rbb / rbc) * lambertw((rbc / rbb) * ((points / rba) ** (1 / rbb)))

## INVERSE WIDE RECEIVER MODEL
def rw(points):
    return (wrb / wrc) * lambertw((wrc / wrb) * ((points / wra) ** (1 / wrb)))

## INVERSE TIGHT END MODEL
def et(points):
    return (teb / tec) * lambertw((tec / teb) * ((points / tea) ** (1 / teb)))
