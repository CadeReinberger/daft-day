import json
import numpy as np
from sklearn.decomposition import PCA

'''
Our working idea is to do the easiest possible thing and fit a hyperplane to 
the space of team 5-tuples. That is, we do a 4-component pca on the separated
distribution of separated fantasy points. We'll use this with rankings in due
time, but this is just the four-component PCA. So you project to the best-fit
hyperplane
'''

def get_datapoints():
    #just returns the list of datapoints
    with open('scrape/points/sep_tot_datapoints.json', 'r') as f:
        return [np.array([sum(d.values()) for d in p]) 
                for p in json.load(f).values()]
    
def train_pca_model():
    data = get_datapoints()
    pca = PCA(n_components = 4)
    pca.fit(data)
    explained_variance = sum(pca.explained_variance_ratio_)
    #print(explained_variance)
    #4-dimensional model explains about 94.7% of the variance. Promising. 
    return (pca.components_, pca.mean_, explained_variance)

def save_pca_model():
    #we'll simply save the numpy arrays
    (C, mu, ev) = train_pca_model()
    np.save('team_models/C.npy', C)
    np.save('team_models/mu.npy', mu)

save_pca_model()
