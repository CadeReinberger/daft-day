import numpy as np
import pickle

C = np.load('team_models/C.npy')
mu = np.load('team_models/mu.npy')
with open('team_models/kpca.pkl', 'rb') as pickle_file:
    kpca = pickle.load(pickle_file)
    
ALPHA = .05 #WEIGHT ON PCA VALUES

def pca_project(team_vec):
    return (team_vec - mu) @ (C.T @ C) + mu

def kpca_project(team_vec):
    return kpca.inverse_transform(kpca.transform(team_vec.reshape(1, -1)))[0]

def pca_soft_project(team_vec, alpha):
    raw = team_vec
    projected = pca_project(team_vec)
    return alpha * projected + (1 - alpha) * raw
    
def project(team_vec):
    #active model is pca version
    return pca_soft_project(team_vec, ALPHA)
    
