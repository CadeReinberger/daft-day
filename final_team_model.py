import numpy as np

C = np.load('team_models/C.npy')
mu = np.load('team_models/mu.npy')

def project(team_vec):
    return (team_vec - mu) @ (C.T @ C) + mu
