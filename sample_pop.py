import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #does 3D garbage
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

'''
The purpose of this file is to get a handle on the data. Really, to see if our
cleaning has done enough. Meh. This module is, I think, the only one that's
useless in terms of the ranking model, but we keep the code anyway
'''

class data_styles:
    #syntactic sugar to act like an enum
    sep_tot = 'sep_tot'
    sep_ppg = 'sep_ppg'
    pos_tot = 'pos_tot'
    pos_ppg = 'pos_ppg'
    
def get_datapoints(style):
    #just returns the list of datapoints
    with open('scrape/points/' + style + '_datapoints.json', 'r') as f:
        if style[:3] == 'sep':
            return [np.array([sum(d.values()) for d in p]) 
                    for p in json.load(f).values()]
        return [np.array(p) for p in json.load(f).values()]

def plot_pos_tot_numbers():
    #3d scatter plot of data points
    data = get_datapoints(data_styles.pos_tot)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    qbs = [dp[0] for dp in data]
    rbs = [dp[1] for dp in data]
    wrs = [dp[2] for dp in data]
    ax.scatter(qbs, rbs, wrs, marker = 'o')
    ax.set_xlabel('QB Points')
    ax.set_ylabel('RB Points')
    ax.set_zlabel('WR Points')
    ax.view_init(elev=-10, azim=120)
    plt.show() #just show it, for now
    
def plot_pair_of_datapoints(ind1, ind2, is_tot = True):
    data = get_datapoints(data_styles.sep_tot 
                          if is_tot else data_styles.sep_ppg)
    xs = [dp[ind1] for dp in data]
    ys = [dp[ind2] for dp in data]
    plt.figure()
    plt.scatter(xs, ys)
    plt.show()
    
def compute_pca(data_style):
    data = get_datapoints(data_style)
    #full dimension just to see how it works
    pca = PCA(n_components = 3 if data_style[:3] == 'pos' else 5)
    pca.fit(data)
    return pca.components_, pca.explained_variance_ratio_    
    
    