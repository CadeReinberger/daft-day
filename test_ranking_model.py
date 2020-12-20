from ranking_model import _rerank
import pandas as pd
import json
import re
import numpy as np
from scipy.stats import spearmanr
from scrape.params import START_YEAR, END_YEAR

'''
In this module we test to determine if our model really works. We use adp data, 
and then we check if the square of the spearman rho correlation coming from the 
indices of the rankings pre and post transfomation are improved when compared 
with the actual fantasy rankings at each position. That's the test, and we'll 
see. 
'''

def get_adp_df(year):
    return pd.read_excel('scrape/adp/' + str(year) + '_adp.xlsx')

def get_rankings_from_adp(year):
    df = get_adp_df(year)
    rankings = []
    for (index, player) in df.iterrows():
        name = re.sub('[\W_]+', '', player['Name'].lower())
        pos = player['Pos'].lower()
        playercode = name + pos + str(year)
        playercode = ''.join(hex(ord(c))[2:].zfill(2) for c in playercode)
        adp = player['Overall']
        if player['Team'] == 'None':
            #just some last minute cleaning. We just ignore them. :(
            continue
        rankings.append((playercode, pos.upper(), player['Team'], adp))  
    return rankings

def get_madp_datapoints(pos, year):
    pos = pos.lower()
    reranked = _rerank(get_rankings_from_adp(year))
    datapoints = []
    with open('scrape/player_points/data.json') as f:
        scores_dict = json.load(f)
    for (playercode, position, team, madp) in reranked:
        if position == pos.upper():
            if playercode in scores_dict and scores_dict[playercode] > 25:
                datapoint = np.array([madp, scores_dict[playercode]])
                datapoints.append(datapoint)
    return datapoints

def get_adp_datapoints(pos, year):
    pos = pos.lower()
    df = get_adp_df(year)
    df = df[df['Pos'] == pos.upper()]
    datapoints = []
    with open('scrape/player_points/data.json') as f:
        scores_dict = json.load(f)
    for (index, player) in df.iterrows():
        name = re.sub('[\W_]+', '', player['Name'].lower())
        playercode = name + pos + str(year)
        playercode = ''.join(hex(ord(c))[2:].zfill(2) for c in playercode)
        adp = player['Overall']
        #the 25 makes outlier handling a lot nicer
        if playercode in scores_dict and scores_dict[playercode] > 25:
            datapoint = np.array([adp, scores_dict[playercode]])
            datapoints.append(datapoint)
    return datapoints
     
def get_all_adp_datapoints(pos, start_year = START_YEAR, end_year = END_YEAR):
    years = range(start_year, end_year + 1)
    return [dp for year in years for dp in get_adp_datapoints(pos, year)]

def get_all_madp_datapoints(pos, start_year = START_YEAR, end_year = END_YEAR):
    years = range(start_year, end_year + 1)
    return [dp for year in years for dp in get_madp_datapoints(pos, year)]

def compare_spearmans(pos):
    adp_points = get_all_adp_datapoints(pos)
    madp_points = get_all_madp_datapoints(pos)
    #get adp spearm
    adps = np.array([dp[0] for dp in adp_points])
    points = np.array([-dp[1] for dp in adp_points]) #makes corr positive
    adp_spearman = spearmanr(adps, points)[0]
    #get madp spearman
    madps = np.array([dp[0] for dp in madp_points])
    points = np.array([-dp[1] for dp in madp_points]) #makes corr positive
    madp_spearman = spearmanr(madps, points)[0]
    #calculate model imporovement
    improvement = (madp_spearman - adp_spearman) / adp_spearman
    improvement_percentage = improvement * 100
    print(pos.upper() + ' RESULTS')
    print('-' * 50)
    print('RAW SPEARMAN:         ' + str(adp_spearman))
    print('MODEL SPEARMAN:       ' + str(madp_spearman))
    print('MODEL IMPROVEMENT[%]: ' + str(improvement_percentage) )
    print('-' * 50 + '\n\n')
    
def compare_full():    
    #just makes sure these bad boys are updated. I had to chagne this and
    from scrape import standardizer
    import adp_points_train
    import team_relationship_train
    print('\n')
    compare_spearmans('QB')
    compare_spearmans('RB')
    compare_spearmans('WR')
    
compare_full()
    