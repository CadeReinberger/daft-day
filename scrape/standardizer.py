import pandas as pd
import json
import numpy as np
import re
from scrape.params import START_YEAR, END_YEAR
import os

FORBIDDEN_TEAMCODES = ['NOR2017', 'ARI2017']

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def get_nfl_teams(year):
    base = 'scrape/' if not os.getcwd()[-6:] == 'scrape' else ''
    filename = base + 'clean_data/' + str(year) + '_leaders.xlsx'
    df = pd.read_excel(filename)
    teams = set()
    for tm in df['Tm']:
        #if handles 2, 3, etc team players
        if not tm[-2:] == 'TM': 
            teams.add(tm)
    #just a sanity check
    assert len(teams) == 32
    return teams

def get_datapoints(year):
    base = 'scrape/' if not os.getcwd()[-6:] == 'scrape' else ''
    filename = base + 'clean_data/' + str(year) + '_leaders.xlsx'
    df = pd.read_excel(filename)
    datapoints = {}
    ppg_datapoints = {}
    for team in get_nfl_teams(year):
        teamcode = team + str(year)
        #now get the qb who scored the most points for this team
        team_qbs = df[(df['FantPos'] == 'QB') & (df['Tm'] == team)]
        team_qbs = team_qbs.sort_values(by='FantPt', ascending=False)
        best_qb = team_qbs.iloc[0]
        qb_points = best_qb['FantPt']
        qb_ppg = qb_points / best_qb['G']
        qb_name = re.sub('[\W_]+', '', best_qb['Player'].lower())
        qb_playercode = qb_name + 'qb' + teamcode.lower()[-4:]
        qb_playercode = ''.join(hex(ord(c))[2:].zfill(2) 
                                for c in qb_playercode)
        #get the top two running backs
        team_rbs = df[(df['FantPos'] == 'RB') & (df['Tm'] == team)]
        team_rbs = team_rbs.sort_values(by='FantPt', ascending=False)
        rb1 = team_rbs.iloc[0]
        rb2 = team_rbs.iloc[1]
        rb1_points = rb1['FantPt']
        rb2_points = rb2['FantPt']
        rb1_ppg = rb1_points / rb1['G']
        rb2_ppg = rb2_points / rb2['G']
        rb1_name = re.sub('[\W_]+', '', rb1['Player'].lower())
        rb1_playercode = rb1_name + 'rb' + teamcode.lower()[-4:]
        rb1_playercode = ''.join(hex(ord(c))[2:].zfill(2) 
                                for c in rb1_playercode)
        rb2_name = re.sub('[\W_]+', '', rb2['Player'].lower())
        rb2_playercode = rb2_name + 'rb' + teamcode.lower()[-4:]
        rb2_playercode = ''.join(hex(ord(c))[2:].zfill(2) 
                                for c in rb2_playercode)
        #get the top two wide receivers
        team_wrs = df[(df['FantPos'] == 'WR') & (df['Tm'] == team)]
        team_wrs = team_wrs.sort_values(by='FantPt', ascending=False)
        wr1 = team_wrs.iloc[0]
        wr2 = team_wrs.iloc[1]
        wr1_points = wr1['FantPt']
        wr2_points = wr2['FantPt']
        wr1_ppg = wr1_points / wr1['G']
        wr2_ppg = wr2_points / wr2['G']
        wr1_name = re.sub('[\W_]+', '', wr1['Player'].lower())
        wr1_playercode = wr1_name + 'wr' + teamcode.lower()[-4:]
        wr1_playercode = ''.join(hex(ord(c))[2:].zfill(2) 
                                for c in wr1_playercode)
        wr2_name = re.sub('[\W_]+', '', wr2['Player'].lower())
        wr2_playercode = wr2_name + 'wr' + teamcode.lower()[-4:]
        wr2_playercode = ''.join(hex(ord(c))[2:].zfill(2) 
                                for c in wr2_playercode)
        #now we have our potential datapoints, let's clean them a bit.
        min_games = min([player['G'] for player in [best_qb, rb1, rb2, wr1, 
                                                    wr2]])
        #cutoff at 12, hand forbidden, or too early
        bad_point = min_games < 12 or teamcode in FORBIDDEN_TEAMCODES
        bad_point = bad_point or int(teamcode[-4:]) < 2008
        if not bad_point:
            #add the datapoints in appropriately. Really json-ify it.
            datapoints[teamcode] = [{qb_playercode : qb_points},
                                    {rb1_playercode : rb1_points}, 
                                    {rb2_playercode : rb2_points}, 
                                    {wr1_playercode : wr1_points}, 
                                    {wr2_playercode : wr2_points}]
            ppg_datapoints[teamcode] = [{qb_playercode : qb_ppg},
                                        {rb1_playercode : rb1_ppg}, 
                                        {rb2_playercode : rb2_ppg}, 
                                        {wr1_playercode : wr1_ppg}, 
                                        {wr2_playercode : wr2_ppg}]
    return datapoints, ppg_datapoints
        
def get_all_datapoints(start_year, end_year):
    #just combine all the datapoints
    years = range(start_year, end_year + 1)
    datapoints = dict(i for year in years 
                      for i in get_datapoints(year)[0].items())
    ppg_datapoints = dict(i for year in years
                          for i in get_datapoints(year)[1].items())
    return datapoints, ppg_datapoints

def get_split_datpoints(star_year, end_year):
    #splits up the data into totals as well
    dps, ppg_dps = get_all_datapoints(star_year, end_year)
    qrw_dps = {k : [sum(v[0].values()), 
                    sum(v[1].values()) + sum(v[2].values()),
                    sum(v[3].values()) + sum(v[4].values())] 
               for (k, v) in ppg_dps.items()}
    qrw_ppg_dps = {k : [sum(v[0].values()), 
                        sum(v[1].values()) + sum(v[2].values()),
                        sum(v[3].values()) + sum(v[4].values())] 
                   for (k, v) in ppg_dps.items()}
    return dps, ppg_dps, qrw_dps, qrw_ppg_dps

def standardize_and_save_data(start_year = START_YEAR, end_year = END_YEAR):
    dps, ppg_dps, qrw_dps, qrw_ppg_dps = get_split_datpoints(start_year, 
                                                             end_year)
    jsons = {'sep_tot' : dps, 
             'sep_ppg' : ppg_dps, 
             'pos_tot' : qrw_dps, 
             'pos_ppg' : qrw_ppg_dps}
    for (name, dataset) in jsons.items():
        base = 'scrape/' if not os.getcwd()[-6:] == 'scrape' else ''
        filename = base + 'points/' + name + '_datapoints.json'
        with open(filename, 'w+') as f:
            json.dump(dataset, f, default = np_encoder, indent = 2)

standardize_and_save_data()
    