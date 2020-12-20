import pandas as pd
import re
import json
import numpy as np

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def get_player_datapoints(year):
    filename = 'clean_data/' + str(year) + '_leaders.xlsx'
    df = pd.read_excel(filename)
    playerpoints = []
    for (index, row) in df.iterrows():
        player_name = re.sub('[\W_]+', '', row['Player'].lower())
        playercode = player_name + row['FantPos'].lower() + str(year)
        playercode = ''.join(hex(ord(c))[2:].zfill(2) for c in playercode)
        points = row['FantPt']
        #to get rid of the injury noise, we require playing in 13 games
        if row['G'] > 12:
            playerpoints.append((playercode, points))
    return playerpoints
        
def get_all_player_datapoints(start_year = 2005, end_year = 2019):
    all_points = [point for year in range(start_year, end_year + 1) 
                  for point in get_player_datapoints(year)]
    points_dict = dict(all_points)
    with open('player_points/data.json', 'w+') as f:
        json.dump(points_dict, f, default = np_encoder, indent = 2)
    
get_all_player_datapoints()