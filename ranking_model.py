import numpy as np
import final_adp_models
import final_team_model

'''
This is the working model. The one that the user can use. It takes in a full
ranking, possibly with adp numbers if the ranker feels fiesty, but not 
neccessarily. It outputs a new draft that has been appropriately modified. 

Also, positions have to be uppercase. Be warned. 
'''

#This is just to test the mechanics of the ranking. Not good as test case. 
test_ranking = [('Saquon Barkley', 'RB', 'NYG', 1.9),
                ('Alvin Kamara', 'RB', 'NO', 2.4),
                ('Christian McCaffrey', 'RB', 'CAR', 3.6),
                ('Ezekiel Elliot', 'RB', 'DAL', 3.8),
                ('James Connor', 'RB', 'PIT', 6.1),
                ('DeAndre Hopkins', 'WR', 'HOU', 6.1),
                ('LeVeon Bell', 'RB', 'NYJ', 7.2),
                ('Nick Chubb', 'RB', 'CLE', 7.2),
                ('Davante Adams', 'WR', 'GB', 9.2),
                ('Dalvin Cook', 'RB', 'MIN', 10.6),
                ('Julio Jones', 'WR', 'ATL', 10.9),
                ('Todd Gurley', 'RB', 'LAR', 11.4),
                ('Tyreek Hill', 'WR', 'KC', 12.9),
                ('Michael Thomas', 'WR', 'NO', 12.9)]

test_lay_ranking = [('Saquon Barkley', 'RB', 'NYG'),
                    ('Alvin Kamara', 'RB', 'NO'),
                    ('Christian McCaffrey', 'RB', 'CAR'),
                    ('Ezekiel Elliot', 'RB', 'DAL'),
                    ('James Connor', 'RB', 'PIT'),
                    ('DeAndre Hopkins', 'WR', 'HOU'),
                    ('LeVeon Bell', 'RB', 'NYJ'),
                    ('Nick Chubb', 'RB', 'CLE'),
                    ('Davante Adams', 'WR', 'GB'),
                    ('Dalvin Cook', 'RB', 'MIN'),
                    ('Julio Jones', 'WR', 'ATL'),
                    ('Todd Gurley', 'RB', 'LAR'),
                    ('Tyreek Hill', 'WR', 'KC'),
                    ('Michael Thomas', 'WR', 'NO')]

def identify_teams(ranking):
    #ranking is here a list of 4-tuples: (name, pos, team, adp)
    teams = set()
    for player in ranking:
        teams.add(player[2])
    return teams

def get_team_vector(ranking, team):
    #returns the team as a np-array to be fit to the model
    players = [player for player in ranking if player[2] == team]
    #first, get the quarterback implied points
    qbs = [player for player in players if player[1] == 'QB']
    qb1 = sorted(qbs, key = lambda qb : qb[3])[0] if len(qbs) > 0 else None
    qb_adp = qb1[3] if qb1 is not None else max(len(ranking) + 1, 32 * 7)
    qb_pts = final_adp_models.qb(qb_adp)
    #now, running back 1 adp
    rbs = [player for player in players if player[1] == 'RB']
    rb1 = sorted(rbs, key = lambda rb : rb[3])[0] if len(rbs) > 0 else None
    rb1_adp = rb1[3] if rb1 is not None else max(len(ranking) + 1, 32 * 7)
    rb1_pts = final_adp_models.rb(rb1_adp)
    #running back 2
    rbs = [player for player in players if player[1] == 'RB']
    rb2 = sorted(rbs, key = lambda rb : rb[3])[1] if len(rbs) > 1 else None
    rb2_adp = rb2[3] if rb2 is not None else max(len(ranking) + 1, 32 * 7)
    rb2_pts = final_adp_models.rb(rb2_adp)  
    #wide receiver 1
    wrs = [player for player in players if player[1] == 'WR']
    wr1 = sorted(wrs, key = lambda wr : wr[3])[0] if len(wrs) > 0 else None
    wr1_adp = wr1[3] if wr1 is not None else max(len(ranking) + 1, 32 * 7)
    wr1_pts = final_adp_models.wr(wr1_adp)   
    #wide receiver 2
    wrs = [player for player in players if player[1] == 'WR']
    wr2 = sorted(wrs, key = lambda wr : wr[3])[1] if len(wrs) > 1 else None
    wr2_adp = wr2[3] if wr2 is not None else max(len(ranking) + 1, 32 * 7)
    wr2_pts = final_adp_models.wr(wr2_adp)
    #return the result
    return np.array([qb_pts, rb1_pts, rb2_pts, wr1_pts, wr2_pts])    

def get_team_players(ranking, team):
    #returns the team as a np-array to be fit to the model
    players = [player for player in ranking if player[2] == team]
    #first, get the quarterback
    qbs = [player for player in players if player[1] == 'QB']
    qb1 = sorted(qbs, key = lambda qb : qb[3])[0] if len(qbs) > 0 else None
    qb1 = qb1[0] if not qb1 is None else None
    #now, running back 1 adp
    rbs = [player for player in players if player[1] == 'RB']
    rb1 = sorted(rbs, key = lambda rb : rb[3])[0] if len(rbs) > 0 else None
    rb1 = rb1[0] if not rb1 is None else None
    #running back 2
    rbs = [player for player in players if player[1] == 'RB']
    rb2 = sorted(rbs, key = lambda rb : rb[3])[1] if len(rbs) > 1 else None
    rb2 = rb2[0] if not rb2 is None else None
    #wide receiver 1
    wrs = [player for player in players if player[1] == 'WR']
    wr1 = sorted(wrs, key = lambda wr : wr[3])[0] if len(wrs) > 0 else None
    wr1 = wr1[0] if not wr1 is None else None
    #wide receiver 2
    wrs = [player for player in players if player[1] == 'WR']
    wr2 = sorted(wrs, key = lambda wr : wr[3])[1] if len(wrs) > 1 else None
    wr2 = wr2[0] if not wr2 is None else None
    #return the result
    return [qb1, rb1, rb2, wr1, wr2]

#This is the core, and it comes from adp numbers as rankings. 
def _rerank(ranking):
    teams = identify_teams(ranking)
    team_players = {team : get_team_players(ranking, team) for team in teams}
    team_vectors = {team : get_team_vector(ranking, team) for team in teams}
    new_ranking = []
    for player in ranking:
        team = player[2]
        teammates = team_players[team]
        if not player[0] in teammates:
            #running back 3, or tight end, or whatever. We fix their rankings. 
            #Sothey don't change ranks, the ones we model divvy up the free
            #ranks
            new_ranking.append((player, None))
        else:
            ind = teammates.index(player[0])
            raw_team_points = team_vectors[team]
            proj_team_points = final_team_model.project(raw_team_points)
            proj_points = proj_team_points[ind]
            #We have to convert these points back into adp
            if player[1] == 'RB':
                corr_proj_adp = final_adp_models.br(proj_points).real
            elif player[1] == 'QB':
                corr_proj_adp = final_adp_models.bq(proj_points).real
            elif player[1] == 'WR':
                corr_proj_adp = final_adp_models.rw(proj_points).real
            else:
                #end this loop cycle, don't add datapoint. Mistakes were made. 
                continue
            new_ranking.append((player, corr_proj_adp))
    #now, we rank the players that we can rank
    reranked_players = [p for p in new_ranking if p[1] is not None]
    open_slots = [ind for (ind, p) in enumerate(new_ranking) 
                  if p[1] is not None]
    reranked_players = sorted(reranked_players, key = lambda a : a[1])
    final_reranking = [p[0] if p[1] is None else None for p in new_ranking]
    for ind in range(len(open_slots)):
        final_reranking[open_slots[ind]] = reranked_players[ind]
    #modify the format to be what you want
    for ind in range(len(final_reranking)):
        final_reranking[ind] = (final_reranking[ind][0][0],
                                final_reranking[ind][0][1],
                                final_reranking[ind][0][2],
                                final_reranking[ind][1])
    return final_reranking
    
#this is the public-facing version for just a list of 3-tuples. 
def rerank(rankings):
    return _rerank([(r[0], r[1], r[2], i+1) for (i, r) in enumerate(rankings)]) 

'''
print(test_ranking)
print(_rerank(test_ranking))
'''

'''
print(test_lay_ranking)
print(rerank(test_lay_ranking))
'''
    