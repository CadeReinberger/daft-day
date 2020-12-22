import pandas as pd
import numpy as np
import re
import json
from matplotlib import pyplot as plt
import sklearn.linear_model
from scipy.stats import pearsonr
from scrape.params import START_YEAR, END_YEAR

def get_adp_df(year):
    return pd.read_excel('scrape/adp/' + str(year) + '_adp.xlsx')

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
    
def adp_scatter_plot(pos):
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    plt.figure()
    plt.scatter(adps, pts)
    plt.show()
    
def adp_log_log_plot(pos):
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    plt.figure()
    plt.scatter(np.log(adps), np.log(pts))
    plt.show()

def adp_log_plot(pos):
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    plt.figure()
    plt.scatter(adps, np.log(pts))
    plt.show()
    
def get_pearson_correlation_squared(pos):
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    return pearsonr(adps, pts)[0] ** 2

def fit_adp_linear_model(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    reg = sklearn.linear_model.LinearRegression().fit(adps, pts)
    r_squared = reg.score(adps, pts)
    return reg.intercept_, reg.coef_[0], r_squared
    
def fit_adp_power_law(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    log_adps = np.log(adps)
    log_pts = np.log(pts)
    reg = sklearn.linear_model.LinearRegression().fit(log_adps, log_pts)
    r_squared = reg.score(log_adps, log_pts)
    #(a, b, c) corresponds to model ax^b with r^2 = c
    return np.exp(reg.intercept_), reg.coef_[0], r_squared

def fit_adp_exponential_model(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    log_pts = np.log(pts)
    reg = sklearn.linear_model.LinearRegression().fit(adps, log_pts)
    r_squared = reg.score(adps, log_pts)
    #(a, b, c) corresponds to model a*exp(bx) with r^2 = c
    return np.exp(reg.intercept_), reg.coef_[0], r_squared

def check_models_viability(pos):
    adp_scatter_plot(pos)
    adp_log_log_plot(pos)
    adp_log_plot(pos)
    print(get_pearson_correlation_squared(pos))
    print(fit_adp_linear_model(pos))
    print(fit_adp_power_law(pos))
    print(fit_adp_exponential_model(pos))

'''
Empirically, especially based on the graphs, an exponential model is best
so from this point forward we look to fitting the specific exponential model 
that we'll use for adp improvement based on team values
'''

def reg_plot_reg_least_squares_exponential_model(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, omega, r_sq) = fit_adp_exponential_model(pos)
    model = lambda adp : a * np.exp(omega * adp)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()
    
'''
Plotting the regular least squares shows that we have the right idea, but the 
fit could be improved. Regular least squares comes from quadratic loss when it
is the case that the residuals are gaussian. But for fantasy points, when adp
is high, the model is highly skewed right--there's breakout potential, and 
those playing poorly are simply benched or cut. We set out to fix this problem.
'''

def lasso_exponential_model_fit(pos, alpha):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    log_pts = np.log(pts)
    reg = sklearn.linear_model.Lasso(alpha=alpha)
    reg.fit(adps, log_pts)
    r_squared = reg.score(adps, log_pts)
    #(a, b, c) corresponds to model a*exp(bx) with r^2 = c
    return np.exp(reg.intercept_), reg.coef_[0], r_squared

def reg_plot_lasso_exponential_model(pos, alpha):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, omega, r_sq) = lasso_exponential_model_fit(pos, alpha)
    #print((a, omega, r_sq))
    model = lambda adp : a * np.exp(omega * adp)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()

'''
Long story short, Lasso doesn't help. It may be the case that black-box linear 
modelling won't quite do the trick, in which case I can play some more fitting
games, but let's try a few more tricks up the ol' sleeve first. 
'''   

def lars_exponential_model_fit(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    log_pts = np.log(pts)
    reg = sklearn.linear_model.Lars()
    reg.fit(adps, log_pts)
    r_squared = reg.score(adps, log_pts)
    #(a, b, c) corresponds to model a*exp(bx) with r^2 = c
    return np.exp(reg.intercept_), reg.coef_[0], r_squared

def reg_plot_lars_exponential_model(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, omega, r_sq) = lars_exponential_model_fit(pos)
    #print((a, omega, r_sq))
    model = lambda adp : a * np.exp(omega * adp)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()
    
    
'''
Lars looks better doubtless. Maybe skewness of residuals as a problem is over-
stated. Let's try something outlier resitsant. LARS is nice to have as a base-
line, though, and it'll probably do the trick if it has to
'''
    
def huber_exponential_model_fit(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    log_pts = np.log(pts)
    reg = sklearn.linear_model.HuberRegressor()
    reg.fit(adps, log_pts)
    r_squared = reg.score(adps, log_pts)
    #(a, b, c) corresponds to model a*exp(bx) with r^2 = c
    return np.exp(reg.intercept_), reg.coef_[0], r_squared

def reg_plot_huber_exponential_model(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, omega, r_sq) = huber_exponential_model_fit(pos)
    #print((a, omega, r_sq))
    model = lambda adp : a * np.exp(omega * adp)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()
    
'''
Huber looks pretty good. We'll probably stick with it's conclusion as our model
, and try and see what we can do about fitting a rankings-changer based on the 
Huber linear fitting of an exponential decay model. I'm not convinced it really
gets it right when adp is small, though, so let's do a bit of robustly-fit 
power law checking, at least to see
'''

def huber_power_model_fit(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    log_adps = np.log(adps)
    log_pts = np.log(pts)
    reg = sklearn.linear_model.HuberRegressor()
    reg.fit(log_adps, log_pts)
    r_squared = reg.score(log_adps, log_pts)
    #(a, b, c) corresponds to model a*x^b with r^2 = c
    return np.exp(reg.intercept_), reg.coef_[0], r_squared

def reg_plot_huber_power_model(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, b, r_sq) = huber_power_model_fit(pos)
    #print((a, b, r_sq))
    model = lambda adp : a * (adp ** b)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()
    
'''
The Huber-fit power law is much more aggressive. It feels true to the modern 
league qualitatively, but it seems to caputr ethe data less well. I really 
don't know how the noise is distributed, so I'm no sure. I don't hate it, but 
it's hard to say. There is some promise, though
'''

def ransac_power_model_fit(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([[dp[0]] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    log_adps = np.log(adps)
    log_pts = np.log(pts)
    reg = sklearn.linear_model.RANSACRegressor()
    reg.fit(log_adps, log_pts)
    estimator = reg.estimator_
    r_squared = reg.score(log_adps, log_pts)
    #(a, b, c) corresponds to model a*x^b with r^2 = c
    return np.exp(estimator.intercept_), estimator.coef_[0], r_squared

def reg_plot_ransac_power_model(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, b, r_sq) = ransac_power_model_fit(pos)
    #print((a, b, r_sq))
    model = lambda adp : a * (adp ** b)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()

'''
Has good characteristics, but that ain't it either, Mahomes (get it, Chief). 
I am gonna try something. Let's fit a transmonomial. What could happen? 
'''

def theil_sen_transmonomial_fit(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([dp[0] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    reg_in = np.array([[np.log(adp), adp] for adp in adps])
    reg_out = np.log(pts)
    reg = sklearn.linear_model.TheilSenRegressor()
    reg.fit(reg_in, reg_out)
    r_squared = reg.score(reg_in, reg_out)
    #(a, b, c, d) corresponds to model a*x^b*exp(cx) with r^2 = d
    return (np.exp(reg.intercept_), reg.coef_[0], reg.coef_[1], r_squared)

def reg_plot_theil_sen_transmonomial(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, b, c, r_sq) = theil_sen_transmonomial_fit(pos)
    #print((a, b, c, r_sq))
    model = lambda adp : a * (adp ** b) * np.exp(c * adp)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()
    
'''
I actually really like the shape here, and I think the model can totally 
afford it with the data. I don't like the fit so much. Let's try good ol'
fashioned last squares and Lars on this model, since I'm guessing I'm going to 
take which of those I like the best
'''

def least_squares_transmonomial_fit(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([dp[0] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    reg_in = np.array([[np.log(adp), adp] for adp in adps])
    reg_out = np.log(pts)
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(reg_in, reg_out)
    r_squared = reg.score(reg_in, reg_out)
    #(a, b, c, d) corresponds to model a*x^b*exp(cx) with r^2 = d
    return (np.exp(reg.intercept_), reg.coef_[0], reg.coef_[1], r_squared)

def reg_plot_least_squares_transmonomial(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, b, c, r_sq) = least_squares_transmonomial_fit(pos)
    #print((a, b, c, r_sq))
    model = lambda adp : a * (adp ** b) * np.exp(c * adp)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()
    
'''
I like that fit better. Still don't love it, especially for qbs, but it's 
definitely better. We'll see about LARS
'''

def lars_transmonomial_fit(pos):
    dps = get_all_adp_datapoints(pos)
    adps = np.array([dp[0] for dp in dps])
    pts = np.array([dp[1] for dp in dps])
    reg_in = np.array([[np.log(adp), adp] for adp in adps])
    reg_out = np.log(pts)
    reg = sklearn.linear_model.Lars()
    reg.fit(reg_in, reg_out)
    r_squared = reg.score(reg_in, reg_out)
    #(a, b, c, d) corresponds to model a*x^b*exp(cx) with r^2 = d
    return (np.exp(reg.intercept_), reg.coef_[0], reg.coef_[1], r_squared)

def reg_plot_lars_transmonomial(pos):
    #plot this in the regular (un-log-ed) space, least squares fit
    #corall the data a bit
    dps = get_all_adp_datapoints(pos)
    adps = [dp[0] for dp in dps]
    pts = [dp[1] for dp in dps]
    #now fit the model
    (a, b, c, r_sq) = lars_transmonomial_fit(pos)
    #print((a, b, c, r_sq))
    model = lambda adp : a * (adp ** b) * np.exp(c * adp)
    #get points to plot the model
    xs = np.linspace(min(adps), max(adps), 10000)
    ys = model(xs)
    #plot everything        
    plt.figure()
    plt.scatter(adps, pts)
    plt.plot(xs, ys, color = 'orange')
    plt.show()

'''
Well that did literally nothing. Not in that regime, I guess. Really, I like
the ordinary least squares model, but I don't like it for quarterback. I would
conjecture that maybe that's me being a dumb-dumb though. Let's roll with the 
ordinary least squares transomonial as our working model, and maybe if we end
up with some real problems we'll figure it out. 
'''

def train_models():
    #We have to do this in top level because of pickle weirdness. 
    #Can't loop because it's a sacrifice I
    positions = ['qb', 'rb', 'wr', 'te'] 
    for pos in positions:
        (a, b, c, r_sq) = least_squares_transmonomial_fit(pos)
        #Can't use a lambda expression becuase of pickle weirdness
        params = np.array([a, b, c])
        np.save('adp_points_models/' + pos + '_model.npy', params)
        
#train_models()
        
reg_plot_least_squares_transmonomial('qb')


        