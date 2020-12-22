# daft-day
A repository with some adp-based models with application to fantasy football. The idea is to take a set of pre-season rankings, and to give them a slight edge by moving them towards more realistic values for the way the players between teams actually behave. 

## The model

The model works in a two-fold manner. Firstly, it associates adp (average draft position values) of a player at each fatnasy position with their expected fantasy output for the season. Secondly, does 4-component principle component analysis on the 5 most important fantasy players (The quarterback and top two wide receivers and top two running backs) of historical fantasy football season. For a given set of rankings, then, the model will look at the expected fantasy outputs of each team, based on the adp to points model, and adjust player scores based on their teammate's scores to gain a slight edge. 

The adp to points model is a transmonomial (the product of an exponential and power law), trained by taking logs of both sides of the datapoints and then using multiple linear regression via ordinary least squares. It's trained for each position, with data coming from historical adp numbers that were scraped rather easily from the web, compared with overall points numbers all scraped fairly easily from the web. 

The PCA model is trained on the historical points data used for the adp model, and I just have numpy save the arrays with the paramaters of both models once trained. Then, it adjusts ranking to gain a bit of an advantage. 


