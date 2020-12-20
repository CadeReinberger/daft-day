import pandas as pd
from time import sleep

def get_scrape_cite_url(year):
    return ('https://www.pro-football-reference.com/years/' 
            + str(year) + '/fantasy.htm')

def get_scrape_cite_dataframe(year):
    tables = pd.read_html(get_scrape_cite_url(year))
    return tables[0]

def scrape_all(start_year = 2005, end_year = 2019):
    for year in range(start_year, end_year + 1):
        filename = 'data/' + str(year) + '_leaders.xlsx'
        get_scrape_cite_dataframe(year).to_excel(filename)
        sleep(.2)
        
scrape_all()