import pandas as pd
import requests
from time import sleep
from random import random as rand

def get_url(year):
    base = 'https://fantasyfootballcalculator.com/adp/standard/12-team/all/'
    url = base + str(year)
    return url

def get_table(year):
    #pretend to be firefox
    header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
              "X-Requested-With": "XMLHttpRequest"}
    r = requests.get(get_url(year), headers = header)    
    return pd.read_html(r.text)[0]

def scrape_years(min_year = 2008, max_year = 2019):
    for year in range(min_year, max_year + 1):
        df = get_table(year)
        df.to_excel('adp/' + str(year) + '_adp.xlsx')
        sleep(.2 + .3 * rand())
        
scrape_years()
    