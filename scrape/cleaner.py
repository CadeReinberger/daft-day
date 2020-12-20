import pandas as pd

def clean_data(year):
    filename = 'data/' + str(year) + '_leaders.xlsx'
    df = pd.read_excel(filename)
    if df.columns[0] == 'Unnamed: 0':
        #hasn't been cleaned yet. Just a sanity check
        #fix the names of the dictionary
        name_dict = {df.columns[ind] : df.iloc[0][ind] 
                     for ind in range(len(df.columns))}
        df = df.rename(columns=name_dict)
        #drop the redundant rows and column
        df = df.drop([0,1])
        df = df.loc[:, df.columns.notnull()]
        #Get rid of the sub-labels    
        df = df[df['Rk']!='Rk']
        #Let's just take the columns we really need
        df = df[['Rk', 'Player', 'Tm', 'FantPos', 
                 'Age', 'G', 'FantPt', 'PosRank']]
        #drop the rows that don't the neccessary data
        df = df.dropna()
        #finally, output the product
        writefilename = 'clean_' + filename
        df.to_excel(writefilename)

def clean_all_data(start_year = 2005, end_year = 2019):
    for year in range(start_year, end_year + 1):
        clean_data(year)

clean_all_data()
 