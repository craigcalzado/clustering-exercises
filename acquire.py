import pandas as pd
import numpy as np
import os

from env import host, user, password

def get_zillow_data(use_cache=True):
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Let me get that for you...")
        # if Unnamed: 0 column exists, drop it
        df = pd.read_csv(filename)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        else:
            pass
        return df
    print("Sorry, nothing on file, let me create one for you...")
    data = 'zillow'
    url = f'mysql+pymysql://{user}:{password}@{host}/{data}'
    query = '''
            SELECT *
            FROM properties_2017 
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)
            JOIN predictions_2017 USING (parcelid)
            LEFT JOIN airconditioningtype USING (airconditioningtypeid)
            LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
            LEFT JOIN buildingclasstype USING (buildingclasstypeid)
            LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
            LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
            LEFT JOIN storytype USING (storytypeid)
            JOIN unique_properties USING (parcelid)
            WHERE transactiondate >= '2017-%%-%%';'''
    zillow17_data = pd.read_sql(query, url)
    zillow17_data.to_csv(filename)
    return zillow17_data