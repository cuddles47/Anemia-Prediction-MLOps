import pandas as pd
import numpy as np


def get_unique_values(df: pd.DataFrame): # Chỗ này sửa để chỉ pass dataframe với cột m cần
    unique_features = set()
    
    for features in df.dropna():
        # Remove brackets and extra quotes
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "")
        # Split by comma
        feature_list = [f.strip() for f in cleaned_features.split(",")]
        unique_features.update(feature_list)

    return unique_features

def check_keywords(features, keywords):

    try:
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "").replace(";", ",")
        list_to_check = [f.strip().lower() for f in cleaned_features.split(',')]
        common = set(list_to_check) & set(keywords)
        
        if common: 
            return 'Yes'  
            
    except AttributeError:
        return 'No'
    
    return 'No'

def check_type(features, types):

    try:
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "").replace(";", ",")
        list_to_check = [f.strip().lower() for f in cleaned_features.split(',')]

        for basement_type, basement_type_list in types.items():
            if set(list_to_check) & set(basement_type_list): 
                return basement_type
        return np.nan

    except AttributeError:
        return np.nan
    
def check_view(features, keywords):

    try:
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "").replace(";", ",")
        list_to_check = [f.strip().lower() for f in cleaned_features.split(',')]
        common = set(list_to_check) & set(keywords)
        
        if common: 
            return list(common)[0]  
            
    except AttributeError:
        return np.nan
    
    return np.nan