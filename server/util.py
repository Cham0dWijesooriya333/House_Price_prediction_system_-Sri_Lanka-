import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0 :
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_location_name():
    return __locations

def load_saved_artifacts():
     print("Loading saved artifacts...")
     global __data_columns
     global __locations

     with open("./artifacts/columns.json", 'r') as f:
         __data_columns = json.load(f)['data_columns']
         __locations = __data_columns[3:]

     global __model
     with open("./artifacts/sri_lanka_home_price_prediction_real_state.pkl", 'rb') as f:
            __model = pickle.load(f)
            print("Loaded Artifaccts>>>>>>....")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_name())

    print(get_estimated_price('badulla city', 30, 5,2))
    print(get_estimated_price('colombo 3', 30, 3,2))