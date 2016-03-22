import re
import numpy as np
from sklearn import preprocessing

'''
	preprocess
	
	Generates a file containing the processed data (numerical values, 
	feature scaling, normalization)
	
	version: 0.1
	authors: Aurelien Hontabat
	license: MIT
'''

def run():
    file = open('../data/german.data.txt', 'r')
    data = []
    
    # for making all values numerical
    le = preprocessing.LabelEncoder()
    
    # load in memory
    for line in file:
        element = re.sub("[^\w]", " ",  line).split()
        data.append(element)
        
    data = np.array(data)
    
    results = data[:,20]
    data = np.delete(data, 20, 1)
    
    res = []
    
    for result in results:
        if result == "2":
            res.append([0,1.5])
        else:
            res.append([1,0])
            
    results = np.array(res)

    # make numerical
    # generate labels
    labels = []
    for element in data:
        for field in element:
            try:
                float(field)
            except ValueError:
                labels.append(field)
    
    le.fit(np.array(labels))
    
    # replace labels
    for index, field in enumerate(data[0]):
        try:
            float(field)
        except ValueError:
            data[:,index] = le.transform(data[:,index])
    
    
    # scale the data
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(data)
    
    
    return scaled_data, results #scaled_result
    