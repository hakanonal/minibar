import pandas as pd

def matrix_to_df(matrix):
    filenames = []
    labels = []
    for filename in matrix:
        filenames.append(filename)
        label = list(matrix[filename].keys())
        label.pop(0)
        labels.append(label)
    return pd.DataFrame(list(zip(filenames,labels)),columns=['filename','class'])

def matrix_to_dfcount(matrix):
    filenames = []
    counts = []
    for filename,val in matrix.items():
        filenames.append(filename)
        counts.append(val['count'])
    return pd.DataFrame(list(zip(filenames,counts)),columns=['filename','count'])
