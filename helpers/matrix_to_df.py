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
