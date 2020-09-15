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

def matrix_to_dfclassed(matrix,classes):
    vals = []
    columns = ['filename'] + list(classes.keys())
    for filename,val in matrix.items():
        i = []
        i.append(filename)
        for c in classes.keys():
            if c in val:
                i.append(val[c])
            else:
                i.append(0)
        vals.append(tuple(i))
    return pd.DataFrame(vals,columns=columns)
