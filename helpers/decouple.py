

def decouple(df):
    matrix = {}
    classes = {}
    for _, row in df.iterrows():
        if row['filename'] not in matrix:
            matrix[row['filename']] = {'count':0}
        if row['class'] not in matrix[row['filename']]:
            matrix[row['filename']][row['class']] = 1
        else:
            matrix[row['filename']][row['class']] += 1
        matrix[row['filename']]['count'] += 1

        if row['class'] not in classes:
            classes[row['class']] = 1
        else:
            classes[row['class']] += 1
    #classes = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1])}            
    return matrix,classes
