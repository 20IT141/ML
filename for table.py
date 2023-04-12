import pandas as pd

data = {'Sr No': [1, 2],
        'CNN': ['CNN 1', 'CNN 2'],
        'Total Trainable Parameters': ['5', '5'],
        'Training Accuracy': ['6', '1'],
        'Testing Accuracy': ['5', '5']}

df = pd.DataFrame(data)
print(df)
