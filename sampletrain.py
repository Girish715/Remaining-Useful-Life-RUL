import pandas as pd

# Load train file
train_df = pd.read_csv('/content/dataset_folder/CMaps/train_FD001.txt',
                       sep='\s+', header=None)

# Preview
train_df.head()
