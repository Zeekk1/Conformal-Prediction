

image_path = "/content/extracted_data/last/images-224-subset"
metadata_csv = r"/content/extracted_data/last/Order 8003922 Fina subset_metadata_30000.csv"

import pandas as pd
df = pd.read_csv(metadata_csv)
df.tail()

#Header converted from Patient Gender to Patient Sex
df = df[['Image Index', 'Finding Labels', 'Patient Age', 'Patient Sex']].dropna()
df = df[df['Finding Labels'] != 'No Finding']
df.head()


df['Primary_Label'] = df['Finding Labels'].apply(lambda x: x.split('|')[0])
df.head()


#then save the file to desired path and name
