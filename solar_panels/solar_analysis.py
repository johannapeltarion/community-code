"""
Script to compare actual values in preprocessed dataset with predicted values

The data in the output will have the following format:
<actual value>,<predicted value>,<image path>
"""

import pandas as pd
import requests
import base64
import time
import json
import os
from fire import Fire


url = ''
token = ''

def get_input_csv(index_source_file_path):
    return pd.read_csv(index_source_file_path,
                       delimiter=',',
                       header = None,
                       names=["image", "prob", "type", "subset", "prob_binary"])

def get_prediction(rows):
    payload = "{\"rows\": ["
    for row in rows:
        payload = payload + row
    payload = payload + "]}"
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, data=payload, headers=headers, auth=(token, ''))
    print(response.text)
    data = json.loads(response.text)["rows"]
    return data

def append_csv(out_file_path, img_rows, act_vals, img_paths):
    out_file = open(out_file_path, 'a')
    x = len(img_rows)
    resp = get_prediction(img_rows)
    for n in range(0, x):
        out_file.write(str(act_vals[n]) + ',' +
                       str(resp[n].get("prob")) + ',' +
                       img_paths[n] + '\n')
    out_file.close()

def create_out_csv(df, request_batch_size, preprocess_dir, out_file_path):
    img_rows = list()
    img_paths = list()
    act_vals = list()
    row_count = 0
    batch_count = 0

    total_rows = df[df["subset"] == 'V'].count()[0]
    print('Rows in validation set: %s' % (total_rows))

    for index, row in df.iloc[1:].iterrows():
        if (row['subset'] == 'T'): # Skip training data
            continue
        # Read image and convert to Base54
        image_file_path = os.path.join(preprocess_dir, row['image'])
        img_paths.append(image_file_path)
        image_type = os.path.splitext(image_file_path)[-1][1:]
        with open(image_file_path, "rb") as image_file:
            encoded_image = 'data:image/{};base64,'.format(image_type) + base64.b64encode(image_file.read()).decode('ascii')

        # Actual values from index file
        act_vals.append(float(row['prob']))
        row_json = "{\"image\": \"" + encoded_image + "\"}"
        row_count += 1
        # Write string containing actual value and prediction at end of batch
        if (row_count % request_batch_size == 0) or (row_count >= total_rows):
            img_rows.append(row_json)
            append_csv(out_file_path, img_rows, act_vals, img_paths)
            img_rows.clear()
            act_vals.clear()
            batch_count += 1
            print('Batch count: ' + str(batch_count))
            print('Row count: ' + str(row_count))
        else:
            row_json = row_json + ','
            img_rows.append(row_json)

def run(root_dir, deploy_url, deploy_token):
    global url
    global token
    url = deploy_url
    token = deploy_token
    request_batch_size = 50
    preprocess_dir = os.path.join(root_dir, 'preprocessed')
    index_file_path = os.path.join(preprocess_dir, 'index.csv')
    out_file_path = os.path.join(root_dir, 'analysis-' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
    df = get_input_csv(index_file_path)
    create_out_csv(df, request_batch_size, preprocess_dir, out_file_path)
    print('Done.')

if __name__ == "__main__":
    Fire(run)

