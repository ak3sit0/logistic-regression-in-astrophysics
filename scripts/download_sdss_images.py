#!/usr/bin/env python3
"""
Download SDSS JPEG cutouts for one example per class from a local CSV.

This script is lightweight and avoids retraining; useful to verify image-download
functionality before submission.
"""
import argparse
import pandas as pd
import numpy as np
import urllib.request
from pathlib import Path


def download_examples(csv_path='data/sdss_real_sample.csv', out_dir='results/sdss_examples'):
    df = pd.read_csv(csv_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if 'class' not in df.columns:
        raise ValueError('CSV must contain a "class" column with labels')

    class_names = sorted(df['class'].unique())
    for class_id, class_name in enumerate(class_names):
        print(f'Downloading example for {class_name}... ', end='')
        mask = df['class'] == class_name
        if mask.sum() == 0:
            print('no objects found')
            continue
        idx = int(np.random.choice(np.where(mask)[0]))
        row = df.iloc[idx]
        ra = row.get('ra')
        dec = row.get('dec')
        objid = row.get('objid') or row.get('specobjid')
        if pd.isna(ra) or pd.isna(dec):
            print('missing coordinates')
            continue

        url = f'https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.3&width=256&height=256'
        if pd.isna(objid):
            filename = f'{class_id:02d}_{class_name.lower()}_{idx}.jpg'
        else:
            filename = f'{class_id:02d}_{class_name.lower()}_objid_{objid}.jpg'

        out_path = Path(out_dir) / filename
        try:
            urllib.request.urlretrieve(url, str(out_path))
            print('saved')
        except Exception as e:
            print('error', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/sdss_real_sample.csv')
    parser.add_argument('--out', default='results/sdss_examples')
    args = parser.parse_args()
    download_examples(args.csv, args.out)
