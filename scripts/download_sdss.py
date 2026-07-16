#!/usr/bin/env python3
"""
Download a small spectroscopic SDSS sample using astroquery and save as CSV

Writes: data/sdss_real_sample.csv

Usage: python scripts/download_sdss.py --rows 5000
"""
import argparse
from astroquery.sdss import SDSS
from astropy.table import Table
import pandas as pd
import os


def main(rows=5000, out_path='data/sdss_real_sample.csv'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sql = f"""
    SELECT TOP {rows}
      p.objid, p.ra, p.dec,
      p.u, p.g, p.r, p.i, p.z,
      p.err_u, p.err_g, p.err_r, p.err_i, p.err_z,
      s.specobjid, s.mjd, s.plate, s.fiberid,
      s.class as class
    FROM PhotoObjAll AS p
    JOIN SpecObjAll AS s ON p.objid = s.bestobjid
    WHERE s.class IN ('STAR','GALAXY','QSO')
    """

    print(f"Running SQL to fetch {rows} rows from SDSS...")
    tbl = SDSS.query_sql(sql)
    if tbl is None:
        raise RuntimeError('No data returned from SDSS. Try reducing rows or check connectivity.')

    df = Table(tbl).to_pandas()
    # Normalize column names to expected pipeline names
    if 'class' not in df.columns and 'spec_class' in df.columns:
        df = df.rename(columns={'spec_class': 'class'})

    # Ensure objid present for optional image downloads
    if 'objid' not in df.columns and 'specobjid' in df.columns:
        df['objid'] = df['specobjid']

    print(f"Fetched dataframe with shape: {df.shape}")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SDSS sample')
    parser.add_argument('--rows', type=int, default=5000, help='Number of rows to fetch')
    parser.add_argument('--out', type=str, default='data/sdss_real_sample.csv', help='Output CSV path')
    args = parser.parse_args()
    main(rows=args.rows, out_path=args.out)
