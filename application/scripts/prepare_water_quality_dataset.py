
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def infer_freq_alias(minutes: int | None, fallback: str = 't') -> str:
    if minutes is None:
        return fallback
    if minutes % 1440 == 0:
        return 'd'
    if minutes % 60 == 0:
        return 'h'
    return 't'


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a water-quality CSV for iTransformer/PF-ABC.')
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--time_col', default='date')
    parser.add_argument('--target_col', default=None, help='Primary target column name; defaults to the last numeric column.')
    parser.add_argument('--freq_minutes', type=int, default=None, help='Sampling interval in minutes; if omitted, kept as generic minute-level freq t.')
    parser.add_argument('--key_channels', nargs='*', default=[], help='Key water-quality variables for reporting.')
    parser.add_argument('--drop_cols', nargs='*', default=[])
    parser.add_argument('--drop_quality_cols', action='store_true', help='Drop columns whose names end with " quality".')
    parser.add_argument('--keep_value_cols', action='store_true', help='Keep only numeric sensor columns whose names end with " value".')
    parser.add_argument('--fill_method', choices=['interpolate', 'dropna'], default='interpolate')
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.time_col not in df.columns:
        raise KeyError(f'time column not found: {args.time_col}')

    df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors='coerce')
    df = df.dropna(subset=[args.time_col])
    df[args.time_col] = df[args.time_col].dt.tz_convert(None)
    df = df.sort_values(args.time_col).reset_index(drop=True)
    if args.drop_cols:
        keep = [c for c in df.columns if c not in set(args.drop_cols)]
        df = df[keep]
    if args.drop_quality_cols:
        df = df[[c for c in df.columns if c == args.time_col or not c.endswith(' quality')]]

    numeric_cols = [c for c in df.columns if c != args.time_col and pd.api.types.is_numeric_dtype(df[c])]
    if args.keep_value_cols:
        numeric_cols = [c for c in numeric_cols if c.endswith(' value')]
    if not numeric_cols:
        raise ValueError('No numeric feature columns found.')

    target_col = args.target_col or numeric_cols[-1]
    if target_col not in numeric_cols:
        raise KeyError(f'target column must be numeric and present: {target_col}')

    feature_cols = [c for c in numeric_cols if c != target_col] + [target_col]
    clean = df[[args.time_col] + feature_cols].copy()
    clean = clean.rename(columns={args.time_col: 'date'})
    if args.freq_minutes:
        clean = clean.set_index('date').resample(f'{args.freq_minutes}min').mean()
        clean = clean.reset_index()
    if args.fill_method == 'interpolate':
        clean[feature_cols] = clean[feature_cols].interpolate(method='linear', limit_direction='both')
    clean = clean.dropna().reset_index(drop=True)

    csv_name = 'water_quality_clean.csv'
    clean.to_csv(output_dir / csv_name, index=False)

    root_path = str(output_dir).replace('\\', '/')
    if not root_path.endswith('/'):
        root_path += '/'

    manifest = {
        'root_path': root_path,
        'data_path': csv_name,
        'data': 'custom',
        'features': 'M',
        'target': target_col,
        'freq': infer_freq_alias(args.freq_minutes),
        'enc_in': len(feature_cols),
        'dec_in': len(feature_cols),
        'c_out': len(feature_cols),
        'feature_columns': feature_cols,
        'key_channels': args.key_channels,
        'time_col': 'date',
        'n_rows': int(len(clean)),
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
