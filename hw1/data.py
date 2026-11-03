"""Download and prepare the public Amazon review data."""

import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse


SOURCE = 'https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/'
SOURCES = {
    'reviews': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz',
    'metadata': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz',
}
SHA256 = {
    'reviews': 'e3ae8a3f0d70bf63242a49539a8847d30216a3aa2938c049c7f43d0473fef1bc',
    'metadata': '8e3f0dbe4c6c1aa5b32d350aa26072232beb6160ff3c5dadb44bca2c473a8b41',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_data(cache_dir='.cache/raw'):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in SOURCES.items():
        path = cache / url.rsplit('/', 1)[1]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            request = urllib.request.Request(url, headers={'User-Agent': 'recsys-coursework/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
            temporary.replace(path)
        digest = sha256(path)
        if name in SHA256 and digest != SHA256[name]:
            raise ValueError(f'Checksum mismatch: {path.name}')
        paths[name] = path
        print(f'{name}: {path.stat().st_size / 1024**2:.1f} MiB; SHA256 {digest}', flush=True)
    return paths


def clean_text(value, maximum=200):
    if not isinstance(value, str):
        return ''
    value = html.unescape(re.sub(r'<[^>]*>', ' ', value))
    return re.sub(r'\s+', ' ', value).strip()[:maximum]


def read_reviews(path):
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            rows.append((row.get('reviewerID'), row.get('asin'), row.get('overall'), row.get('unixReviewTime')))
    frame = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    diagnostics = {'raw_rows': len(frame), 'missing': frame.isna().sum().to_dict(),
                   'exact_duplicates': int(frame.duplicated().sum()),
                   'duplicate_pairs': int(frame.duplicated(['user_id', 'item_id']).sum())}
    frame = frame.dropna().copy()
    frame['rating'] = pd.to_numeric(frame['rating'], errors='coerce')
    frame['timestamp'] = pd.to_numeric(frame['timestamp'], errors='coerce')
    valid = frame['rating'].between(1, 5) & (frame['timestamp'] > 0)
    diagnostics['invalid_rows'] = int((~valid).sum())
    frame = frame[valid].sort_values(['timestamp', 'user_id', 'item_id'], kind='stable')
    frame = frame.drop_duplicates(['user_id', 'item_id'], keep='first').reset_index(drop=True)
    frame['timestamp'] = frame['timestamp'].astype('int64')
    diagnostics['deduplicated_rows'] = len(frame)
    return frame, diagnostics


def read_metadata(path, item_ids):
    wanted, rows, seen = set(item_ids), [], set()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            item_id = row.get('asin')
            if item_id not in wanted or item_id in seen:
                continue
            seen.add(item_id)
            categories = row.get('category') or row.get('categories') or []
            while categories and isinstance(categories[0], list):
                categories = categories[0]
            categories = [clean_text(c, 120) for c in categories if isinstance(c, str)]
            category = categories[-1] if categories else ''
            rows.append((item_id, clean_text(row.get('title')), clean_text(row.get('brand'), 120), category))
    return pd.DataFrame(rows, columns=['item_id', 'title', 'brand', 'category'])


def prepare_data(frame, metadata, max_items=6000, min_history=3):
    cuts = [int(frame.timestamp.quantile(q)) for q in (0.70, 0.85)]
    train = frame[frame.timestamp < cuts[0]].copy()
    counts = train.groupby('item_id').size().rename('count').reset_index()
    catalog = counts.sort_values(['count', 'item_id'], ascending=[False, True]).head(max_items).item_id
    train = train[train.item_id.isin(catalog)]
    while True:
        previous = len(train)
        users = train.groupby('user_id').size()
        items = train.groupby('item_id').size()
        train = train[train.user_id.isin(users[users >= min_history].index)
                      & train.item_id.isin(items[items >= min_history].index)]
        if len(train) == previous:
            break
    user_ids, item_ids = sorted(train.user_id.unique()), sorted(train.item_id.unique())
    selected = frame[frame.user_id.isin(user_ids) & frame.item_id.isin(item_ids)].copy()
    selected['u'] = selected.user_id.map({key: i for i, key in enumerate(user_ids)})
    selected['i'] = selected.item_id.map({key: i for i, key in enumerate(item_ids)})
    selected['split'] = np.select([selected.timestamp < cuts[0], selected.timestamp < cuts[1]],
                                  ['train', 'validation'], default='test')
    parts = {name: selected[selected.split == name].reset_index(drop=True)
             for name in ('train', 'validation', 'test')}
    items = metadata.set_index('item_id').reindex(item_ids).fillna('').reset_index()
    codes, vocabularies = {}, {}
    for field in ('brand', 'category'):
        items[field] = items[field].replace('', '[UNKNOWN]')
        vocabulary = sorted(items[field].unique())
        codes[field] = items[field].map({value: i for i, value in enumerate(vocabulary)}).to_numpy()
        vocabularies[field] = vocabulary
    assert 100_000 <= len(selected) <= 2_000_000
    assert len(user_ids) >= 1000 and len(item_ids) >= 1000
    assert parts['train'].timestamp.max() < parts['validation'].timestamp.min()
    assert parts['validation'].timestamp.max() < parts['test'].timestamp.min()
    return {'frame': selected, 'parts': parts, 'items': items, 'codes': codes,
            'vocabularies': vocabularies, 'shape': (len(user_ids), len(item_ids)), 'cuts': cuts,
            'user_ids': user_ids, 'item_ids': item_ids}


def interaction_matrix(frame, shape, positive_only=False, values=None):
    if positive_only:
        frame = frame[frame.rating >= 4]
    if values is None:
        values = np.ones(len(frame), dtype=np.float64)
    return sparse.csr_matrix((values, (frame.u.to_numpy(), frame.i.to_numpy())), shape=shape)


if __name__ == '__main__':
    download_data()
