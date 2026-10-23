"""Download the official gzip IDX files when Assignment3 data is unavailable."""
import json
from pathlib import Path
from urllib.request import urlretrieve

root=Path(__file__).resolve().parent
cfg=json.loads((root/'config/config.json').read_text())
destination=(root/cfg['data_dir']).resolve()
destination.mkdir(parents=True,exist_ok=True)
for filename in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']:
    path=destination/filename
    if not path.exists():
        urlretrieve('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+filename,path)
    print(path)
