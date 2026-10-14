"""Download apipop from the CRAN `survey` package, pinned, and save data/apipop.pkl.

The version is pinned and the extracted api.rda is checked against its SHA-256, so the
application does not drift with new releases. The data are not redistributed here.
"""
import hashlib
import io
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import rdata

VERSION = '4.5'
SHA256 = 'ec72230e19bdcffd3bb531b4893ab05b524b30ffa84282e3db99e0546102de0b'
DATA = Path(__file__).resolve().parents[1] / 'data'
DATA.mkdir(exist_ok=True)
name = f'survey_{VERSION}.tar.gz'
for url in (f'https://cran.r-project.org/src/contrib/{name}',
            f'https://cran.r-project.org/src/contrib/Archive/survey/{name}'):
    try:
        raw = urllib.request.urlopen(url).read()
        break
    except urllib.error.HTTPError:
        continue
else:
    raise SystemExit(f'{name} not found on CRAN')
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    blob = tf.extractfile('survey/data/api.rda').read()
digest = hashlib.sha256(blob).hexdigest()
if SHA256 is not None and digest != SHA256:
    raise SystemExit(f'api.rda checksum mismatch: {digest}')
(DATA / 'api.rda').write_bytes(blob)
api = rdata.conversion.convert(rdata.parser.parse_file(DATA / 'api.rda'))
api['apipop'].to_pickle(DATA / 'apipop.pkl')
print(f'{name}: api.rda sha256 {digest}; apipop {api["apipop"].shape} -> data/apipop.pkl')
