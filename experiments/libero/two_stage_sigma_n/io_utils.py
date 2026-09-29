import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_plan(path):
    plan=read(path)
    for row in plan['files']:
        assert sha(row['path'])==row['sha256'],row['path']
    root=Path(plan['source'])
    for row in read(root/'SOURCE_MANIFEST.json')['files']:
        assert sha(root/row['path'])==row['sha256'],row['path']
    assert plan['sigma_shift']==5 and plan['phase']=='A'
    assert plan['expected_measured_nfe']==77200 and plan['expected_warmup_nfe']==30
    return plan
