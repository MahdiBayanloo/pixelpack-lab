"""Small bounded reference container over a C++ transform + zlib backend.

Not JPEG/TIFF compatible: Pillow is used ONLY for importing/exporting pixels.
"""
import ctypes as C
import struct
import sys
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EXT = 'dylib' if sys.platform == 'darwin' else 'so'
lib = C.CDLL(str(ROOT / 'build' / ('libpixelpack.' + EXT)))
P = C.POINTER(C.c_uint8)
lib.pp_transform.argtypes = [P, P] + [C.c_size_t]*4 + [C.c_int]
lib.pp_compress.argtypes = [P,C.c_size_t,P,C.POINTER(C.c_size_t),C.c_int]
lib.pp_decompress.argtypes = [P,C.c_size_t,P,C.c_size_t]
lib.pp_zlib_version.restype = C.c_char_p
HEADER = struct.Struct('<4sBBBBIIQI')
LIMIT = 256*1024*1024
DTYPES = {1:np.dtype('u1'),2:np.dtype('<u2'),3:np.dtype('<f4')}

def _ptr(a): return a.ctypes.data_as(P)
def _check(rc):
    if rc: raise ValueError(f'pixelpack backend error {rc}')

def normalize(a):
    a = np.asarray(a)
    if a.ndim == 2: a = a[:,:,None]
    if a.ndim != 3 or not all(a.shape) or a.shape[2] > 4: raise ValueError('expected nonempty HWC, channels 1..4')
    if a.dtype.kind == 'u' and a.dtype.itemsize in (1,2): code = a.dtype.itemsize
    elif a.dtype.kind == 'f' and a.dtype.itemsize == 4: code = 3
    else: raise ValueError('supported: uint8, uint16, float32')
    if a.nbytes > LIMIT: raise ValueError('image exceeds 256 MiB')
    return np.ascontiguousarray(a,dtype=DTYPES[code]),code

def transform(a, inverse=False):
    a = np.asarray(a)
    if not a.flags.c_contiguous: raise ValueError('transform requires contiguous input')
    if a.ndim != 3 or a.dtype not in DTYPES.values(): raise ValueError('transform requires canonical HWC dtype')
    out = np.empty_like(a)
    h,w,c = a.shape
    _check(lib.pp_transform(_ptr(a),_ptr(out),w,h,c,a.dtype.itemsize,int(inverse)))
    return out

def compress(raw, level=6):
    src=np.frombuffer(raw,dtype=np.uint8)
    out=np.empty(len(raw)+len(raw)//1000+128,dtype=np.uint8)
    cap=C.c_size_t(out.size)
    _check(lib.pp_compress(_ptr(src),src.size,_ptr(out),C.byref(cap),level))
    return out[:cap.value].tobytes()

def pack(a, mode='adaptive', level=6):
    a,code=normalize(a)
    if mode not in ('raw','predict','adaptive'): raise ValueError('bad mode')
    raw=a.tobytes()
    candidates=[]
    if mode != 'predict': candidates.append((compress(raw,level),0))
    if mode != 'raw': candidates.append((compress(transform(a).tobytes(),level),1))
    payload,flag=min(candidates,key=lambda x:len(x[0]))
    h,w,c=a.shape
    return HEADER.pack(b'PPL1',1,code,c,flag,w,h,len(payload),zlib.crc32(raw))+payload

def unpack(blob):
    if len(blob)<HEADER.size: raise ValueError('truncated header')
    magic,ver,code,c,flag,w,h,n,crc=HEADER.unpack_from(blob)
    if magic!=b'PPL1' or ver!=1 or code not in DTYPES or not 1<=c<=4 or flag not in (0,1): raise ValueError('invalid header')
    size=w*h*c*DTYPES[code].itemsize
    if not w or not h or size>LIMIT: raise ValueError('invalid dimensions')
    if n!=len(blob)-HEADER.size or n>LIMIT+1048576: raise ValueError('payload length mismatch')
    src=np.frombuffer(blob,offset=HEADER.size,dtype=np.uint8)
    raw=np.empty(size,dtype=np.uint8)
    _check(lib.pp_decompress(_ptr(src),src.size,_ptr(raw),size))
    a=raw.view(DTYPES[code]).reshape(h,w,c)
    if flag: a=transform(a,inverse=True)
    if zlib.crc32(a.tobytes())!=crc: raise ValueError('checksum mismatch')
    return a

def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['pack','unpack'])
    ap.add_argument('input'); ap.add_argument('output')
    args=ap.parse_args()
    from PIL import Image
    if args.action=='pack':
        with Image.open(args.input) as im: a=np.asarray(im)
        Path(args.output).write_bytes(pack(a))
    else:
        a=unpack(Path(args.input).read_bytes())
        if a.shape[2]==1: a=a[:,:,0]
        Image.fromarray(a).save(args.output)
if __name__=='__main__': main()
