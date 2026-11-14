import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ctypes as C
import struct
import unittest
import zlib
import numpy as np
import pixelpack as pp

def oracle(a):
    """Independent NumPy prediction and byte shuffle, not a second C++ call."""
    b=a.dtype.itemsize
    u=a.view({1:np.uint8,2:np.dtype('<u2'),4:np.dtype('<u4')}[b])
    prev=np.zeros_like(u); prev[1:]=u[:-1]
    if b==4: residual=np.bitwise_xor(u,prev)
    else: residual=np.subtract(u,prev,dtype=u.dtype)
    return residual.view(np.uint8).reshape(-1,b).T.copy().tobytes()

class PixelPackTests(unittest.TestCase):
    def roundtrip(self,a):
        original,_=pp.normalize(a)
        actual=pp.transform(original)
        self.assertEqual(actual.tobytes(),oracle(original))
        self.assertEqual(pp.transform(actual,True).tobytes(),original.tobytes())
        for mode in ('raw','predict','adaptive'):
            blob=pp.pack(a,mode)
            result=pp.unpack(blob)
            self.assertEqual(result.shape,original.shape)
            self.assertEqual(result.dtype,original.dtype)
            self.assertEqual(result.tobytes(),original.tobytes())

    def test_randomized_oracle_and_roundtrip(self):
        rng=np.random.default_rng(2026569)
        for shape in [(1,1,1),(1,37,3),(29,1,4),(17,19,2),(257,259,3)]:
            for dt in ('u1','<u2','<f4'):
                with self.subTest(shape=shape,dtype=dt):
                    a=rng.integers(0,256,size=np.prod(shape)*np.dtype(dt).itemsize,dtype='u1').view(dt).reshape(shape)
                    self.roundtrip(a)
    def test_modular_wraparound(self):
        for dt in ('u1','<u2'):
            self.roundtrip(np.array([0,np.iinfo(dt).max,0,1],dtype=dt).reshape(4,1,1))
    def test_float_special_bit_patterns(self):
        bits=np.array([0x00000000,0x80000000,0x7f800000,0xff800000,0x7fc00001,0x7fc00123,0x00000001,0x7f7fffff],dtype='<u4')
        self.roundtrip(bits.view('<f4').reshape(4,2,1))
    def test_constants_and_gradients(self):
        for b in (1,2):
            self.roundtrip(np.zeros((31,29,4),dtype=f'<u{b}'))
            self.roundtrip(np.arange(31*29*3,dtype=f'<u{b}').reshape(31,29,3))
    def test_noncontiguous_and_big_endian(self):
        a=np.arange(500,dtype='>u2').reshape(20,25)
        self.roundtrip(a[::2,::-1])
    def test_every_header_truncation(self):
        blob=pp.pack(np.zeros((8,8),dtype='u1'))
        for k in range(pp.HEADER.size):
            with self.assertRaises(ValueError): pp.unpack(blob[:k])
    def test_invalid_header_fields(self):
        blob=pp.pack(np.ones((8,9),dtype='u1'))
        fields=list(pp.HEADER.unpack_from(blob))
        for idx,value in [(0,b'EVIL'),(1,9),(2,9),(3,0),(3,5),(4,2),(5,0),(6,0),(5,0xffffffff),(7,0xffffffff)]:
            bad=fields.copy(); bad[idx]=value
            with self.subTest(idx=idx,value=value),self.assertRaises(ValueError):
                pp.unpack(pp.HEADER.pack(*bad)+blob[pp.HEADER.size:])
    def test_payload_corruption_and_truncation(self):
        blob=pp.pack(np.arange(1024,dtype='<u2').reshape(32,32))
        for k in [pp.HEADER.size,pp.HEADER.size+3,len(blob)-1]:
            bad=bytearray(blob); bad[k]^=0x5a
            with self.assertRaises(ValueError): pp.unpack(bytes(bad))
        with self.assertRaises(ValueError): pp.unpack(blob[:-1])
    def test_crc_mismatch(self):
        blob=bytearray(pp.pack(np.ones((4,4),dtype='u1')))
        blob[pp.HEADER.size-1]^=1
        with self.assertRaisesRegex(ValueError,'checksum'): pp.unpack(bytes(blob))
    def test_trailing_bytes_rejected(self):
        blob=pp.pack(np.ones((4,4),dtype='u1'))
        with self.assertRaises(ValueError): pp.unpack(blob+b'x')
        fields=list(pp.HEADER.unpack_from(blob)); fields[7]+=1
        with self.assertRaises(ValueError): pp.unpack(pp.HEADER.pack(*fields)+blob[pp.HEADER.size:]+b'x')
    def test_adaptive_never_larger_than_raw_candidate(self):
        rng=np.random.default_rng(8)
        for _ in range(10):
            a=rng.integers(0,256,(64,64,3),dtype='u1')
            self.assertLessEqual(len(pp.pack(a)),len(pp.pack(a,'raw')))
    def test_invalid_arrays(self):
        for a in [np.array([],dtype='u1'),np.zeros((2,2,5),dtype='u1'),np.zeros((3,3),dtype='f8'),np.zeros((2,2),dtype='i2')]:
            with self.assertRaises(ValueError): pp.pack(a)
    def test_cross_zlib_decode(self):
        a=np.arange(900,dtype='<u2').reshape(30,30,1)
        for mode in ('raw','predict'):
            blob=pp.pack(a,mode)
            expected=a.tobytes() if mode=='raw' else oracle(a)
            self.assertEqual(zlib.decompress(blob[pp.HEADER.size:]),expected)
    def test_cpp_argument_guards(self):
        a=np.ones((3,3,1),dtype='u1'); out=np.empty_like(a)
        self.assertEqual(pp.lib.pp_transform(pp._ptr(a),pp._ptr(out),0,3,1,1,0),-1)
        self.assertEqual(pp.lib.pp_transform(pp._ptr(a),pp._ptr(a),3,3,1,1,0),-1)
        self.assertEqual(pp.lib.pp_transform(pp._ptr(a),pp._ptr(out),3,3,1,3,0),-1)
        self.assertEqual(pp.lib.pp_transform(pp._ptr(a),pp._ptr(out),3,3,1,1,2),-1)
    def test_png_tiff_import_export(self):
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            for ext,dt in [('png','u1'),('tiff','<u2')]:
                a=np.arange(256,dtype=dt).reshape(16,16)
                src=Path(folder)/('source.'+ext); dst=Path(folder)/('decoded.'+ext)
                Image.fromarray(a).save(src)
                with Image.open(src) as im: restored=pp.unpack(pp.pack(np.asarray(im)))[:,:,0]
                Image.fromarray(restored).save(dst)
                with Image.open(dst) as im: np.testing.assert_array_equal(np.asarray(im),a)
if __name__=='__main__': unittest.main()
