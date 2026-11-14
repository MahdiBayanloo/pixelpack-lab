"""Reproduce all CPU numbers; no GPU timings are manufactured."""
import sys,os,json,platform,subprocess,time,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pixelpack as pp

def datasets():
    rng=np.random.default_rng(2026569)
    y,x=np.mgrid[:512,:513]
    yield 'u8_constant',np.full((512,513),128,dtype='u1'),'synthetic'
    yield 'u8_horizontal_ramp',(x%256).astype('u1'),'synthetic'
    yield 'u8_vertical_ramp',(y%256).astype('u1'),'synthetic'
    yield 'u8_rgb_gradient',np.stack((x%256,y%256,(x+y)%256),axis=-1).astype('u1'),'synthetic'
    yield 'u8_random',rng.integers(0,256,(512,513,3),dtype='u1'),'synthetic negative control'
    yield 'u16_smooth_field',((x*73+y*31)%65536).astype('<u2'),'synthetic'
    yield 'u16_noisy_field',((x*73+y*31+rng.integers(0,8,x.shape))%65536).astype('<u2'),'synthetic'
    yield 'u16_random',rng.integers(0,65536,(512,513),dtype='<u2'),'synthetic negative control'
    yield 'f32_smooth_field',(np.sin(x/63)+np.cos(y/57)).astype('<f4'),'synthetic'
    yield 'f32_random_bits',rng.integers(0,2**32,(512,513),dtype='<u4').view('<f4'),'synthetic negative control'
    # scikit-learn bundled UCI optical digits: NOT MNIST, 1797 images, 8x8 each.
    from sklearn.datasets import load_digits
    images=load_digits().images.astype('u1')
    sheet=np.zeros((43*8,42*8),dtype='u1')
    for i,a in enumerate(images):
        row,col=divmod(i,42);sheet[row*8:(row+1)*8,col*8:(col+1)*8]=a
    yield 'uci_optical_digits_sheet',sheet,'UCI optical digits via sklearn; all 1797 images, padded mosaic'

def main():
    repeats=7;rows=[]
    for name,a,origin in datasets():
        canonical,_=pp.normalize(a); raw=canonical.tobytes()
        item={'name':name,'shape':list(canonical.shape),'dtype':str(canonical.dtype),'origin':origin,'input_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'modes':{}}
        for mode in ('raw','predict','adaptive'):
            blob=pp.pack(a,mode);assert pp.unpack(blob).tobytes()==raw
            enc=[];dec=[]
            for _ in range(repeats):
                start=time.perf_counter_ns();blob=pp.pack(a,mode);enc.append((time.perf_counter_ns()-start)/1e6)
                start=time.perf_counter_ns();out=pp.unpack(blob);dec.append((time.perf_counter_ns()-start)/1e6)
                assert out.tobytes()==raw
            item['modes'][mode]={'container_bytes':len(blob),'payload_bytes':len(blob)-pp.HEADER.size,'chosen_predictor':bool(blob[7]),'encode_ms_median':statistics.median(enc),'encode_ms_min':min(enc),'encode_ms_max':max(enc),'decode_ms_median':statistics.median(dec),'encode_input_MiB_s':len(raw)/(2**20)/(statistics.median(enc)/1000)}
        rows.append(item)
    command_file=ROOT/'build/build_command.txt'
    env={'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),'numpy':np.__version__,'zlib':pp.lib.pp_zlib_version().decode(),'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],'omp_num_threads':os.getenv('OMP_NUM_THREADS','unset'),'build_command':command_file.read_text().strip() if command_file.exists() else 'unknown: rebuild with current Makefile','repeats':repeats,'timer':'perf_counter_ns; medians, seven repetitions after one warm-up','seed':2026569,'cuda':'NOT compiled or run: no nvcc / NVIDIA GPU in authoring environment','scope':'local execution environment; hardware was not reserved exclusively','implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/pixelpack.cpp',ROOT/'src/pixelpack.h',ROOT/'pixelpack.py']}}
    result={'environment':env,'datasets':rows}
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'cpu_results.json').write_text(json.dumps(result,indent=2))
    lines=['# Measured CPU benchmark','', 'Date: 14 November 2026. Re-run `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/benchmark.py`.','',
           'All figures below were measured on this run. Full distributions (min/median/max encode times), dataset hashes and environment are in `cpu_results.json`. Each mode is warmed once and repeated seven times. All decodes are compared byte-for-byte.','',
           'Baseline = the same container and zlib level 6 without the predictor. Adaptive tries both representations and keeps the smaller; its encoding time includes BOTH compression passes. Savings include the identical container header. This is not a comparison against PNG, TIFF, nvCOMP or nvTIFF.','',
           '| Dataset | Input bytes | Baseline bytes | Adaptive bytes | Size reduction | Baseline encode ms | Adaptive encode ms | Adaptive decode ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    totalraw=totaladaptive=0
    for r in rows:
        base=r['modes']['raw'];a=r['modes']['adaptive'];totalraw+=base['container_bytes'];totaladaptive+=a['container_bytes']
        lines.append(f"| {r['name']} | {r['input_bytes']} | {base['container_bytes']} | {a['container_bytes']} | {100*(1-a['container_bytes']/base['container_bytes']):.2f}% | {base['encode_ms_median']:.3f} | {a['encode_ms_median']:.3f} | {a['decode_ms_median']:.3f} |")
    lines+=['',f'Across this fixed mixed suite: {totalraw:,} baseline bytes versus {totaladaptive:,} adaptive bytes ({100*(1-totaladaptive/totalraw):.2f}% smaller). This aggregate is strongly dataset-dependent; it is not a production compression claim.', '',
            '## Interpretation','', 'Vertical prediction exploits similarity between adjacent rows. Shuffling groups residual byte positions before DEFLATE. Uniform random inputs act as negative controls: prediction has no dependable compression benefit. The adaptive selector prevents a larger payload than the raw candidate, but often makes encoding slower. The report retains these costs and all negative controls.', '',
            '32-bit samples use bitwise XOR, not floating-point arithmetic. Consequently signed zero, infinity, subnormal values and NaN payload bits can round-trip exactly. This is bit-preservation evidence, not floating-point numerical-error analysis.', '',
            '## Validation and limitations','', '15 Python unittest methods (including parameterized subcases); independent NumPy transform oracle; PNG/TIFF pixel import/export; malformed headers, truncation, trailing bytes and CRC mismatch. Separate native ASan/UBSan execution covers 81 shape/type combinations.', '',
            'CUDA source provides forward and inverse transforms plus a CPU/GPU comparison harness. It has NOT been compiled or executed in this environment. No GPU performance, correctness or speedup result is claimed. DEFLATE remains CPU zlib; there is no GPU entropy coder. The decoder is bounded to 256 MiB; this is a research prototype, not an audited untrusted-input production library.', '',
            '## Environment','', '```json',json.dumps(env,indent=2),'```']
    (out/'BENCHMARK.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:25]))
if __name__=='__main__':main()
