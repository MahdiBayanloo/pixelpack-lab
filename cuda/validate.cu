// Optional CPU/GPU transform-equivalence validator, not a GPU DEFLATE codec.
// This source MUST be compiled/run on NVIDIA hardware before claiming validation.
#include <cuda_runtime.h>
#include "../src/pixelpack.h"
#include <vector>
#include <random>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cstdlib>
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ uint32_t read_le(const uint8_t* p,size_t b){uint32_t x=0;for(size_t j=0;j<b;j++)x|=uint32_t(p[j])<<(8*j);return x;}
__global__ void forward(const uint8_t* in,uint8_t* out,size_t n,size_t stride,size_t b){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    uint32_t x=read_le(in+i*b,b),p=i>=stride?read_le(in+(i-stride)*b,b):0;
    uint32_t r=b==4?(x^p):(x-p);
    for(size_t j=0;j<b;j++)out[j*n+i]=uint8_t(r>>(8*j));
}
__global__ void inverse(const uint8_t* in,uint8_t* out,size_t h,size_t stride,size_t b){
    size_t col=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(col>=stride)return;
    size_t n=h*stride;uint32_t p=0;
    for(size_t row=0;row<h;row++){
        size_t i=row*stride+col;uint32_t r=0;
        for(size_t j=0;j<b;j++)r|=uint32_t(in[j*n+i])<<(8*j);
        uint32_t x=b==4?(r^p):(r+p);
        for(size_t j=0;j<b;j++)out[i*b+j]=uint8_t(x>>(8*j));
        p=x;
    }
}
int main(){try{
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
    std::cout<<"device="<<prop.name<<"\n";
    std::mt19937 rng(2026569);size_t cases=0;
    for(size_t b:{1,2,4})for(size_t c:{1,3,4})for(size_t w:{1,17,513})for(size_t h:{1,19,511}){
        size_t n=w*h*c,bytes=n*b;
        std::vector<uint8_t>a(bytes),ref(bytes),gpu(bytes),back(bytes);
        for(auto&x:a)x=rng();
        if(pp_transform(a.data(),ref.data(),w,h,c,b,0))throw std::runtime_error("CPU error");
        uint8_t *di=nullptr,*dt=nullptr,*dr=nullptr;
        ck(cudaMalloc(&di,bytes));ck(cudaMalloc(&dt,bytes));ck(cudaMalloc(&dr,bytes));
        cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
        ck(cudaMemcpy(di,a.data(),bytes,cudaMemcpyHostToDevice));
        // Warm-up and validate outside timed region.
        forward<<<(n+255)/256,256>>>(di,dt,n,w*c,b);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
        ck(cudaMemcpy(gpu.data(),dt,bytes,cudaMemcpyDeviceToHost));
        if(gpu!=ref)throw std::runtime_error("GPU forward mismatch");
        inverse<<<(w*c+255)/256,256>>>(dt,dr,h,w*c,b);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
        ck(cudaMemcpy(back.data(),dr,bytes,cudaMemcpyDeviceToHost));
        if(back!=a)throw std::runtime_error("GPU inverse mismatch");
        constexpr int repeats=20;
        ck(cudaEventRecord(start));
        for(int i=0;i<repeats;i++){forward<<<(n+255)/256,256>>>(di,dt,n,w*c,b);ck(cudaGetLastError());}
        ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms=0;ck(cudaEventElapsedTime(&ms,start,end));
        // Kernel-only latency, excluding allocation and host transfers. No end-to-end speedup claim.
        std::cout<<"b="<<b<<",c="<<c<<",w="<<w<<",h="<<h<<",forward_ms="<<ms/repeats<<"\n";
        ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
        ck(cudaFree(di));ck(cudaFree(dt));ck(cudaFree(dr));++cases;
    }
    std::cout<<"PASS "<<cases<<" CPU/GPU equivalence combinations\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
