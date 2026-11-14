#include "../src/pixelpack.h"
#include <vector>
#include <cassert>
#include <random>
#include <iostream>
int main(){
    std::mt19937 gen(2026569);
    size_t cases=0;
    for(size_t b:{1,2,4}) for(size_t c:{1,3,4}) for(size_t w:{1,17,129}) for(size_t h:{1,19,67}) {
        size_t n=b*c*w*h;
        std::vector<uint8_t> a(n),t(n),r(n),z(n+n/1000+128),u(n);
        for(auto& x:a) x=gen();
        assert(pp_transform(a.data(),t.data(),w,h,c,b,0)==0);
        assert(pp_transform(t.data(),r.data(),w,h,c,b,1)==0);
        assert(a==r);
        size_t cap=z.size();
        assert(pp_compress(t.data(),n,z.data(),&cap,6)==0);
        assert(pp_decompress(z.data(),cap,u.data(),n)==0);
        assert(t==u);
        assert(pp_decompress(z.data(),cap-1,u.data(),n)!=0);
        ++cases;
    }
    std::cout<<cases<<" native shape/type combinations passed\n";
}
