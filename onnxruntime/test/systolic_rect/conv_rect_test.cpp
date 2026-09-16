// Standalone host command-decoder test, or real Gemmini test with -DRECT_RISCV.
#include "core/mlas/lib/systolic/conv_rect.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <stdexcept>
using systolic_rect::Shape;
#ifdef RECT_RISCV
bool SystolicConvRect(char, int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t, int64_t, int64_t, int64_t, int64_t,
                     const float*, const float*, const float*, float*, bool, float);
#endif
static void Require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
static uint64_t Bits(uint64_t x, int low, int count=16) {
  return (x >> low) & ((uint64_t(1) << count)-1);
}
// Decode the wire ABI independently of the tile scheduler, then execute each
// tile as scalar convolution. This validates geometry, padding, pointer strides,
// bias/zero initialization, channel accumulation and final-store boundaries.
struct Decoder {
  uint64_t r1[6]{}, r2[6]{};
  std::vector<float> acc;
  int calls=0;
  void operator()(int op, uint64_t a, uint64_t b) {
    if (op>=16 && op<=21) { r1[op-16]=a; r2[op-16]=b; return; }
    Require(op==15, "bad opcode"); ++calls;
    const int ih=Bits(r1[0],16), iw=Bits(r2[2],0);
    const int ci=Bits(r1[0],32), co=Bits(r1[0],48);
    const int oh=Bits(r2[0],0), ow=Bits(r2[0],32), stride=Bits(r2[0],48,8);
    const int kh=Bits(r1[1],48), rows=Bits(r2[1],32), cols=Bits(r2[1],16);
    const int och=Bits(r2[1],0), ich=Bits(r1[2],16);
    const int left=Bits(r1[2],0), right=Bits(r2[2],48), top=Bits(r2[2],32), bottom=Bits(r2[2],24,8);
    Require(ih>0 && iw>0 && oh>0 && ow>0, "zero shape");
    Require(Bits(r2[3],48)==uint64_t(ci) && Bits(r2[3],32)==uint64_t(co) && Bits(r2[3],16)==uint64_t(co), "bad tensor strides");
    Require(Bits(r1[3],48)==uint64_t(rows) && Bits(r2[3],0)==uint64_t(cols), "bad tile dimensions");
    Require(Bits(r1[2],48)==uint64_t(kh) && Bits(r1[2],32)==uint64_t(kh), "bad kernel");
    Require((b & 1) && !(b & 6) && Bits(a,8,8)==1, "unexpected pooling/downsample/packing");
    auto in=reinterpret_cast<const float*>(r2[5]);
    auto weights=reinterpret_cast<const float*>(r1[4]);
    auto out=reinterpret_cast<float*>(r2[4]);
    if (r1[5]) {
      acc.assign(rows*cols*och,0);
      if (!(a&1)) {
        auto bias=reinterpret_cast<const float*>(r1[5]);
        for (int p=0;p<rows*cols;++p) for(int oc=0;oc<och;++oc) acc[p*och+oc]=bias[oc];
      }
    }
    Require(acc.size()==size_t(rows*cols*och), "accumulator not initialized");
    const int valid_h=rows*stride+kh-1-top-bottom;
    const int valid_w=cols*stride+kh-1-left-right;
    for(int y=0;y<rows;++y) for(int x=0;x<cols;++x)
      for(int oc=0;oc<och;++oc) {
        float& sum=acc[(y*cols+x)*och+oc];
        for(int ky=0;ky<kh;++ky) for(int kx=0;kx<kh;++kx) {
          const int iy=y*stride+ky-top, ix=x*stride+kx-left;
          if(iy<0||ix<0||iy>=valid_h||ix>=valid_w) continue;
          for(int ic=0;ic<ich;++ic)
            sum+=in[(iy*iw+ix)*ci+ic]*weights[((ky*kh+kx)*ci+ic)*co+oc];
        }
        if(out) out[(y*ow+x)*co+oc]=((b>>3)&1)?std::max(0.0f,sum):sum;
      }
  }
};
static void Test(Shape s, bool bias_on, bool relu) {
  Require(systolic_rect::Supported(s),"shape unexpectedly rejected");
  std::vector<float> in(s.batch*s.ih*s.iw*s.ci), w(s.kernel*s.kernel*s.ci*s.co), bias(s.co);
  std::vector<float> out(s.batch*s.oh*s.ow*s.co,12345), ref(out.size());
  for(size_t i=0;i<in.size();++i) in[i]=(int(i%13)-6)*0.125f;
  for(size_t i=0;i<w.size();++i) w[i]=(int(i%7)-3)*0.0625f;
  for(size_t i=0;i<bias.size();++i) bias[i]=(int(i%5)-2)*0.25f;
  for(int n=0;n<s.batch;++n) for(int y=0;y<s.oh;++y) for(int x=0;x<s.ow;++x)
    for(int oc=0;oc<s.co;++oc) {
      float sum=bias_on?bias[oc]:0;
      for(int ky=0;ky<s.kernel;++ky) for(int kx=0;kx<s.kernel;++kx) {
        int iy=y*s.stride+ky-s.pad, ix=x*s.stride+kx-s.pad;
        if(iy<0||ix<0||iy>=s.ih||ix>=s.iw) continue;
        for(int ic=0;ic<s.ci;++ic)
          sum+=in[((n*s.ih+iy)*s.iw+ix)*s.ci+ic]*w[((ky*s.kernel+kx)*s.ci+ic)*s.co+oc];
      }
      ref[((n*s.oh+y)*s.ow+x)*s.co+oc]=relu?std::max(0.0f,sum):sum;
    }
#ifdef RECT_RISCV
  Require(SystolicConvRect(2,s.batch,s.ih,s.iw,s.ci,s.co,s.oh,s.ow,s.stride,s.pad,s.kernel,
                          in.data(),w.data(),bias_on?bias.data():nullptr,out.data(),relu,1.0f),"driver rejected");
#else
  Decoder d;
  const auto t=systolic_rect::SelectTile(s,16,4096,1024);
  Require(t.rows && systolic_rect::Fits(s,t,16,4096,1024),"tile overflow");
  systolic_rect::Run(s,t,in.data(),w.data(),bias_on?bias.data():nullptr,out.data(),relu,
                    [&](int f,uint64_t a,uint64_t b){d(f,a,b);});
#endif
  for(size_t i=0;i<out.size();++i) if(std::fabs(out[i]-ref[i])>1e-4f) {
    std::fprintf(stderr,"mismatch %zu got %g expected %g\n",i,out[i],ref[i]);std::exit(1);
  }
  std::printf("PASS %lldx%lld ci=%lld co=%lld k=%lld s=%lld bias=%d relu=%d\n",
    (long long)s.ih,(long long)s.iw,(long long)s.ci,(long long)s.co,(long long)s.kernel,(long long)s.stride,bias_on,relu);
}
int main() {
  for (auto dims : {std::array<int,6>{9,13,3,19,7,2}, {9,13,17,19,3,1},
                   {8,14,33,35,3,2}, {9,9,17,19,3,1}, {5,37,3,17,1,1},
                   {9,13,17,19,1,2}, {8,14,17,19,1,2}, {3,5,1,1,3,1},
                   {23,31,33,35,3,1}}) {
    const int h=dims[0],w=dims[1],k=dims[4],s=dims[5],p=k/2;
    Shape shape{2,h,w,dims[2],dims[3],(h+2*p-k)/s+1,(w+2*p-k)/s+1,k,s,p};
    Test(shape,true,false); Test(shape,false,true);
  }
  Require(!systolic_rect::Supported({1,9,13,3,16,9,12,3,1,1}),"bad output accepted");
  Require(!systolic_rect::Supported({1,9,70000,3,16,9,70000,3,1,1}),"ABI overflow accepted");
  std::puts("all 18 rectangular/square convolution cases passed");
}
