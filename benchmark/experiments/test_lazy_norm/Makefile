NVCC = /usr/local/cuda-12.8/bin/nvcc
CUDA_ARCH = -arch=sm_70
INCLUDES = -I../../../prototype/core
CXXFLAGS = -O3 -std=c++14
NVCCFLAGS = $(CUDA_ARCH) $(CXXFLAGS) $(INCLUDES) --expt-relaxed-constexpr

test_lazy_norm: test_lazy_norm.cu
	$(NVCC) $(NVCCFLAGS) -o test_lazy_norm test_lazy_norm.cu

clean:
	rm -f test_lazy_norm

.PHONY: clean
