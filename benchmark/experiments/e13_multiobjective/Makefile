NVCC = nvcc
CUDA_ARCH = -arch=sm_75
INCLUDES = -I../../../prototype/core
CXXFLAGS = -O3 -std=c++14
NVCCFLAGS = $(CUDA_ARCH) $(CXXFLAGS) $(INCLUDES) --expt-relaxed-constexpr

TARGET = e13_multiobjective
SRC = gpu.cu

all: $(TARGET)

$(TARGET): $(SRC) bi_objective_vrp.cuh tri_objective_vrp.cuh bi_objective_knapsack.cuh
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
