# GenSolver Makefile
#
# 用法:
#   make e1 e2 e3 e4 e5 e6   → 编译单个实验
#   make diag                  → 编译诊断程序
#   make all                   → 编译全部
#   make clean                 → 清理

NVCC     = nvcc
ARCH     ?= -arch=sm_75
CFLAGS   = -O2 -std=c++17 --extended-lambda
INCLUDES = -I core -I problems -I ../benchmark/common

CORE_HEADERS = $(wildcard core/*.cuh)
PROB_HEADERS = $(wildcard problems/*.cuh)
COMMON_HEADERS = $(wildcard ../benchmark/common/*.cuh)
ALL_HEADERS  = $(CORE_HEADERS) $(PROB_HEADERS) $(COMMON_HEADERS)

BENCH_DIR = ../benchmark
EXP_DIR   = $(BENCH_DIR)/experiments
EXPERIMENTS = e0_diagnosis e1_vs_mip e2_vs_routing e2.1_custom_routing e3_ablation e4_scalability e5_generality e6_gpu_hardware e8_p2_search_strategy opt_init_solution

.PHONY: all clean diag test_multi_gpu test_multi_gpu_b3 $(patsubst %,e%,0 1 2 2.1 3 4 5 6 8)

all: e0 e1 e2 e2.1 e3 e4 e5 e6 e8 test_multi_gpu test_multi_gpu_b3

e0 diag: $(EXP_DIR)/e0_diagnosis/bench_diagnosis
e1: $(EXP_DIR)/e1_vs_mip/gpu
e2: $(EXP_DIR)/e2_vs_routing/gpu
e2.1: $(EXP_DIR)/e2.1_custom_routing/gpu
e3: $(EXP_DIR)/e3_ablation/gpu
e4: $(EXP_DIR)/e4_scalability/gpu
e5: $(EXP_DIR)/e5_generality/gpu
e6: $(EXP_DIR)/e6_gpu_hardware/gpu
e8: $(EXP_DIR)/e8_p2_search_strategy/gpu

$(EXP_DIR)/%/gpu: $(EXP_DIR)/%/gpu.cu $(ALL_HEADERS) problems/tsplib_data.h
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o $@ $<

$(EXP_DIR)/e0_diagnosis/bench_diagnosis: $(EXP_DIR)/e0_diagnosis/bench_diagnosis.cu $(ALL_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o $@ $<

test_multi_gpu: test_multi_gpu.cu $(ALL_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o $@ $<

test_multi_gpu_b3: test_multi_gpu_b3.cu $(ALL_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(foreach e,$(EXPERIMENTS),$(EXP_DIR)/$(e)/gpu) $(EXP_DIR)/e0_diagnosis/bench_diagnosis test_multi_gpu test_multi_gpu_b3
	@echo "Cleaned all experiment binaries."
