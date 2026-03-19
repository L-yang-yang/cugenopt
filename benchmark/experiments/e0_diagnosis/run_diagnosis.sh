#!/bin/bash
# GenSolver 性能诊断 - 一键启动脚本
#
# 用法:
#   ./run_diagnosis.sh [host]           # 运行完整诊断（all 模式）
#   ./run_diagnosis.sh [host] profile   # 仅 nvprof profiling
#
# host: tc_new (T4) | tch (V100), 默认 tc_new

set -e

DIAG_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$DIAG_DIR")"
ROOT_DIR="$(dirname "$BENCH_DIR")"
RESULTS_DIR="$DIAG_DIR/results"

REMOTE_HOST="${1:-tc_new}"
MODE="${2:-all}"
REMOTE_DIR="~/gensolver"

echo ">>> 使用服务器: $REMOTE_HOST"

ARCH="sm_75"
if [ "$REMOTE_HOST" = "tch" ]; then
    ARCH="sm_70"
fi

NVCC_CMD="nvcc -arch=$ARCH -O2 -std=c++17 --extended-lambda -I ../../prototype/core -I ../../prototype/problems"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  GenSolver 性能诊断"
echo "  时间: $(date)"
echo "  服务器: $REMOTE_HOST (arch=$ARCH)"
echo "=========================================="

sync_code() {
    echo ">>> 同步代码到 $REMOTE_HOST ..."
    ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR/prototype/core $REMOTE_DIR/prototype/problems $REMOTE_DIR/benchmark/experiments/e0_diagnosis"
    scp "$ROOT_DIR"/prototype/core/*.cuh $REMOTE_HOST:$REMOTE_DIR/prototype/core/
    scp "$ROOT_DIR"/prototype/problems/*.cuh $REMOTE_HOST:$REMOTE_DIR/prototype/problems/
    scp "$DIAG_DIR"/bench_diagnosis.cu $REMOTE_HOST:$REMOTE_DIR/benchmark/experiments/e0_diagnosis/
    echo "    done."
}

compile() {
    echo ">>> 编译 bench_diagnosis (arch=$ARCH) ..."
    ssh $REMOTE_HOST "export PATH=/usr/local/cuda/bin:\$PATH && cd $REMOTE_DIR/benchmark/experiments/e0_diagnosis && $NVCC_CMD -o bench_diagnosis bench_diagnosis.cu 2>&1"
    echo "    done."
}

run_all() {
    echo ">>> 运行完整诊断 ..."
    local gpu_name=$(ssh $REMOTE_HOST "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1" | tr ' ' '_')
    local outfile="bench_${gpu_name}_$(date +%Y%m%d_%H%M%S).csv"

    ssh $REMOTE_HOST "export PATH=/usr/local/cuda/bin:\$PATH && cd $REMOTE_DIR/benchmark/experiments/e0_diagnosis && ./bench_diagnosis all 2>&1 >/tmp/diag_out.csv && cat /tmp/diag_out.csv" > "$RESULTS_DIR/$outfile"

    echo "    结果: $RESULTS_DIR/$outfile"
    local lines=$(wc -l < "$RESULTS_DIR/$outfile" 2>/dev/null || echo 0)
    echo "    数据行: $((lines - 1))"
}

run_profile() {
    echo ">>> 运行 nvprof profiling ..."
    echo "--- baseline (batch=2000, AOS=off) ---"
    ssh $REMOTE_HOST "export PATH=/usr/local/cuda/bin:\$PATH && cd $REMOTE_DIR/benchmark/experiments/e0_diagnosis && nvprof --print-gpu-summary ./bench_diagnosis baseline 2>&1" | tee "$RESULTS_DIR/nvprof_baseline_$REMOTE_HOST.txt"
    echo ""
    echo "--- default (batch=50, AOS=on) ---"
    ssh $REMOTE_HOST "export PATH=/usr/local/cuda/bin:\$PATH && cd $REMOTE_DIR/benchmark/experiments/e0_diagnosis && nvprof --print-gpu-summary ./bench_diagnosis default 2>&1" | tee "$RESULTS_DIR/nvprof_default_$REMOTE_HOST.txt"
}

sync_code
compile

case "$MODE" in
    all)     run_all ;;
    profile) run_profile ;;
    *)
        echo "未知模式: $MODE"
        echo "用法: ./run_diagnosis.sh [host] [all|profile]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  诊断完成"
echo "  服务器: $REMOTE_HOST"
echo "  结果目录: $RESULTS_DIR"
echo "=========================================="
ls -lh "$RESULTS_DIR"/ 2>/dev/null || true
