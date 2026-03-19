#!/bin/bash
# GenSolver 统一实验入口
#
# 用法:
#   ./run.sh e1 [host]          → E1: vs MIP (GPU + Python)
#   ./run.sh e2 [host]          → E2: vs Routing (GPU + Python)
#   ./run.sh e2.1 [host]        → E2.1: Custom Routing - Priority VRP
#   ./run.sh e3 [host]          → E3: Ablation (GPU only)
#   ./run.sh e4 [host]          → E4: Scalability (GPU only)
#   ./run.sh e5 [host]          → E5: Generality (GPU + Python)
#   ./run.sh e6 [host]          → E6: GPU Hardware (GPU only)
#   ./run.sh diag [host]        → 性能诊断
#   ./run.sh status [host]      → 查看远程任务状态
#   ./run.sh clean [host]       → 远程清理
#
# host: tc_new (T4, 默认) | tch (V100)

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_DIR="$ROOT_DIR/prototype"
BENCH_DIR="$ROOT_DIR/benchmark"
EXP_DIR="$BENCH_DIR/experiments"
DIAG_DIR="$EXP_DIR/e0_diagnosis"
COMMON_DIR="$BENCH_DIR/common"
RESULTS_DIR="$BENCH_DIR/results"

REMOTE_DIR="~/gensolver"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

get_host() { echo "${1:-tc_new}"; }
get_arch() {
    local host="$1"
    case "$host" in
        tch)  echo "sm_70" ;;   # V100
        a800) echo "sm_80" ;;   # A800
        *)    echo "sm_75" ;;   # T4 (tc_new)
    esac
}

NVCC_BASE="nvcc -O2 -std=c++17 --extended-lambda"

# ─────────────────────────────────────────────
# 同步代码到远程
# ─────────────────────────────────────────────
sync_to_remote() {
    local host="$1"
    echo ">>> 同步代码到 $host ..."
    ssh "$host" "mkdir -p $REMOTE_DIR/prototype/core $REMOTE_DIR/prototype/problems \
        $REMOTE_DIR/benchmark/common $REMOTE_DIR/benchmark/experiments/e0_diagnosis \
        $REMOTE_DIR/benchmark/experiments/e1_vs_mip \
        $REMOTE_DIR/benchmark/experiments/e2_vs_routing \
        $REMOTE_DIR/benchmark/experiments/e2.1_custom_routing \
        $REMOTE_DIR/benchmark/experiments/e3_ablation \
        $REMOTE_DIR/benchmark/experiments/e4_scalability \
        $REMOTE_DIR/benchmark/experiments/e5_generality \
        $REMOTE_DIR/benchmark/experiments/e6_gpu_hardware \
        $REMOTE_DIR/benchmark/experiments/e7_medium_scale \
        $REMOTE_DIR/benchmark/data/solomon \
        $REMOTE_DIR/benchmark/data/qaplib \
        $REMOTE_DIR/benchmark/data/jsp \
        $REMOTE_DIR/benchmark/data/knapsack"
    scp "$PROTO_DIR"/core/*.cuh "$host":$REMOTE_DIR/prototype/core/ 2>/dev/null
    scp "$PROTO_DIR"/problems/*.cuh "$PROTO_DIR"/problems/*.h "$host":$REMOTE_DIR/prototype/problems/ 2>/dev/null
    scp "$COMMON_DIR"/*.cuh "$host":$REMOTE_DIR/benchmark/common/ 2>/dev/null
    scp "$DIAG_DIR"/bench_diagnosis.cu "$host":$REMOTE_DIR/benchmark/experiments/e0_diagnosis/ 2>/dev/null
    for exp in e1_vs_mip e2_vs_routing e2.1_custom_routing e3_ablation e4_scalability e5_generality e6_gpu_hardware e7_medium_scale; do
        scp "$EXP_DIR/$exp"/gpu.cu "$host":$REMOTE_DIR/benchmark/experiments/$exp/ 2>/dev/null
    done
    # E7 benchmark data
    for ddir in solomon qaplib jsp knapsack; do
        if [ -d "$BENCH_DIR/data/$ddir" ]; then
            scp "$BENCH_DIR/data/$ddir"/* "$host":$REMOTE_DIR/benchmark/data/$ddir/ 2>/dev/null
        fi
    done
    echo "    done."
}

# ─────────────────────────────────────────────
# 编译远程 GPU 程序
# ─────────────────────────────────────────────
compile_remote() {
    local host="$1"
    local arch="$2"
    local exp_name="$3"
    local src_dir="$REMOTE_DIR/benchmark/experiments/$exp_name"

    echo ">>> 编译 $exp_name/gpu (arch=$arch) ..."
    ssh "$host" "export PATH=/usr/local/cuda/bin:\$PATH && cd $src_dir && \
        $NVCC_BASE -arch=$arch \
        -I $REMOTE_DIR/prototype/core \
        -I $REMOTE_DIR/prototype/problems \
        -I $REMOTE_DIR/benchmark/common \
        -o gpu gpu.cu 2>&1"
}

# ─────────────────────────────────────────────
# 运行单个实验（GPU 侧）
# ─────────────────────────────────────────────
run_gpu() {
    local exp_name="$1"
    local host="$2"
    local arch="$3"
    local args="${4:-}"

    sync_to_remote "$host"
    compile_remote "$host" "$arch" "$exp_name"

    local outfile="${exp_name}_gpu_${host}_${TIMESTAMP}.csv"
    local logfile="${exp_name}_gpu_${host}_${TIMESTAMP}.log"
    local src_dir="$REMOTE_DIR/benchmark/experiments/$exp_name"

    echo ">>> 运行 $exp_name/gpu on $host ..."
    ssh "$host" "export PATH=/usr/local/cuda/bin:\$PATH && cd $src_dir && \
        ./gpu $args 2>/tmp/exp_log.txt" > "$RESULTS_DIR/$outfile"
    ssh "$host" "cat /tmp/exp_log.txt" > "$RESULTS_DIR/$logfile" 2>/dev/null

    echo "    CSV: $RESULTS_DIR/$outfile"
    local lines=$(wc -l < "$RESULTS_DIR/$outfile" 2>/dev/null || echo 0)
    echo "    数据行: $((lines - 1))"
}

# ─────────────────────────────────────────────
# 运行 Python 侧
# ─────────────────────────────────────────────
CONDA_ENV="ph-mp-model"
CONDA_BASE="$HOME/miniforge3"

run_python() {
    local script="$1"
    local name="$2"
    local outfile="${name}_${TIMESTAMP}.csv"
    local logfile="${name}_${TIMESTAMP}.log"

    if [ ! -f "$script" ]; then
        echo "    跳过 $name（$script 不存在）"
        return
    fi

    echo ">>> 运行 $name ..."
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    python "$script" > "$RESULTS_DIR/$outfile" 2>"$RESULTS_DIR/$logfile"
    local lines=$(wc -l < "$RESULTS_DIR/$outfile" 2>/dev/null || echo 0)
    echo "    CSV: $RESULTS_DIR/$outfile ($((lines - 1)) 行)"
}

# ─────────────────────────────────────────────
# 各实验入口
# ─────────────────────────────────────────────
run_e1() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E1: vs MIP =========="
    run_gpu "e1_vs_mip" "$host" "$arch"
    run_python "$EXP_DIR/e1_vs_mip/mip.py" "e1_mip"
}

run_e2() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E2: vs Routing =========="
    run_gpu "e2_vs_routing" "$host" "$arch"
    run_python "$EXP_DIR/e2_vs_routing/routing.py" "e2_routing"
}

run_e2_1() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E2.1: Custom Routing (Priority VRP) =========="
    run_gpu "e2.1_custom_routing" "$host" "$arch"
    run_python "$EXP_DIR/e2.1_custom_routing/routing_baseline.py" "e2.1_routing"
}

run_e3() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E3: Ablation =========="
    run_gpu "e3_ablation" "$host" "$arch"
}

run_e4() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E4: Scalability =========="
    run_gpu "e4_scalability" "$host" "$arch"
}

run_e5() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E5: Generality =========="
    run_gpu "e5_generality" "$host" "$arch"
    run_python "$EXP_DIR/e5_generality/cpsat.py" "e5_cpsat"
}

run_e6() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E6: GPU Hardware =========="
    run_gpu "e6_gpu_hardware" "$host" "$arch"
}

run_e7() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")
    echo "========== E7: Medium Scale =========="
    run_gpu "e7_medium_scale" "$host" "$arch" "$REMOTE_DIR/benchmark/data"
}

run_diag() {
    local host=$(get_host "$1")
    local arch=$(get_arch "$host")

    sync_to_remote "$host"

    echo ">>> 编译 bench_diagnosis (arch=$arch) ..."
    ssh "$host" "export PATH=/usr/local/cuda/bin:\$PATH && cd $REMOTE_DIR/benchmark/experiments/e0_diagnosis && \
        $NVCC_BASE -arch=$arch \
        -I $REMOTE_DIR/prototype/core \
        -I $REMOTE_DIR/prototype/problems \
        -I $REMOTE_DIR/benchmark/common \
        -o bench_diagnosis bench_diagnosis.cu 2>&1"

    local gpu_name=$(ssh "$host" "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1" | tr ' ' '_')
    local outfile="diag_${gpu_name}_${TIMESTAMP}.csv"

    echo ">>> 运行 bench_diagnosis on $host ..."
    ssh "$host" "export PATH=/usr/local/cuda/bin:\$PATH && cd $REMOTE_DIR/benchmark/experiments/e0_diagnosis && \
        ./bench_diagnosis all 2>/tmp/diag_log.txt" > "$RESULTS_DIR/$outfile"
    echo "    结果: $RESULTS_DIR/$outfile"
}

# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
case "${1:-}" in
    e1)     run_e1 "$2" ;;
    e2)     run_e2 "$2" ;;
    e2.1)   run_e2_1 "$2" ;;
    e3)     run_e3 "$2" ;;
    e4)     run_e4 "$2" ;;
    e5)     run_e5 "$2" ;;
    e6)     run_e6 "$2" ;;
    e7)     run_e7 "$2" ;;
    diag)   run_diag "$2" ;;
    all)
        host="${2:-tc_new}"
        run_e1 "$host"
        echo ""
        run_e2 "$host"
        echo ""
        run_e2_1 "$host"
        echo ""
        run_e3 "$host"
        echo ""
        run_e4 "$host"
        echo ""
        run_e5 "$host"
        echo ""
        run_e6 "$host"
        echo ""
        echo "========== 全部完成 =========="
        echo "结果目录: $RESULTS_DIR/"
        ls -lh "$RESULTS_DIR/"*.csv 2>/dev/null | tail -20
        ;;
    status)
        host=$(get_host "$2")
        echo ">>> 检查 $host 远程任务 ..."
        ssh "$host" "ps aux | grep -E 'gpu|bench_diagnosis' | grep -v grep || echo '  无运行中的任务'"
        ;;
    clean)
        host=$(get_host "$2")
        echo ">>> 远程清理 $host ..."
        ssh "$host" "find $REMOTE_DIR/benchmark -name 'gpu' -type f -delete; \
            rm -f $REMOTE_DIR/benchmark/experiments/e0_diagnosis/bench_diagnosis"
        echo "    done."
        ;;
    "")
        echo "GenSolver 统一实验入口"
        echo ""
        echo "实验:"
        echo "  ./run.sh e1 [host]    E1: vs MIP（公平对比）"
        echo "  ./run.sh e2 [host]    E2: vs Routing（参考对比）"
        echo "  ./run.sh e2.1 [host]  E2.1: Custom Routing（优先级 VRP）"
        echo "  ./run.sh e3 [host]    E3: Ablation（消融实验）"
        echo "  ./run.sh e4 [host]    E4: Scalability（可扩展性）"
        echo "  ./run.sh e5 [host]    E5: Generality（通用性验证）"
        echo "  ./run.sh e6 [host]    E6: GPU Hardware（硬件对比）"
        echo "  ./run.sh diag [host]  性能诊断"
        echo "  ./run.sh all [host]   全部运行"
        echo ""
        echo "工具:"
        echo "  ./run.sh status [host]  查看远程任务"
        echo "  ./run.sh clean [host]   远程清理"
        echo ""
        echo "host: tc_new (T4, 默认) | tch (V100)"
        echo "结果: benchmark/results/"
        ;;
    *)
        echo "未知命令: $1"
        echo "运行 ./run.sh 查看帮助"
        exit 1
        ;;
esac
