"""
标准实例解析器 — 从 TSPLIB / CVRPLIB 官方文件读取数据
数据文件位于 data/tsplib/ 和 data/cvrplib/
"""
import math
import os

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TSPLIB_DIR = os.path.join(DATA_ROOT, "tsplib")
CVRPLIB_DIR = os.path.join(DATA_ROOT, "cvrplib")


def parse_tsp(filepath):
    """解析 TSPLIB .tsp 文件（EUC_2D 格式）"""
    meta = {}
    coords = []
    reading_coords = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            if line in ("EOF", "DISPLAY_DATA_SECTION"):
                break
            if reading_coords:
                parts = line.split()
                coords.append((float(parts[1]), float(parts[2])))
            else:
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip()

    n = int(meta.get("DIMENSION", len(coords)))
    assert len(coords) == n, f"Expected {n} coords, got {len(coords)}"
    return {"name": meta.get("NAME", ""), "n": n, "coords": coords}


def parse_vrp(filepath):
    """解析 CVRPLIB .vrp 文件"""
    meta = {}
    coords = []
    demands = []
    section = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "NODE_COORD_SECTION":
                section = "coord"
                continue
            elif line == "DEMAND_SECTION":
                section = "demand"
                continue
            elif line in ("DEPOT_SECTION", "EOF"):
                section = None
                continue

            if section == "coord":
                parts = line.split()
                coords.append((float(parts[1]), float(parts[2])))
            elif section == "demand":
                parts = line.split()
                demands.append(int(parts[1]))
            elif ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()

    n = int(meta.get("DIMENSION", len(coords)))
    capacity = int(meta.get("CAPACITY", 0))
    name = meta.get("NAME", "")

    comment = meta.get("COMMENT", "")
    optimal = 0
    if "Optimal value:" in comment:
        optimal = int(comment.split("Optimal value:")[-1].strip().rstrip(")"))

    return {
        "name": name,
        "n": n,
        "coords": coords,
        "demands": demands,
        "capacity": capacity,
        "optimal": optimal,
    }


def euc2d_dist_matrix(coords):
    """EUC_2D 距离矩阵（四舍五入到整数，与 TSPLIB 标准一致）"""
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist[i][j] = round(math.sqrt(dx * dx + dy * dy))
    return dist


# ============================================================
# 预定义实例列表（文件名 → 已知最优）
# ============================================================

TSP_INSTANCES = [
    {"file": "eil51.tsp",   "optimal": 426},
    {"file": "eil76.tsp",   "optimal": 538},
    {"file": "kroA100.tsp", "optimal": 21282},
    {"file": "ch150.tsp",   "optimal": 6528},
    {"file": "tsp225.tsp",  "optimal": 3916},
    {"file": "lin318.tsp",  "optimal": 42029},
    {"file": "pcb442.tsp",  "optimal": 50778},
]

VRP_INSTANCES = [
    {"file": "A-n32-k5.vrp", "optimal": 784, "n_vehicles": 5},
]


def load_tsp(entry):
    """加载一个 TSP 实例"""
    data = parse_tsp(os.path.join(TSPLIB_DIR, entry["file"]))
    data["optimal"] = entry["optimal"]
    return data


def load_vrp(entry):
    """加载一个 VRP 实例"""
    data = parse_vrp(os.path.join(CVRPLIB_DIR, entry["file"]))
    data["optimal"] = entry["optimal"]
    data["n_vehicles"] = entry["n_vehicles"]
    return data
