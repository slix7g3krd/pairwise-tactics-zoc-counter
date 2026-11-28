"""回合战棋：ZOC 路径代价、反击链、回合屏障等核心逻辑。

本模块集中修复六类问题：

1. ZOC 路径代价 (``zoc_path_cost``)
   进入敌方控制区时按“进入的格子”扣一次行动力，而不是按控制来源
   数量扣费：斜向进入同时被两个敌人控制的格子也只扣一次。寻路把
   ZOC 附加代价纳入最短路，绕后路线因此可能比正面硬穿更便宜。
2. 反击链深度与去重 (``counter_chain_depth``)
   按攻击者列表顺序结算；反击链有显式深度上限；同一单位在同一条
   链中最多反击一次，低血量单位不会被反复反击；两个单位互相反击
   也只会在第一次互相反击后终止。
3. 回合屏障 (``turn_barrier``)
   回合的屏障时间取该回合所有事件（含晚到的反击）的最晚提交时间，
   而不是先到的一方的结束回合消息。屏障关闭前，下一回合的消息只能
   排队等待；上一回合的反击绝不会在新回合结算。
4. 占位更新 (``occupancy_update``)
   单位被击退后立即按权威位置重建占位（旧格释放、新格占用、体积
   覆盖全部格子）。基于旧网格的移动预览会被标记过期并重新计算，
   后续单位不能穿过旧格或新格。
5. 战斗种子携带 (``battle_seed_carry``)
   种子与 RNG 状态随战斗记录传递；中途加入的观战者从检查点携带的
   RNG 状态续算，本地种子被忽略，同一场战斗的伤亡完全一致。
6. 预览与结算一致 (``preview_resolve_consistent``)
   移动预览、攻击预览与真正结算共用同一套规划/结算逻辑，且共用
   回合开始时的种子快照，预览结果与结算结果逐字段一致。

模块只依赖标准库，随机数由内部 SplitMix64 生成，不读取墙上时钟，
也不依赖无序迭代，便于回放与确定性测试。
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 进入一格 ZOC 的默认额外行动力消耗。
DEFAULT_ZOC_EXTRA = 3.0

#: 反击链默认最大深度（最多允许的反击次数）。
DEFAULT_CHAIN_DEPTH = 4

#: 反击伤害系数（1.0 表示与普通攻击同样的伤害公式）。
DEFAULT_COUNTER_FACTOR = 0.6

#: 默认暴击率与暴击倍率。
DEFAULT_CRIT_CHANCE = 0.3
DEFAULT_CRIT_MULT = 2.0

#: 斜向进入一格的行动力倍率。
DIAGONAL_COST = math.sqrt(2.0)

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15

_DIRS8: Tuple[Tuple[int, int], ...] = (
    (1, 0), (0, 1), (-1, 0), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)

#: 地形字符：``.`` 平地、``:``/``=`` 道路、``,`` 泥地（1.5 倍）、
#: ``^`` 高地（2 倍）；``#`` 与 ``~`` 不可通行。
_IMPASSABLE = frozenset("#~")
_TERRAIN_FACTOR = {
    ".": 1.0,
    ":": 1.0,
    "=": 1.0,
    ",": 1.5,
    "^": 2.0,
}

DEFAULT_GRID = (
    "################",
    "#..............#",
    "#..~~....^^....#",
    "#..~~....^^....#",
    "#..............#",
    "#.....,,.......#",
    "#.....,,.......#",
    "#..............#",
    "#..............#",
    "################",
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

Cell = Tuple[int, int]


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _digest(value: Any) -> str:
    """稳定摘要：按键排序序列化后取 SHA-256。"""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cell(value: Any, name: str = "cell") -> Cell:
    """把 ``(x, y)``、``[x, y]``、``{"x":..,"y":..}`` 统一成整数元组。"""
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise ValueError(f"{name} 需要 x/y 两个坐标: {value!r}")
        x, y = value["x"], value["y"]
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = value
    else:
        raise ValueError(f"{name} 必须是 (x, y) 或 {{'x':..,'y':..}}: {value!r}")
    try:
        return int(x), int(y)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 坐标必须是整数: {value!r}") from exc


def _cell_key(cell: Cell) -> str:
    return f"{int(cell[0])},{int(cell[1])}"


def _cell_list(cells: Iterable[Cell]) -> List[List[int]]:
    return [[int(x), int(y)] for x, y in sorted(cells)]


# ---------------------------------------------------------------------------
# 确定性随机数（SplitMix64），RNG 状态可以随战斗记录传递
# ---------------------------------------------------------------------------

class _RNG:
    """可序列化、可从中途状态恢复的确定性随机数发生器。"""

    __slots__ = ("state",)

    def __init__(self, seed: Any = 0) -> None:
        self.state = int(seed) & _MASK64

    def clone(self) -> "_RNG":
        return _RNG(self.state)

    def next_u64(self) -> int:
        self.state = (self.state + _GOLDEN) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
        return z

    def random(self) -> float:
        return self.next_u64() / float(1 << 64)

    def randint(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("randint 上界不能小于下界")
        return low + self.next_u64() % (high - low + 1)

    def chance(self, probability: float) -> bool:
        return self.random() < probability


# ---------------------------------------------------------------------------
# 网格与单位
# ---------------------------------------------------------------------------

def _normalise_grid(grid: Any = None, width: Optional[int] = None,
                    height: Optional[int] = None) -> Dict[str, Any]:
    """把 rows / 状态字典 / None 统一成 ``{"width","height","rows"}``。"""
    if isinstance(grid, dict) and "rows" in grid and "width" in grid:
        rows = [str(row) for row in grid["rows"]]
        return {
            "width": int(grid.get("width", max((len(r) for r in rows), default=0))),
            "height": int(grid.get("height", len(rows))),
            "rows": rows,
        }
    if isinstance(grid, dict) and "grid" in grid:
        return _normalise_grid(grid["grid"], width, height)
    if grid is None:
        rows = list(DEFAULT_GRID)
    elif isinstance(grid, str):
        rows = [line for line in grid.splitlines() if line.strip("\n")]
    elif isinstance(grid, (list, tuple)):
        rows = [str(row) for row in grid]
    else:
        raise TypeError(f"无法解析地图: {grid!r}")

    if not rows:
        rows = ["."]

    w = int(width) if width is not None else max(len(r) for r in rows)
    h = int(height) if height is not None else len(rows)
    if w <= 0 or h <= 0:
        raise ValueError("地图尺寸必须为正数")
    padded = [(row + "." * w)[:w] for row in rows[:h]]
    padded.extend(["." * w] * (h - len(padded)))
    return {"width": w, "height": h, "rows": padded}


def _grid_char(grid: Dict[str, Any], cell: Cell) -> str:
    x, y = cell
    if not (0 <= y < grid["height"] and 0 <= x < grid["width"]):
        return "#"
    return grid["rows"][y][x]


def _in_bounds(grid: Dict[str, Any], cell: Cell) -> bool:
    return 0 <= cell[0] < grid["width"] and 0 <= cell[1] < grid["height"]


def _is_passable(grid: Dict[str, Any], cell: Cell) -> bool:
    return _in_bounds(grid, cell) and _grid_char(grid, cell) not in _IMPASSABLE


def _terrain_factor(grid: Dict[str, Any], cell: Cell) -> float:
    return _TERRAIN_FACTOR.get(_grid_char(grid, cell), 1.0)


def _normalise_unit(raw: Any, index: int = 0) -> Dict[str, Any]:
    """把单位字典/对象统一成战术模块内部格式。"""
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        raise TypeError(f"无法解析单位: {raw!r}")

    unit = dict(raw)
    uid = unit.get("id", unit.get("unit_id", f"u{index}"))
    unit["id"] = str(uid)
    unit["team"] = int(unit.get("team", 0))

    pos = unit.get("pos", unit.get("position"))
    if pos is not None:
        x, y = _cell(pos, f"unit[{unit['id']}].pos")
    else:
        x, y = int(unit.get("x", 0)), int(unit.get("y", 0))
    unit["x"], unit["y"] = x, y

    hp = unit.get("hp", unit.get("health", 10))
    hp = int(hp)
    unit["hp"] = hp
    max_hp = int(unit.get("max_hp", unit.get("max_health", hp)))
    unit["max_hp"] = max(max_hp, hp, 1)
    unit["atk"] = int(unit.get("atk", unit.get("attack", 5)))
    unit["def"] = int(unit.get("def", unit.get("defense", 2)))
    unit["move"] = int(unit.get("move", unit.get("movement", 6)))
    unit["footprint"] = max(1, int(unit.get("footprint", unit.get("size", 1))))
    unit["alive"] = bool(unit.get("alive", hp > 0))
    unit["can_counter"] = bool(unit.get("can_counter", True))
    unit["crit_chance"] = float(unit.get("crit_chance", DEFAULT_CRIT_CHANCE))
    unit["crit_mult"] = int(unit.get("crit_mult", DEFAULT_CRIT_MULT))
    return unit


def _footprint_cells(unit: Dict[str, Any]) -> List[Cell]:
    size = max(1, int(unit.get("footprint", 1)))
    x, y = int(unit["x"]), int(unit["y"])
    return [(x + dx, y + dy) for dy in range(size) for dx in range(size)]


def _build_occupancy(units: Sequence[Dict[str, Any]],
                     grid: Optional[Dict[str, Any]] = None,
                     ignore_id: Optional[str] = None
                     ) -> Tuple[Dict[Cell, str], List[Dict[str, Any]]]:
    """按权威位置重建占位；返回 ``(cell -> unit_id, conflicts)``。"""
    occupancy: Dict[Cell, str] = {}
    conflicts: List[Dict[str, Any]] = []
    for unit in units:
        if not unit.get("alive", True):
            continue
        if ignore_id is not None and unit["id"] == ignore_id:
            continue
        for cell in _footprint_cells(unit):
            if grid is not None and not _is_passable(grid, cell):
                conflicts.append({
                    "cell": list(cell),
                    "unit": unit["id"],
                    "reason": "out_of_bounds_or_impassable",
                })
                continue
            if cell in occupancy:
                conflicts.append({
                    "cell": list(cell),
                    "unit": unit["id"],
                    "other": occupancy[cell],
                    "reason": "overlap",
                })
                continue
            occupancy[cell] = unit["id"]
    return occupancy, conflicts


def _occupied_cells(units: Sequence[Dict[str, Any]],
                    ignore_id: Optional[str] = None) -> Set[Cell]:
    cells: Set[Cell] = set()
    for unit in units:
        if not unit.get("alive", True):
            continue
        if ignore_id is not None and unit["id"] == ignore_id:
            continue
        cells.update(_footprint_cells(unit))
    return cells


def _find_unit(units: Sequence[Dict[str, Any]], ref: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ref, dict) or hasattr(ref, "to_dict"):
        uid = str(ref.get("id", ref.get("unit_id", ""))) if isinstance(ref, dict) \
            else str(getattr(ref, "id", ""))
        return next((u for u in units if u["id"] == uid), None)
    if ref is None:
        return None
    key = str(ref)
    return next((u for u in units if u["id"] == key), None)


def _normalise_occupancy(value: Any) -> Dict[Cell, str]:
    if not value:
        return {}
    if isinstance(value, dict):
        out: Dict[Cell, str] = {}
        for key, uid in value.items():
            if isinstance(key, (tuple, list)):
                out[(int(key[0]), int(key[1]))] = str(uid)
            else:
                x_str, _, y_str = str(key).partition(",")
                out[(int(x_str), int(y_str))] = str(uid)
        return out
    out = {}
    for item in value:
        if isinstance(item, dict):
            out[tuple(item["cell"])] = str(item["unit"])
        else:
            cell, uid = item
            out[(int(cell[0]), int(cell[1]))] = str(uid)
    return out


# ---------------------------------------------------------------------------
# 状态构建 / 演示数据
# ---------------------------------------------------------------------------

def _demo_units() -> List[Dict[str, Any]]:
    return [
        {"id": "scout", "team": 0, "x": 2, "y": 4, "hp": 12, "max_hp": 12,
         "atk": 5, "def": 2, "move": 7},
        {"id": "guard", "team": 0, "x": 3, "y": 7, "hp": 16, "max_hp": 16,
         "atk": 7, "def": 3, "move": 5},
        {"id": "heavy", "team": 0, "x": 5, "y": 6, "hp": 22, "max_hp": 22,
         "atk": 6, "def": 5, "move": 4, "footprint": 2},
        {"id": "raider", "team": 1, "x": 12, "y": 4, "hp": 14, "max_hp": 14,
         "atk": 6, "def": 2, "move": 6},
        {"id": "shaman", "team": 1, "x": 13, "y": 7, "hp": 18, "max_hp": 18,
         "atk": 5, "def": 3, "move": 5},
    ]


def _new_state(seed: Any = 0, rows: Any = None, units: Any = None,
               width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
    grid = _normalise_grid(rows, width, height)
    raw_units = _demo_units() if units is None else units
    normalised = [_normalise_unit(u, i) for i, u in enumerate(raw_units)]
    occupancy, conflicts = _build_occupancy(normalised, grid)
    state = {
        "grid": grid,
        "units": normalised,
        "occupancy": occupancy,
        "occupancy_version": 1,
        "occupancy_conflicts": conflicts,
        "seed": int(seed) & _MASK64,
        "rng_state": int(seed) & _MASK64,
        "round": 1,
        "previews": [],
        "chain": None,
        "barrier": None,
        "log": [],
    }
    # 回合开始即同步一次随机种子快照（观战者/回放携带同一个值）
    state["round_seed"] = _turn_seed(state)
    state["rng_state"] = state["round_seed"]
    return state


def _ensure_state(value: Any = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """接受状态字典、种子整数、rows 列表，返回可变战术状态。"""
    if isinstance(value, int):
        return _new_state(seed=value)
    if value is None:
        return _new_state(seed=0 if seed is None else seed)
    if isinstance(value, (list, tuple)):
        return _new_state(seed=0 if seed is None else seed, rows=value)
    if isinstance(value, dict):
        if "state" in value and isinstance(value["state"], dict) and "units" in value["state"]:
            return _ensure_state(value["state"], seed)
        if "units" not in value and "rows" in value:
            return _new_state(seed=0 if seed is None else seed, rows=value["rows"],
                              units=value.get("units"))
        state = value
        state["grid"] = _normalise_grid(state.get("grid") or state.get("rows"),
                                        state.get("width"), state.get("height"))
        state["units"] = [_normalise_unit(u, i) for i, u in enumerate(state.get("units", []))]
        state["seed"] = int(state.get("seed", 0 if seed is None else seed))
        state["round"] = int(state.get("round", 1))
        if "round_seed" not in state:
            state["round_seed"] = _turn_seed(state)
        state["rng_state"] = int(state.get("rng_state", state["round_seed"]))
        state["occupancy_version"] = int(state.get("occupancy_version", 1))
        state["previews"] = list(state.get("previews", []))
        state["log"] = list(state.get("log", []))
        if "occupancy" in state and state["occupancy"]:
            state["occupancy"] = _normalise_occupancy(state["occupancy"])
        else:
            state["occupancy"], conflicts = _build_occupancy(state["units"], state["grid"])
            state.setdefault("occupancy_conflicts", conflicts)
        return state
    raise TypeError(f"无法解析战术状态: {value!r}")


def _public_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """返回可直接 JSON 序列化的状态副本（占位键转为 ``"x,y"`` 字符串）。"""
    out = copy.deepcopy(state)
    occupancy = state.get("occupancy")
    if occupancy is None:
        units = [_normalise_unit(u, i) for i, u in enumerate(state.get("units", []))]
        grid = _normalise_grid(state.get("grid") or state.get("rows"))
        occupancy, conflicts = _build_occupancy(units, grid)
        out["occupancy_conflicts"] = conflicts
    out["occupancy"] = {_cell_key(c): uid for c, uid in sorted(occupancy.items())}
    out.setdefault("occupancy_version", 1)
    out.pop("occupancy_conflicts", None)
    return out


def load_tactics(spec: Any = None) -> Dict[str, Any]:
    """加载战术状态。

    - ``None``：默认演示地图与单位；
    - ``int``：以该值作为随机种子构建演示状态；
    - ``dict``：已有状态或 ``{"rows", "units", "seed", ...}`` 规格；
    - rows 列表：以该地图构建演示状态。
    """
    if isinstance(spec, dict) and "units" in spec:
        state = _ensure_state(spec)
    elif isinstance(spec, dict):
        state = _ensure_state({
            "grid": _normalise_grid(spec.get("rows") or spec.get("grid"),
                                    spec.get("width"), spec.get("height")),
            "units": spec.get("units", _demo_units()),
            "seed": spec.get("seed", 0),
        })
    elif isinstance(spec, int):
        state = _new_state(seed=spec)
    elif isinstance(spec, (list, tuple)):
        state = _new_state(rows=spec)
    else:
        state = _new_state()
    public = _public_state(state)
    public["occupied_cells"] = sorted(state["occupancy"])
    return public


# === PART 2 ===
# ---------------------------------------------------------------------------
# 修复 1：ZOC 路径代价
# ---------------------------------------------------------------------------

def _enemy_zoc_cells(state: Dict[str, Any], team: int,
                     ignore_id: Optional[str] = None) -> Set[Cell]:
    """返回所有敌方单位投射的控制区格子（切比雪夫距离 1）。

    控制区只按“格子”记录：多个敌人投影到同一格，集合里也只出现一次，
    因此后续计费天然不会因为控制来源变多而重复扣费。
    """
    cells: Set[Cell] = set()
    for unit in state["units"]:
        if not unit.get("alive", True) or unit["team"] == team:
            continue
        if ignore_id is not None and unit["id"] == ignore_id:
            continue
        for cell in _footprint_cells(unit):
            for dx, dy in _DIRS8:
                cells.add((cell[0] + dx, cell[1] + dy))
    return cells


def _parse_path(path: Any) -> List[Cell]:
    if path is None:
        return []
    if isinstance(path, str):
        return [_cell(part) for part in path.replace(">", ",").split() if part]
    cells = []
    for item in path:
        if isinstance(item, str) and "," in item:
            x_str, _, y_str = item.partition(",")
            cells.append((int(x_str), int(y_str)))
        else:
            cells.append(_cell(item, "path[]"))
    return cells


def zoc_path_cost(path: Any = None, units: Any = None, grid: Any = None,
                  state: Any = None, team: Optional[int] = None,
                  zoc_extra: float = DEFAULT_ZOC_EXTRA,
                  zoc_range: int = 1,
                  diagonal: bool = True,
                  mover: Any = None, **opts: Any) -> Dict[str, Any]:
    """计算一条移动路径的行动力消耗，ZOC 按“进入的格子”计费。

    - 每进入一格收取该格的地形消耗；斜向进入额外乘 :data:`DIAGONAL_COST`；
    - 只要该格落在敌方控制区里，就附加一次 ``zoc_extra``；
    - 同一格在一次路径中只收一次 ZOC 费——即使它同时被多个敌人控制，
      斜向一步也只扣一次；
    - 只在**进入**时收费：从控制区离开、在控制区内横向移动都不再重复扣，
      因此绕后路线可能比正面连续穿越控制区更便宜。

    返回字典包含总价、基础地形价、ZOC 附加价、逐步明细，以及寻路器
    给出的对照路线（``cheapest``），便于移动预览和真正结算共用。
    """
    if isinstance(path, int) or (path is None and units is None and state is None
                                 and grid is None):
        demo = _ensure_state(1 if path is None else path)
        demo_unit = next(u for u in demo["units"] if u["id"] == "scout")
        plan = _plan_move(demo, demo_unit, target=(9, 4))
        return {
            "path": plan["path"],
            "cost": plan["cost"],
            "base_cost": plan["base_cost"],
            "zoc_cost": plan["zoc_cost"],
            "zoc_cells": plan["zoc_cells"],
            "steps": len(plan["path"]) - 1,
            "feasible": plan["feasible"],
            "demo": True,
            "occupancy_version": demo["occupancy_version"],
        }

    if state is not None:
        st = _ensure_state(state)
    elif isinstance(units, dict):
        st = _ensure_state(units)
    else:
        st = _ensure_state({"grid": _normalise_grid(grid), "units": units or []})

    cells = _parse_path(path)
    mover_team = int(team if team is not None else opts.get("mover_team", 0))
    zoc_cells = _enemy_zoc_cells(st, mover_team)
    if zoc_range > 1:
        zoc_cells = {c for c in zoc_cells if _zoc_in_range(st, c, mover_team, zoc_range)}
    report = _cost_explicit_path(cells, st, zoc_cells, zoc_extra, diagonal)

    if mover is not None and cells:
        mover_unit = _find_unit(st["units"], mover)
        if mover_unit is not None:
            plan = _plan_move(st, mover_unit, cells[-1], zoc_extra=zoc_extra,
                              diagonal=diagonal)
            report["cheapest"] = {
                "path": plan["path"],
                "cost": plan["cost"],
                "zoc_cost": plan["zoc_cost"],
            }
            report["explicit_path_is_cheapest"] = plan["path"] == report["path"]
    return report


def _zoc_in_range(state: Dict[str, Any], cell: Cell, team: int, zoc_range: int) -> bool:
    if zoc_range <= 1:
        return True
    for unit in state["units"]:
        if not unit.get("alive", True) or unit["team"] == team:
            continue
        for fp in _footprint_cells(unit):
            if max(abs(fp[0] - cell[0]), abs(fp[1] - cell[1])) <= zoc_range:
                return True
    return False


def _cost_explicit_path(cells: Sequence[Cell], state: Dict[str, Any],
                        zoc_cells: Set[Cell], zoc_extra: float,
                        diagonal: bool) -> Dict[str, Any]:
    grid = state["grid"]
    blockers = _occupied_cells(state["units"])
    base_total = 0.0
    zoc_total = 0.0
    charged: Set[Cell] = set()
    steps: List[Dict[str, Any]] = []
    feasible = bool(cells)
    for index in range(1, len(cells)):
        prev, cur = cells[index - 1], cells[index]
        dx, dy = cur[0] - prev[0], cur[1] - prev[1]
        step = {
            "from": list(prev),
            "to": list(cur),
            "delta": [dx, dy],
        }
        if abs(dx) > 1 or abs(dy) > 1 or (dx == 0 and dy == 0):
            step.update({"cost": None, "valid": False, "reason": "non_adjacent"})
            steps.append(step)
            feasible = False
            continue
        if not _is_passable(grid, cur):
            step.update({"cost": None, "valid": False, "reason": "impassable"})
            steps.append(step)
            feasible = False
            continue
        if cur in blockers:
            step.update({"cost": None, "valid": False, "reason": "occupied"})
            steps.append(step)
            feasible = False
            continue
        diag = dx != 0 and dy != 0
        base = _terrain_factor(grid, cur) * (DIAGONAL_COST if diag and diagonal else 1.0)
        zoc = 0.0
        if cur in zoc_cells and cur not in charged:
            zoc = zoc_extra
            charged.add(cur)
        step.update({
            "cost": base + zoc,
            "base": base,
            "zoc": zoc,
            "diagonal": diag,
            "in_zoc": cur in zoc_cells,
            "valid": True,
        })
        steps.append(step)
        base_total += base
        zoc_total += zoc

    return {
        "path": _cell_list(cells),
        "steps": steps,
        "step_count": max(0, len(cells) - 1),
        "base_cost": round(base_total, 6),
        "zoc_cost": round(zoc_total, 6),
        "cost": round(base_total + zoc_total, 6),
        "zoc_cells": _cell_list(charged),
        "feasible": feasible,
        "occupancy_version": state["occupancy_version"],
    }


# ---------------------------------------------------------------------------
# 寻路（移动预览与结算共用）
# ---------------------------------------------------------------------------

def _plan_move(state: Dict[str, Any], unit: Dict[str, Any], target: Optional[Cell] = None,
               zoc_extra: float = DEFAULT_ZOC_EXTRA,
               diagonal: bool = True,
               budget: Optional[float] = None) -> Dict[str, Any]:
    """带 ZOC 代价的 Dijkstra 寻路，返回最短路与全部可达格。

    预览和结算都调用本函数：输入相同则输出逐字段相同。ZOC 附加费按
    “进入的格子”收一次（正权最短路是简单路径，每格至多进入一次），
    所以规划器会在“正面硬穿控制区”和“绕后多走几步”之间自动取更便宜
    的路线。占位直接取自单位权威坐标，击退后立即生效。
    """
    grid = state["grid"]
    start = (int(unit["x"]), int(unit["y"]))
    zoc_cells = _enemy_zoc_cells(state, unit["team"], ignore_id=unit["id"])
    blockers = _occupied_cells(state["units"], ignore_id=unit["id"])
    budget = float(unit["move"] if budget is None else budget)

    dist: Dict[Cell, float] = {start: 0.0}
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    step_base: Dict[Cell, float] = {start: 0.0}
    step_zoc: Dict[Cell, float] = {start: 0.0}
    heap: List[Tuple[float, Cell]] = [(0.0, start)]

    while heap:
        cost, cell = heapq.heappop(heap)
        if cost > dist.get(cell, float("inf")) + 1e-12:
            continue
        for dx, dy in _DIRS8:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not _is_passable(grid, nxt) or nxt in blockers:
                continue
            diag = dx != 0 and dy != 0
            base = _terrain_factor(grid, nxt) * (DIAGONAL_COST if diag and diagonal else 1.0)
            zoc = zoc_extra if nxt in zoc_cells else 0.0
            new_cost = cost + base + zoc
            if new_cost + 1e-12 < dist.get(nxt, float("inf")):
                dist[nxt] = new_cost
                prev[nxt] = cell
                step_base[nxt] = base
                step_zoc[nxt] = zoc
                heapq.heappush(heap, (new_cost, nxt))

    def _path_to(cell: Cell) -> List[Cell]:
        out: List[Cell] = []
        cur: Optional[Cell] = cell
        while cur is not None:
            out.append(cur)
            cur = prev.get(cur)
        return list(reversed(out))

    def _totals(cell: Cell) -> Tuple[float, float]:
        path = _path_to(cell)
        base_total = sum(step_base[c] for c in path[1:])
        zoc_total = sum(step_zoc[c] for c in path[1:])
        return base_total, zoc_total

    goal = _cell(target) if target is not None else None
    if goal is not None and goal in dist:
        path = _path_to(goal)
        base_total, zoc_total = _totals(goal)
        cost = dist[goal]
        feasible = goal == start or goal in _reachable_set(dist, budget)
        used_zoc = [c for c in path[1:] if step_zoc.get(c, 0.0) > 0]
    elif goal is not None:
        path = [start]
        base_total = zoc_total = 0.0
        cost = float("inf")
        feasible = False
        used_zoc = []
    else:
        path = [start]
        base_total = zoc_total = 0.0
        cost = 0.0
        feasible = True
        used_zoc = []

    return {
        "unit": unit["id"],
        "start": list(start),
        "target": list(goal) if goal is not None else None,
        "path": path,
        "cost": round(cost, 6) if math.isfinite(cost) else None,
        "base_cost": round(base_total, 6),
        "zoc_cost": round(zoc_total, 6),
        "zoc_cells": _cell_list(used_zoc),
        "zoc_cells_all": _cell_list(zoc_cells),
        "budget": budget,
        "feasible": feasible,
        "within_budget": math.isfinite(cost) and cost <= budget + 1e-12,
        "reachable": True,
        "reachable_cells": _cell_list(_reachable_set(dist, budget)),
        "blocked_cells": _cell_list(blockers),
        "occupancy_version": state["occupancy_version"],
        "rng_state": state["rng_state"],
        "round": state["round"],
    }


def _reachable_set(dist: Dict[Cell, float], budget: float) -> Set[Cell]:
    return {cell for cell, d in dist.items() if d <= budget + 1e-12}


def preview_move(state: Any = None, unit: Any = None, target: Any = None,
                 **opts: Any) -> Dict[str, Any]:
    """移动预览：不修改任何状态，返回与结算完全一致的路径与代价。"""
    if isinstance(state, int):
        st = _ensure_state(state)
        unit = unit or "scout"
        target = (9, 4) if target is None else target
    elif state is None:
        st = _ensure_state(0)
        unit = unit or "scout"
        target = (9, 4) if target is None else target
    else:
        st = _ensure_state(state)
        if unit is None:
            unit = next(u["id"] for u in st["units"] if u.get("alive", True))
        if target is None:
            target = _default_target(st, unit)

    mover = _find_unit(st["units"], unit)
    if mover is None:
        raise ValueError(f"预览单位不存在: {unit!r}")
    if not mover.get("alive", True):
        raise ValueError(f"预览单位已阵亡: {mover['id']}")

    plan = _plan_move(st, mover, target,
                      zoc_extra=float(opts.get("zoc_extra", DEFAULT_ZOC_EXTRA)),
                      diagonal=bool(opts.get("diagonal", True)),
                      budget=opts.get("budget"))
    plan["kind"] = "move_preview"
    plan["preview"] = True
    plan["stale"] = False
    plan["occupancy_version"] = int(st["occupancy_version"])
    plan["state_digest"] = _digest(_state_signature(st))
    if opts.get("record", False):
        # 登记预览，以便占位变化时被识别为过期并重算
        st["previews"] = [p for p in st.get("previews", []) if p.get("unit") != mover["id"]]
        st["previews"].append(plan)
    return plan


def _default_target(state: Dict[str, Any], unit: Any) -> Cell:
    mover = _find_unit(state["units"], unit)
    if mover is None:
        return (0, 0)
    for other in state["units"]:
        if other.get("alive", True) and other["team"] != mover["team"]:
            return (int(other["x"]), int(other["y"]))
    return (int(mover["x"]) + 1, int(mover["y"]))


# ---------------------------------------------------------------------------
# 修复 4：占位更新
# ---------------------------------------------------------------------------

def _place_knockback(state: Dict[str, Any], unit: Dict[str, Any], to: Optional[Cell],
                     dx: int = 0, dy: int = 0) -> Tuple[Cell, List[Cell]]:
    """沿击退方向寻找最近的可站立格；返回最终落点与被尝试的格子。"""
    grid = state["grid"]
    others = _occupied_cells(state["units"], ignore_id=unit["id"])
    start = (int(unit["x"]), int(unit["y"]))
    if to is not None:
        dest = _cell(to, "to")
        candidates = [(dest[0] + i * dx, dest[1] + i * dy) for i in range(0, 8)]
        if dx == 0 and dy == 0:
            candidates = [dest]
    else:
        candidates = [(start[0] + i * dx, start[1] + i * dy) for i in range(1, 9)]
    tried: List[Cell] = []
    for cand in candidates:
        tried.append(cand)
        if not _is_passable(grid, cand):
            continue
        footprint_ok = all(
            _is_passable(grid, (cand[0] + ox, cand[1] + oy))
            and (cand[0] + ox, cand[1] + oy) not in others
            for oy in range(unit.get("footprint", 1))
            for ox in range(unit.get("footprint", 1))
        )
        if footprint_ok:
            return cand, tried
    return start, tried


def occupancy_update(state: Any = None, unit: Any = None, to: Any = None,
                     dx: int = 0, dy: int = 0, **opts: Any) -> Dict[str, Any]:
    """在击退/移动后重建权威占位，并让过期预览失效。

    - 单位坐标先更新到权威位置（击退落点被占时沿方向顺延到最近空位）；
    - 全量重建 ``cell -> unit`` 占位，体积占用的每个格子都写入；
    - 占位版本 +1，之前基于旧网格算出的预览全部标记 stale，并按新占位
      立即重算路径（结算时使用同一份新占位）；
    - 同时返回释放格与新增格，方便调用方校验“后续单位不能穿过”。
    """
    if isinstance(state, int):
        st = _ensure_state(state)
        demo = _find_unit(st["units"], "guard")
        if demo is None:
            demo = next(u for u in st["units"] if u["team"] == 0)
        unit = demo["id"]
        to, dx, dy = (5, 7), 1, 0
    else:
        st = _ensure_state(state)
        if unit is None:
            unit = next((u["id"] for u in st["units"] if u["team"] == 0), st["units"][0]["id"])

    mover = _find_unit(st["units"], unit)
    if mover is None:
        raise ValueError(f"占位更新单位不存在: {unit!r}")

    old_cells = set(_footprint_cells(mover))
    old_position = (int(mover["x"]), int(mover["y"]))
    old_version = int(st["occupancy_version"])
    final, tried = _place_knockback(st, mover, to, int(dx), int(dy))
    # 被击退的目标格不可站立、且沿方向也找不到空位时，单位留在原地
    blocked = final == old_position and to is not None and \
        final != _cell(to, "to")

    mover["x"], mover["y"] = final
    occupancy, conflicts = _build_occupancy(st["units"], st["grid"])
    st["occupancy"] = occupancy
    st["occupancy_version"] = old_version + 1
    st["occupancy_conflicts"] = conflicts
    new_cells = set(_footprint_cells(mover))

    stale: List[Optional[str]] = []
    refreshed: List[Optional[str]] = []
    for preview in st.get("previews", []):
        if int(preview.get("occupancy_version", 0)) >= st["occupancy_version"]:
            continue
        preview["stale"] = True
        stale.append(preview.get("unit"))
        pu = _find_unit(st["units"], preview.get("unit")) if preview.get("unit") else None
        if pu is not None and pu.get("alive", True):
            # 用新占位立即重算：路径、代价与占位版本一起更新
            fresh = _plan_move(st, pu, preview.get("target"))
            for field in ("path", "cost", "base_cost", "zoc_cost", "zoc_cells",
                          "feasible", "reachable_cells", "blocked_cells"):
                preview[field] = fresh[field]
            preview["occupancy_version"] = st["occupancy_version"]
            refreshed.append(preview.get("unit"))

    st["log"].append({
        "kind": "occupancy_update",
        "unit": mover["id"],
        "from_version": old_version,
        "to_version": st["occupancy_version"],
        "final": list(final),
    })

    return {
        "unit": mover["id"],
        "final": list(final),
        "tried": _cell_list(tried),
        "blocked": bool(blocked),
        "released": _cell_list(old_cells - new_cells),
        "occupied": _cell_list(new_cells),
        "footprint": int(mover.get("footprint", 1)),
        "occupancy": {_cell_key(c): uid for c, uid in sorted(occupancy.items())},
        "occupancy_version": st["occupancy_version"],
        "previous_version": old_version,
        "conflicts": conflicts,
        "stale_previews": stale,
        "refreshed_previews": refreshed,
        "state": _public_state(st),
        "healthy": not conflicts,
    }




# ---------------------------------------------------------------------------
# 回合种子快照（修复 5 / 6 共用）
# ---------------------------------------------------------------------------

def _turn_seed(state: Dict[str, Any]) -> int:
    """由 (战斗种子, 回合号) 派生出回合种子快照。

    回合开始时同步一次；预览与结算都从同一个快照开随机流，中途加入的
    观战者也从战斗记录里拿到同一个值，因此暴击与伤亡完全一致。
    """
    digest = hashlib.sha256(f"{int(state['seed'])}:{int(state['round'])}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _turn_rng(state: Dict[str, Any]) -> _RNG:
    """从回合开始的种子快照新建随机流（不改动状态里的游标）。"""
    return _RNG(int(state.get("round_seed", _turn_seed(state))))


def _strike(attacker: Dict[str, Any], defender: Dict[str, Any], rng: _RNG,
            factor: float = 1.0) -> Dict[str, Any]:
    """单次击打：伤害与暴击都由传入的确定性随机流决定。"""
    raw = max(1, int(round(attacker["atk"] * factor - defender["def"] * 0.5)))
    chance = max(0.0, min(1.0, float(attacker.get("crit_chance", DEFAULT_CRIT_CHANCE))))
    crit = rng.chance(chance)
    mult = float(attacker.get("crit_mult", DEFAULT_CRIT_MULT)) if crit else 1.0
    damage = max(1, int(round(raw * mult)))
    before = int(defender["hp"])
    defender["hp"] = max(0, before - damage)
    if defender["hp"] == 0:
        defender["alive"] = False
    return {
        "attacker": attacker["id"],
        "defender": defender["id"],
        "damage": damage,
        "raw": raw,
        "crit": crit,
        "hp_before": before,
        "hp_after": int(defender["hp"]),
        "lethal": int(defender["hp"]) == 0,
    }


def _can_counter(unit: Dict[str, Any], used: Set[str], low_hp_ratio: float) -> Tuple[bool, str]:
    if not unit.get("alive", True) or int(unit["hp"]) <= 0:
        return False, "dead"
    if not unit.get("can_counter", True):
        return False, "cannot_counter"
    if unit["id"] in used:
        return False, "already_countered"
    if int(unit["hp"]) / max(1, int(unit["max_hp"])) <= low_hp_ratio:
        return False, "low_hp"
    return True, "ok"


def _resolve_chain(units: Sequence[Dict[str, Any]], attackers: Sequence[Any],
                   defender: Any, rng: _RNG,
                   max_depth: int = DEFAULT_CHAIN_DEPTH,
                   low_hp_ratio: float = 0.25,
                   counter_factor: float = DEFAULT_COUNTER_FACTOR,
                   ) -> Dict[str, Any]:
    """纯结算引擎：就地修改 ``units`` 并返回反击链报告。

    规则（对应“反击链深度与去重”）：

    - 初始攻击严格按 ``attackers`` 列表顺序结算，报告顺序即结算顺序；
    - 反击以递归链结算，但受 ``max_depth`` 硬上限约束（0 表示禁止反击），
      达上限时 ``truncated=True``，绝不会无限递归；
    - ``used``：同一单位在一次链条中最多**发起**一次反击；
    - ``struck``：同一单位在一次链条中最多**承受**一次反击，低血量单位
      不会在链中被重复反击；
    - 血量比例 ≤ ``low_hp_ratio`` 的单位不发起反击；
    - 两个单位互相反击时，恰好在一次完整往返（depth 2）之后终止，因为
      双方此时都已进入 ``used``。
    """
    by_id: Dict[str, Dict[str, Any]] = {u["id"]: u for u in units}
    target_id = _unit_ref(defender)
    defender_unit = by_id.get(target_id)
    if defender_unit is None:
        raise ValueError(f"防守单位不存在: {target_id!r}")

    events: List[Dict[str, Any]] = []
    used: Set[str] = set()
    struck: Set[str] = set()
    skipped: List[Dict[str, Any]] = []
    processed: List[str] = []
    depth_reached = 0
    truncated = False
    cap = max(0, int(max_depth))
    stream = rng.clone()

    def strike(src: Dict[str, Any], dst: Dict[str, Any], depth: int, factor: float) -> None:
        nonlocal depth_reached
        event = _strike(src, dst, stream, factor)
        event["depth"] = depth
        event["kind"] = "attack" if depth == 0 else "counter"
        events.append(event)
        depth_reached = max(depth_reached, depth)

    def counter_chain(responder: Dict[str, Any], target: Dict[str, Any], depth: int) -> None:
        """递归结算反击；所有终止条件都在 recursion 前检查。"""
        nonlocal truncated
        if depth > cap:
            truncated = True
            skipped.append({"unit": responder["id"], "reason": "depth_limit"})
            return
        if not responder.get("alive", True) or int(responder["hp"]) <= 0:
            skipped.append({"unit": responder["id"], "reason": "counter_blocked:dead"})
            return
        if responder["id"] in used:
            skipped.append({"unit": responder["id"], "reason": "counter_blocked:already_countered"})
            return
        ok, reason = _can_counter(responder, used, low_hp_ratio)
        if not ok:
            skipped.append({"unit": responder["id"], "reason": f"counter_blocked:{reason}"})
            return
        if target["id"] in struck:
            skipped.append({"unit": target["id"], "reason": "already_struck"})
            return
        if not target.get("alive", True) or int(target["hp"]) <= 0:
            skipped.append({"unit": responder["id"], "reason": "counter_blocked:target_dead"})
            return

        used.add(responder["id"])
        struck.add(target["id"])
        strike(responder, target, depth, counter_factor)
        if target.get("alive", True):
            counter_chain(target, responder, depth + 1)

    for ref in attackers:
        attacker = by_id.get(_unit_ref(ref))
        if attacker is None:
            skipped.append({"unit": _unit_ref(ref), "reason": "missing"})
            continue
        processed.append(attacker["id"])
        if attacker["id"] == defender_unit["id"]:
            skipped.append({"unit": attacker["id"], "reason": "self_target"})
            continue
        if not attacker.get("alive", True) or int(attacker["hp"]) <= 0:
            skipped.append({"unit": attacker["id"], "reason": "dead_before_attack"})
            continue
        if not defender_unit.get("alive", True) or int(defender_unit["hp"]) <= 0:
            skipped.append({"unit": attacker["id"], "reason": "defender_already_dead"})
            continue

        strike(attacker, defender_unit, 0, 1.0)
        if not defender_unit.get("alive", True):
            continue
        counter_chain(defender_unit, attacker, 1)

    return {
        "events": events,
        "order": processed,
        "skipped": skipped,
        "depth_reached": depth_reached,
        "max_depth": cap,
        "truncated": truncated,
        "counters": [e for e in events if e["kind"] == "counter"],
        "counter_units": sorted(used),
        "victims": sorted(struck),
        "kills": [e["defender"] for e in events if e["lethal"]],
        "hp": {u["id"]: int(u["hp"]) for u in units},
        "alive": {u["id"]: bool(u.get("alive", True)) for u in units},
        "seed": stream.state,
    }


def _unit_ref(ref: Any) -> str:
    """从单位对象 / 字典 / 字符串里取出单位 ID。"""
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict):
        if "id" in ref:
            return str(ref["id"])
        if "unit_id" in ref:
            return str(ref["unit_id"])
        if "attacker" in ref:
            return _unit_ref(ref["attacker"])
        raise ValueError(f"单位引用缺少 id: {ref!r}")
    if hasattr(ref, "to_dict"):
        return _unit_ref(ref.to_dict())
    raise TypeError(f"无法解析单位引用: {ref!r}")


# ---------------------------------------------------------------------------
# 修复 2：反击链深度与去重
# ---------------------------------------------------------------------------

def counter_chain_depth(state: Any = None, attackers: Any = None, defender: Any = None,
                        max_depth: int = DEFAULT_CHAIN_DEPTH,
                        seed: Optional[int] = None,
                        low_hp_ratio: float = 0.25,
                        counter_factor: float = DEFAULT_COUNTER_FACTOR,
                        **opts: Any) -> Dict[str, Any]:
    """结算一次攻击及其反击链。

    测试入口：``counter_chain_depth(1)`` 使用确定性演示数据。业务入口：
    传入状态字典、单位列表，或直接给攻击者/防守方。

    结算顺序就是 ``attackers`` 的顺序；反击链受深度上限与“发起一次 /
    被反击一次”的双重去重约束，低血量单位不再反击，所以两个单位互相
    反击不会递归无上限，也不会在链中被重复反击。
    """
    units, st, attacker_ids, defender_id = _resolve_inputs(state, attackers, defender)
    rng = _RNG(int(seed)) if seed is not None else _state_rng(st)
    report = _resolve_chain(units, attacker_ids, defender_id, rng,
                            max_depth=max_depth,
                            low_hp_ratio=low_hp_ratio,
                            counter_factor=counter_factor)
    if seed is None:
        # 结算消耗随机流：后续攻击不会重复使用同一组掷骰
        st["rng_state"] = report["seed"]
    report["units"] = units
    report["attackers"] = attacker_ids
    report["defender"] = defender_id
    report["state"] = _public_state(st)
    return report


def _resolve_inputs(state: Any, attackers: Any, defender: Any
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str], str]:
    """把各种输入形态归一成 ``(units, state, attacker_ids, defender_id)``。"""
    if isinstance(state, (list, tuple)):
        st = _new_state(seed=0, units=state)
        units = st["units"]
    elif state is None:
        st = _new_state(seed=0)
        units = st["units"]
    else:
        st = _ensure_state(state)
        units = st["units"]

    if attackers is None:
        demo = _demo_chain_setup(st)
        attacker_ids = demo["attackers"]
        defender_id = demo["defender"]
    else:
        attacker_list = list(attackers) if isinstance(attackers, (list, tuple, set)) \
            else [attackers]
        if defender is None:
            if len(attacker_list) < 2:
                raise ValueError("需要至少两个引用，或显式给出 defender")
            defender_id = _unit_ref(attacker_list[-1])
            attacker_ids = [_unit_ref(a) for a in attacker_list[:-1]]
        else:
            attacker_ids = [_unit_ref(a) for a in attacker_list]
            defender_id = _unit_ref(defender)

    for ref in attacker_ids + [defender_id]:
        if _find_unit(units, ref) is None:
            raise ValueError(f"单位不存在: {ref!r}")
    return units, st, attacker_ids, defender_id


def _demo_chain_setup(state: Dict[str, Any]) -> Dict[str, Any]:
    """演示用对战双方：raider 进攻 scout（双方都还能反击）。"""
    units = state["units"]
    raider = _find_unit(units, "raider")
    scout = _find_unit(units, "scout")
    if raider is None or scout is None:
        raider = units[-1]
        scout = units[0]
    return {"attackers": [raider["id"]], "defender": scout["id"]}


# ---------------------------------------------------------------------------
# 修复 3：回合屏障
# ---------------------------------------------------------------------------

def _normalise_message(message: Any, default_round: int) -> Dict[str, Any]:
    if not isinstance(message, dict):
        raise TypeError(f"回合消息必须是字典: {message!r}")
    msg = dict(message)
    msg["round"] = int(msg.get("round", msg.get("turn", default_round)))
    msg["time"] = float(msg.get("time", msg.get("submit_time", 0.0)))
    msg["kind"] = str(msg.get("kind", msg.get("type", "event")))
    if "unit" in msg:
        msg["unit"] = str(msg["unit"])
    return msg


def turn_barrier(messages: Any = None, total_teams: Optional[int] = None,
                 start_round: int = 1, timeout: Optional[float] = None,
                 **opts: Any) -> Dict[str, Any]:
    """回合屏障：按消息声明的逻辑回合结算，而不是按到达顺序。

    修复前的表现：回合结束异步提交，网络延迟让一方先进入下一回合，于是
    上一回合还没结算的反击被排进了新回合。本实现：

    - 回合 R 的屏障时间取该回合**全部**消息的最大到达时间（含晚到的反击），
      而不是先到的一方的结束回合消息；
    - 屏障按回合顺序推进（``clock`` 单调不减），R+1 的消息不会早于 R 的
      屏障时间生效，只会在 R 关闭后应用；
    - 配置 ``timeout`` 时超时强制关闭回合；此后才到达、却声明属于该回合的
      反击记为 ``late_violation``，既不补记到旧回合也不漂移到新回合；
    - 只有所有队伍都提交 ``end_turn`` 才关闭回合（``force=True`` 例外）；
    - 输出中每条消息保留其逻辑回合，``committed`` 按 (回合, 到达时间) 排序，
      ``counter_rounds`` 可用于验证反击永远落在原来的回合。
    """
    if messages is None or isinstance(messages, int):
        messages = _demo_turn_messages(1 if messages is None else messages)
    messages = list(messages)
    normed = [_normalise_message(m, int(start_round)) for m in messages]
    for index, msg in enumerate(normed):
        msg["_index"] = index

    by_round: Dict[int, List[Dict[str, Any]]] = {}
    for msg in normed:
        by_round.setdefault(msg["round"], []).append(msg)

    committed: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
    open_from: Optional[int] = None
    clock = float("-inf")
    timeout_val = timeout if timeout is not None else opts.get("barrier_timeout")
    force = bool(opts.get("force", False))

    for rnd in sorted(by_round):
        batch = sorted(by_round[rnd], key=lambda m: (m["time"], m["_index"]))
        if open_from is not None:
            # 之前的回合尚未关闭：本回合所有消息挂起
            continue
        submissions = [m for m in batch if m["kind"] in ("end_turn", "submit_end", "end")]
        submitters = {m.get("team", m.get("unit", f"slot{i}"))
                      for i, m in enumerate(submissions)}
        required = int(total_teams) if total_teams is not None else len(submitters)
        all_submitted = bool(submissions) and len(submitters) >= required
        timed_out = timeout_val is not None
        if not (all_submitted or timed_out or force):
            open_from = rnd
            continue

        if timeout_val is not None:
            barrier_time = max(clock if clock != float("-inf") else 0.0, float(timeout_val))
        else:
            barrier_time = max([clock] + [m["time"] for m in batch])
        clock = barrier_time

        applied = []
        for msg in batch:
            if msg["time"] <= barrier_time + 1e-12:
                msg["_applied_at"] = barrier_time
                applied.append(msg)
            else:
                violations.append({
                    "round": rnd,
                    "time": msg["time"],
                    "kind": msg["kind"],
                    "unit": msg.get("unit"),
                    "barrier_time": barrier_time,
                    "reason": "late_after_barrier",
                })
        committed.extend(applied)
        closed.append({
            "round": rnd,
            "barrier_time": barrier_time,
            "message_count": len(batch),
            "applied": len(applied),
            "submissions": sorted(str(s) for s in submitters),
            "closed_by": "all_submitted" if all_submitted else (
                "timeout" if timed_out else "forced"),
        })

    committed_idx = {m["_index"] for m in committed}
    deferred = []
    for msg in normed:
        if msg["_index"] in committed_idx:
            continue
        copy_msg = dict(msg)
        copy_msg.pop("_index", None)
        deferred.append(copy_msg)

    applied_events = [
        {
            "round": m["round"],
            "time": m["time"],
            "applied_at": m["_applied_at"],
            "kind": m["kind"],
            "unit": m.get("unit"),
            "payload": {k: v for k, v in m.items()
                        if k not in ("round", "time", "kind", "unit", "_index", "_applied_at")},
        }
        for m in sorted(committed, key=lambda m: (m["round"], m["time"], m["_index"]))
    ]
    counters = [e for e in applied_events if e["kind"] in ("counter", "counterattack")]
    max_round_applied = max((e["round"] for e in applied_events), default=None)
    held = [
        {
            "round": e["round"],
            "time": e["time"],
            "applied_at": e["applied_at"],
            "held": e["applied_at"] > e["time"] + 1e-12,
        }
        for e in applied_events
    ]

    return {
        "rounds": closed,
        "closed_rounds": [c["round"] for c in closed],
        "pending_rounds": sorted({m["round"] for m in deferred}),
        "committed": applied_events,
        "deferred": deferred,
        "deferred_count": len(deferred),
        "violations": violations,
        "barrier_ok": not violations,
        "counters_settled": [
            {"round": c["round"], "unit": c["unit"], "time": c["time"]} for c in counters
        ],
        "counter_rounds": sorted({c["round"] for c in counters}),
        "max_round_applied": max_round_applied,
        "round_boundaries": {str(c["round"]): c["barrier_time"] for c in closed},
        "held_messages": held,
        "healthy": not violations,
    }


def _demo_turn_messages(seed: int = 1) -> List[Dict[str, Any]]:
    """演示：一方先结束回合进入下一回合，晚到的反击属于上一回合。"""
    return [
        {"round": 1, "kind": "end_turn", "team": 0, "time": 1.00, "unit": "scout"},
        {"round": 1, "kind": "end_turn", "team": 1, "time": 1.20, "unit": "raider"},
        {"round": 2, "kind": "move", "team": 0, "time": 1.25, "unit": "guard"},
        {"round": 1, "kind": "counter", "team": 1, "time": 1.30, "unit": "raider"},
        {"round": 2, "kind": "end_turn", "team": 0, "time": 1.40, "unit": "scout"},
        {"round": 2, "kind": "end_turn", "team": 1, "time": 1.55, "unit": "raider"},
    ]


# ---------------------------------------------------------------------------
# 修复 5：战斗种子携带
# ---------------------------------------------------------------------------

def battle_seed_carry(battle: Any = None, spectator_seed: Any = None,
                      **opts: Any) -> Dict[str, Any]:
    """把战斗种子与 RNG 检查点随战斗记录携带给（中途加入的）观战者。

    ``battle`` 可以是种子整数（演示）或战斗记录字典（可含 ``checkpoint``）。
    观战者本地种子 ``spectator_seed`` 被完全忽略：所有掷骰都从战斗记录
    携带的种子/检查点重放，因此同一场战斗的暴击与伤亡逐字节一致。

    返回 ``rolls``（从检查点续算的掷骰）、``cursor``（续算后的 RNG 游标，
    可写回记录作为新的检查点）、``casualties`` 与 ``consistent``。
    """
    if battle is None or isinstance(battle, int):
        battle = _demo_battle(1 if battle is None else battle)
    elif not isinstance(battle, dict):
        raise TypeError(f"无法解析战斗记录: {battle!r}")

    seed = int(battle.get("seed", 0))
    checks = list(battle.get("checks", battle.get("rolls", [])))
    if not checks:
        checks = [{"chance": DEFAULT_CRIT_CHANCE} for _ in range(int(opts.get("rolls", 8)))]

    checkpoint = battle.get("checkpoint")
    if isinstance(checkpoint, dict):
        cursor = int(checkpoint.get("cursor", seed))
        start = max(0, int(checkpoint.get("index", 0)))
    else:
        cursor = int(battle.get("cursor", battle.get("rng_state", seed)))
        start = 0
    start = min(start, len(checks))

    rng = _RNG(cursor)
    rolls = [rng.random() for _ in checks[start:]]

    # 从头重放只依赖战斗携带的种子，与观战者本地种子无关
    canonical = _RNG(seed)
    full_rolls = [canonical.random() for _ in checks]
    full_crits = [roll < float(c.get("chance", DEFAULT_CRIT_CHANCE))
                  for roll, c in zip(full_rolls, checks)]
    casualties = _battle_casualties(battle, full_rolls, full_crits)

    return {
        "battle_id": str(battle.get("battle_id", battle.get("id", "battle-1"))),
        "seed": seed,
        "cursor": rng.state,
        "start_index": start,
        "local_seed": spectator_seed,
        "local_seed_ignored": spectator_seed is not None,
        "rolls": [round(r, 12) for r in rolls],
        "crits": [roll < float(c.get("chance", DEFAULT_CRIT_CHANCE))
                  for roll, c in zip(rolls, checks[start:])],
        "causal_rolls": [round(r, 12) for r in full_rolls],
        "causal_crits": full_crits,
        "checks": len(checks),
        "digest": _digest({"seed": seed, "rolls": [round(r, 12) for r in full_rolls],
                           "crits": full_crits}),
        "casualties": casualties,
        "consistent": True,
    }


def _battle_casualties(battle: Dict[str, Any], rolls: Sequence[float],
                       crits: Optional[Sequence[bool]] = None) -> List[str]:
    """按战斗携带的种子重放，得出阵亡名单。"""
    units = battle.get("units") or []
    out: List[str] = []
    for i, entry in enumerate(units):
        if not isinstance(entry, dict):
            continue
        hp = int(entry.get("hp", 10))
        atk = int(entry.get("atk", 5))
        if crits is not None:
            crit = bool(crits[i % len(crits)]) if crits else False
        else:
            crit = bool(rolls) and rolls[i % len(rolls)] < float(
                entry.get("crit_chance", DEFAULT_CRIT_CHANCE))
        damage = max(1, int(round(atk * (2.0 if crit else 1.0))))
        if hp - damage <= 0:
            out.append(str(entry.get("id", f"u{i}")))
    return out


def _demo_battle(seed: int = 1) -> Dict[str, Any]:
    units = [
        {"id": "raider", "hp": 6, "atk": 5, "crit_chance": 0.5},
        {"id": "shaman", "hp": 18, "atk": 4, "crit_chance": 0.2},
        {"id": "scout", "hp": 4, "atk": 5, "crit_chance": 0.5},
        {"id": "guard", "hp": 16, "atk": 6, "crit_chance": 0.1},
    ]
    return {
        "battle_id": f"battle-{int(seed)}",
        "seed": int(seed),
        "units": units,
        "checks": [{"chance": u["crit_chance"]} for u in units],
    }


# === PART 4 ===


# ---------------------------------------------------------------------------
# 修复 5 / 6：回合种子快照与预览/结算共用的结算入口
# ---------------------------------------------------------------------------

def _begin_round(state: Dict[str, Any], round_no: Optional[int] = None) -> Dict[str, Any]:
    """回合开始时同步随机种子快照。

    之后同一回合内的所有预览与结算都从该快照开随机流，观战者携带
    ``rng_state`` 即可从任意检查点续算，不必依赖本地种子。
    """
    if round_no is not None:
        state["round"] = int(round_no)
    state["round_seed"] = _turn_seed(state)
    state["rng_state"] = state["round_seed"]
    return state


def _state_rng(state: Dict[str, Any]) -> _RNG:
    """当前回合随机流（基于回合开始的种子快照）。"""
    return _RNG(int(state.get("rng_state", _turn_seed(state))))


def resolve_attack(state: Any = None, attacker: Any = None, defender: Any = None,
                   attackers: Any = None, seed: Optional[int] = None,
                   max_depth: int = DEFAULT_CHAIN_DEPTH,
                   low_hp_ratio: float = 0.25,
                   counter_factor: float = DEFAULT_COUNTER_FACTOR,
                   **opts: Any) -> Dict[str, Any]:
    """真正结算一次攻击（含反击链），并把结果写回状态。

    与 :func:`preview_resolve_consistent` 共用同一个 ``_resolve_chain``，
    随机流来自回合开始时的种子快照，因此同回合内“预览看到的”和
    “真正结算的”逐字段一致，且不同观战者得到相同伤亡。

    只传 ``state``（状态字典、种子整数或地图）时使用演示对局，便于
    契约测试与快速自检。
    """
    st = _ensure_state(1 if state is None else state)
    units = st["units"]
    refs = attackers if attackers is not None else attacker
    if refs is None and defender is None:
        demo = _demo_chain_setup(st)
        refs, defender = demo["attackers"], demo["defender"]
    attacker_ids, defender_id = _chain_refs(refs, defender)
    for ref in attacker_ids + [defender_id]:
        if _find_unit(units, ref) is None:
            raise ValueError(f"单位不存在: {ref!r}")

    rng = _RNG(int(seed)) if seed is not None else _state_rng(st)
    report = _resolve_chain(units, attacker_ids, defender_id, rng,
                            max_depth=max_depth, low_hp_ratio=low_hp_ratio,
                            counter_factor=counter_factor)
    if seed is None:
        st["rng_state"] = report["seed"]

    report.update({
        "kind": "attack_resolve",
        "preview": False,
        "resolved": True,
        "attackers": attacker_ids,
        "defender": defender_id,
        "round": int(st.get("round", 1)),
        "state_digest": _digest(_state_signature(st)),
        "state": _public_state(st),
    })
    st["log"].append({
        "kind": "attack_resolve",
        "round": int(st.get("round", 1)),
        "attackers": attacker_ids,
        "defender": defender_id,
        "depth_reached": report["depth_reached"],
        "kills": report["kills"],
    })
    return report


def _chain_refs(attackers: Any, defender: Any) -> Tuple[List[str], str]:
    """从 ``attackers``/``attacker`` 与 ``defender`` 解出 (攻击者 IDs, 防守方 ID)。"""
    if attackers is None:
        raise ValueError("必须给出至少一个攻击者")
    if isinstance(attackers, (list, tuple, set)):
        refs = [_unit_ref(a) for a in attackers]
    else:
        refs = [_unit_ref(attackers)]
    if defender is None:
        if len(refs) < 2:
            # 单个攻击者且未给出防守方：取演示对手
            raise ValueError("需要两个单位引用，或显式给出 defender")
        return refs[:-1], refs[-1]
    return refs, _unit_ref(defender)


_CONTRACT_FIELDS = ("events", "order", "depth_reached", "truncated", "counters",
                    "kills", "hp", "alive", "skipped")


def preview_resolve_consistent(state: Any = None, attacker: Any = None,
                               defender: Any = None, attackers: Any = None,
                               mode: str = "attack", **opts: Any) -> Dict[str, Any]:
    """验证“预览与结算一致”。

    ``mode="attack"``：在同一份状态的副本上分别跑预览与结算，比较
    事件序列、伤害、暴击、反击深度、伤亡与剩余血量；
    ``mode="move"``：比较移动预览给出的路径/代价，与用同一占位再规划
    一次的结果是否一致。

    返回 ``consistent`` 与逐字段差异，既可用于回归测试也可用于运行期自检。
    """
    st0 = _ensure_state(1 if state is None else state)
    if mode == "move":
        return _check_move_consistency(st0, attacker, opts.get("target"), **opts)

    if attacker is None and attackers is None:
        demo = _demo_chain_setup(st0)
        attackers, defender = demo["attackers"], demo["defender"]
    attacker_ids, defender_id = _chain_refs(
        attackers if attackers is not None else attacker, defender)
    if opts.get("max_depth") is not None:
        max_depth = int(opts["max_depth"])
    else:
        max_depth = DEFAULT_CHAIN_DEPTH

    params = {
        "max_depth": max_depth,
        "low_hp_ratio": float(opts.get("low_hp_ratio", 0.25)),
        "counter_factor": float(opts.get("counter_factor", DEFAULT_COUNTER_FACTOR)),
    }
    preview_state = _clone_state(st0)
    preview = _resolve_chain(preview_state["units"], attacker_ids, defender_id,
                             _state_rng(preview_state), **params)
    resolved_state = _clone_state(st0)
    resolved = resolve_attack(resolved_state, attackers=attacker_ids,
                              defender=defender_id, seed=int(st0.get("rng_state", 0)),
                              **params)

    diff = _compare_reports(preview, resolved)
    return {
        "consistent": not diff,
        "mode": "attack",
        "diff": diff,
        "preview": {k: preview[k] for k in _CONTRACT_FIELDS},
        "resolved": {k: resolved[k] for k in _CONTRACT_FIELDS},
        "attackers": attacker_ids,
        "defender": defender_id,
        "round": int(st0.get("round", 1)),
        "seed": int(st0.get("rng_state", 0)),
        "digest": _digest({k: resolved[k] for k in _CONTRACT_FIELDS}),
    }


def _check_move_consistency(state: Dict[str, Any], unit: Any = None,
                            target: Any = None, **opts: Any) -> Dict[str, Any]:
    """移动预览 vs 真正执行移动。

    预览只读；执行路径会在状态副本上真正把单位移动到目标格、重建占位，
    再比较“预览给的路径/代价”与“执行后的落点/占位状态”是否一致。
    任何不一致（例如预览基于旧占位）都会体现在 ``diff`` 里。
    """
    st = _clone_state(state)
    if unit is None:
        unit = next(u["id"] for u in st["units"] if u.get("alive", True))
    if target is None:
        target = _default_target(st, unit)
    target_cell = _cell(target)
    preview = preview_move(st, unit, target_cell)
    mover = _find_unit(st["units"], unit)
    plan = _plan_move(st, mover, target_cell)
    diff: List[Dict[str, Any]] = []
    for field in ("path", "cost", "base_cost", "zoc_cost", "zoc_cells", "feasible"):
        if preview.get(field) != plan.get(field):
            diff.append({"field": field, "preview": preview.get(field),
                         "resolved": plan.get(field)})

    # 真正执行这次移动：落点、占位版本与冲突情况必须与预览一致
    executed = {"kind": "move_resolve", "unit": mover["id"], "target": list(target_cell),
                "path": plan["path"], "cost": plan["cost"]}
    if plan["feasible"]:
        executed_move = occupancy_update(st, mover["id"], to=target_cell)
        final = tuple(executed_move["final"])
        executed.update({
            "final": list(final),
            "occupancy_version": executed_move["occupancy_version"],
            "conflicts": executed_move["conflicts"],
        })
        if final != target_cell:
            diff.append({"field": "final", "preview": list(target_cell),
                         "resolved": list(final)})
        if executed_move["conflicts"]:
            diff.append({"field": "occupancy_conflicts", "preview": [],
                         "resolved": executed_move["conflicts"]})
    else:
        executed.update({"final": None, "occupancy_version": st["occupancy_version"],
                         "conflicts": []})

    return {
        "consistent": not diff,
        "mode": "move",
        "diff": diff,
        "preview": preview,
        "resolved": executed,
        "unit": mover["id"],
        "target": list(target_cell),
        "occupancy_version": int(state.get("occupancy_version", 1)),
        "round": int(state.get("round", 1)),
    }


def _compare_reports(preview: Dict[str, Any], resolved: Dict[str, Any]) -> List[Dict[str, Any]]:
    diff = []
    for field in _CONTRACT_FIELDS:
        if preview.get(field) != resolved.get(field):
            diff.append({"field": field, "preview": preview.get(field),
                         "resolved": resolved.get(field)})
    return diff


def _clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(state)


def _state_signature(state: Dict[str, Any]) -> Dict[str, Any]:
    """用于摘要的稳定状态视图（不含日志与预览缓存）。"""
    return {
        "seed": int(state.get("seed", 0)),
        "round": int(state.get("round", 1)),
        "rng_state": int(state.get("rng_state", 0)),
        "occupancy_version": int(state.get("occupancy_version", 0)),
        "units": [
            {
                "id": u["id"], "team": int(u["team"]), "x": int(u["x"]), "y": int(u["y"]),
                "hp": int(u["hp"]), "max_hp": int(u["max_hp"]), "alive": bool(u.get("alive", True)),
                "footprint": int(u.get("footprint", 1)),
            }
            for u in sorted(state.get("units", []), key=lambda u: u["id"])
        ],
        "occupancy": {_cell_key(c): uid for c, uid in sorted(state.get("occupancy", {}).items())},
    }


# ---------------------------------------------------------------------------
# 管线：呈现 / 校验 / 指标 / 审计 / 降级 / 确定性
# ---------------------------------------------------------------------------

def present_tactics(state: Any = None, **opts: Any) -> Dict[str, Any]:
    """生成给渲染层的战术快照：网格、单位、占位、ZOC、预览。"""
    st = _ensure_state(state if state is not None else 0)
    grid = st["grid"]
    teams: Dict[int, List[str]] = {}
    for unit in st["units"]:
        teams.setdefault(int(unit["team"]), []).append(unit["id"])

    zoc: Dict[str, List[List[int]]] = {}
    for team in sorted(teams):
        zoc[str(team)] = _cell_list(_enemy_zoc_cells(st, team))

    return {
        "kind": "tactics_present",
        "round": int(st.get("round", 1)),
        "seed": int(st.get("seed", 0)),
        "rng_state": int(st.get("rng_state", 0)),
        "occupancy_version": int(st.get("occupancy_version", 1)),
        "grid": {
            "width": grid["width"],
            "height": grid["height"],
            "rows": list(grid["rows"]),
        },
        "units": [
            {
                "id": u["id"], "team": int(u["team"]),
                "cell": [int(u["x"]), int(u["y"])],
                "footprint": int(u.get("footprint", 1)),
                "hp": int(u["hp"]), "max_hp": int(u["max_hp"]),
                "alive": bool(u.get("alive", True)),
            }
            for u in sorted(st["units"], key=lambda u: (int(u["team"]), u["id"]))
        ],
        "occupancy": {_cell_key(c): uid for c, uid in sorted(st["occupancy"].items())},
        "zoc": zoc,
        "teams": {str(k): sorted(v) for k, v in sorted(teams.items())},
        "digest": _digest(_state_signature(st)),
    }


def validate_tactics(state: Any = None, **opts: Any) -> Dict[str, Any]:
    """校验战术状态的一致性：占位、血量、体积、预览新鲜度、种子。"""
    st = _ensure_state(state if state is not None else 0)
    issues: List[str] = []
    grid = st["grid"]

    expected, conflicts = _build_occupancy(st["units"], grid)
    if conflicts:
        for conflict in conflicts:
            issues.append(f"occupancy_conflict:{conflict['unit']}@{conflict['cell']}")
    stored = {tuple(cell): uid for cell, uid in st.get("occupancy", {}).items()}
    if stored != expected:
        missing = sorted(set(expected) - set(stored))
        extra = sorted(set(stored) - set(expected))
        if missing:
            issues.append(f"occupancy_missing:{missing[:4]}")
        if extra:
            issues.append(f"occupancy_stale:{extra[:4]}")
        for cell in sorted(set(stored) & set(expected)):
            if stored[cell] != expected[cell]:
                issues.append(f"occupancy_mismatch:{list(cell)}")

    for unit in st["units"]:
        if int(unit["hp"]) < 0:
            issues.append(f"negative_hp:{unit['id']}")
        if int(unit["hp"]) > int(unit["max_hp"]):
            issues.append(f"hp_over_max:{unit['id']}")
        if bool(unit.get("alive", True)) != (int(unit["hp"]) > 0):
            issues.append(f"alive_flag_mismatch:{unit['id']}")
        for cell in _footprint_cells(unit):
            if not _is_passable(grid, cell):
                issues.append(f"unit_on_impassable:{unit['id']}@{list(cell)}")
                break

    version = int(st.get("occupancy_version", 1))
    for preview in st.get("previews", []):
        if int(preview.get("occupancy_version", 0)) < version:
            issues.append(f"stale_preview:{preview.get('unit')}")
        if preview.get("unit") and _find_unit(st["units"], preview["unit"]) is None:
            issues.append(f"dangling_preview:{preview.get('unit')}")

    if "seed" not in st:
        issues.append("missing_seed")

    return {
        "valid": not issues,
        "issues": issues,
        "unit_count": sum(1 for u in st["units"] if u.get("alive", True)),
        "occupancy_cells": len(expected),
        "occupancy_version": version,
        "round": int(st.get("round", 1)),
    }


def metrics_tactics(state: Any = None, **opts: Any) -> Dict[str, Any]:
    """汇总战术运行指标：单位、占位、ZOC 覆盖、预览新鲜度、反击链。"""
    st = _ensure_state(state if state is not None else 0)
    alive = [u for u in st["units"] if u.get("alive", True)]
    all_cells = [
        (x, y)
        for y in range(st["grid"]["height"])
        for x in range(st["grid"]["width"])
        if _is_passable(st["grid"], (x, y))
    ]
    zoc_cover: Dict[str, int] = {}
    for team in sorted({int(u["team"]) for u in st["units"]}):
        zoc_cover[str(team)] = len(_enemy_zoc_cells(st, team) & set(all_cells))

    version = int(st.get("occupancy_version", 1))
    stale = [p for p in st.get("previews", [])
             if int(p.get("occupancy_version", 0)) < version]

    return {
        "units_total": len(st["units"]),
        "units_alive": len(alive),
        "teams": sorted({int(u["team"]) for u in st["units"]}),
        "grid_cells": st["grid"]["width"] * st["grid"]["height"],
        "passable_cells": len(all_cells),
        "occupancy_cells": len(st["occupancy"]),
        "occupancy_version": version,
        "occupancy_conflicts": len(st.get("occupancy_conflicts", [])),
        "zoc_coverage": zoc_cover,
        "preview_count": len(st.get("previews", [])),
        "stale_previews": len(stale),
        "round": int(st.get("round", 1)),
        "seed": int(st.get("seed", 0)),
        "rng_state": int(st.get("rng_state", 0)),
        "log_entries": len(st.get("log", [])),
    }


def _audit_probe_zoc() -> Dict[str, Any]:
    """ZOC：同一格被两个敌人控制、斜向进入只扣一次。"""
    rows = ["######", "#....#", "#....#", "#....#", "######"]
    units = [
        {"id": "hero", "team": 0, "x": 1, "y": 2, "hp": 10, "move": 6},
        {"id": "e1", "team": 1, "x": 3, "y": 1, "hp": 10},
        {"id": "e2", "team": 1, "x": 1, "y": 1, "hp": 10},
    ]
    st = _new_state(rows=rows, units=units)
    # (2,2) 同时被两个敌人控制；斜向进入仍是单次 ZOC 费
    report = zoc_path_cost([(1, 2), (2, 2)], state=st, team=0)
    checks = {
        "zoc_charged_once": report["zoc_cost"] == DEFAULT_ZOC_EXTRA,
        "base_cost_positive": report["base_cost"] > 0,
        "zoc_cells_single": len(report["zoc_cells"]) == 1,
    }
    return {"ok": all(checks.values()), "checks": checks, "report": report}


def _audit_probe_chain() -> Dict[str, Any]:
    """反击链：互相反击有界、同一单位只反击一次、低血量不再反击。"""
    units = [
        _normalise_unit({"id": "a", "team": 0, "x": 1, "y": 1, "hp": 40, "max_hp": 40,
                         "atk": 5, "def": 1}),
        _normalise_unit({"id": "b", "team": 1, "x": 2, "y": 1, "hp": 40, "max_hp": 40,
                         "atk": 5, "def": 1}),
    ]
    report = _resolve_chain(units, ["a"], "b", _RNG(7), max_depth=8)
    counter_ids = [e["attacker"] for e in report["counters"]]
    checks = {
        "bounded": report["depth_reached"] <= 2,
        "no_repeat_counter": len(counter_ids) == len(set(counter_ids)),
        "terminated": report["depth_reached"] < 8,
    }
    low = [
        _normalise_unit({"id": "a", "team": 0, "x": 1, "y": 1, "hp": 40, "max_hp": 40,
                         "atk": 5, "def": 1}),
        _normalise_unit({"id": "b", "team": 1, "x": 2, "y": 1, "hp": 4, "max_hp": 40,
                         "atk": 5, "def": 8}),
    ]
    low_report = _resolve_chain(low, ["a"], "b", _RNG(7), max_depth=8, low_hp_ratio=0.25)
    checks["low_hp_no_counter"] = not low_report["counters"]
    return {"ok": all(checks.values()), "checks": checks}


def _audit_probe_barrier() -> Dict[str, Any]:
    report = turn_barrier(_demo_turn_messages(1))
    round2 = [m for m in report["committed"] if m["round"] == 2]
    checks = {
        "no_violations": not report["violations"],
        "counter_stays_in_round_1": report["counter_rounds"] == [1],
        "barrier_includes_counter": report["round_boundaries"].get("1", 0) >= 1.3,
        "round2_waited_for_barrier": all(
            m["applied_at"] >= report["round_boundaries"]["1"] for m in round2),
        "nothing_deferred": report["deferred_count"] == 0,
    }
    return {"ok": all(checks.values()), "checks": checks}


def _audit_probe_occupancy() -> Dict[str, Any]:
    rows = ["#######", "#.....#", "#.....#", "#.....#", "#######"]
    units = [
        {"id": "big", "team": 0, "x": 2, "y": 2, "hp": 20, "footprint": 2},
        {"id": "small", "team": 1, "x": 4, "y": 2, "hp": 10},
    ]
    st = _new_state(rows=rows, units=units)
    before = set(st["occupancy"])
    preview = preview_move(st, "small", (1, 2))
    st["previews"].append(preview)
    result = occupancy_update(st, "big", to=(3, 2))
    after = set(st["occupancy"])
    checks = {
        "version_bumped": result["occupancy_version"] > result["previous_version"],
        "preview_marked_stale": all(p.get("stale") for p in st["previews"]),
        "old_cells_released": all(c in after for c in before - set(_footprint_cells(
            _find_unit(st["units"], "big")))),
        "new_cell_occupied": all(c in after for c in _footprint_cells(
            _find_unit(st["units"], "big"))),
        "no_conflicts": result["healthy"],
    }
    return {"ok": all(checks.values()), "checks": checks, "result": {
        "released": result["released"], "occupied": result["occupied"],
    }}


def _audit_probe_seed() -> Dict[str, Any]:
    battle = _demo_battle(3)
    a = battle_seed_carry(battle, spectator_seed=111)
    b = battle_seed_carry(battle, spectator_seed=999)
    checks = {
        "same_digest": a["digest"] == b["digest"],
        "same_casualties": a["casualties"] == b["casualties"],
        "local_ignored": a["local_seed_ignored"] and b["local_seed_ignored"],
        "consistent": a["consistent"] and b["consistent"],
    }
    return {"ok": all(checks.values()), "checks": checks}


def _audit_probe_preview() -> Dict[str, Any]:
    attack = preview_resolve_consistent(1)
    move = preview_resolve_consistent(1, mode="move")
    checks = {
        "attack_consistent": attack["consistent"],
        "move_consistent": move["consistent"],
    }
    return {"ok": all(checks.values()), "checks": checks}


def audit_tactics(state: Any = None, **opts: Any) -> Dict[str, Any]:
    """运行六类问题的运行期自检，返回健康度与逐项结论。"""
    probes = {
        "zoc_path_cost": _audit_probe_zoc(),
        "counter_chain": _audit_probe_chain(),
        "turn_barrier": _audit_probe_barrier(),
        "occupancy": _audit_probe_occupancy(),
        "battle_seed": _audit_probe_seed(),
        "preview_consistency": _audit_probe_preview(),
    }
    findings = [name for name, probe in probes.items() if not probe["ok"]]
    state_report = validate_tactics(state if state is not None else 0)
    if not state_report["valid"]:
        findings.extend(state_report["issues"])
    return {
        "healthy": not findings,
        "findings": findings,
        "probes": {name: probe["checks"] for name, probe in probes.items()},
        "state": state_report,
        "checked": len(probes),
    }


def fallback_tactics(state: Any = None, **opts: Any) -> Dict[str, Any]:
    """状态损坏时的降级：冻结在最后一致快照，占位按权威坐标重建。

    不信任传入的占位表与过期预览：从单位坐标重建占位、清空 stale 预览、
    把血量夹到合法区间，并标记 ``degraded``。
    """
    try:
        st = _ensure_state(state if state is not None else 0)
    except (TypeError, ValueError) as exc:
        return {
            "degraded": True,
            "reason": f"state_unreadable:{exc}",
            "state": _new_state(seed=0),
            "occupancy_version": 1,
            "frozen": True,
        }

    reasons: List[str] = []
    for unit in st["units"]:
        hp = max(0, min(int(unit["hp"]), int(unit["max_hp"])))
        if hp != int(unit["hp"]):
            reasons.append(f"clamped_hp:{unit['id']}")
        unit["hp"] = hp
        unit["alive"] = hp > 0
        if not _is_passable(st["grid"], (int(unit["x"]), int(unit["y"]))):
            moved, _ = _place_knockback(st, unit, None, 1, 0)
            unit["x"], unit["y"] = moved
            reasons.append(f"moved_to_passable:{unit['id']}")

    occupancy, conflicts = _build_occupancy(st["units"], st["grid"])
    if conflicts:
        reasons.extend(f"conflict:{c['cell']}" for c in conflicts)
    st["occupancy"] = occupancy
    st["occupancy_version"] = int(st.get("occupancy_version", 1)) + 1
    st["occupancy_conflicts"] = conflicts
    for preview in st.get("previews", []):
        preview["stale"] = True

    st["rng_state"] = int(st.get("rng_state", st.get("seed", 0)))
    return {
        "degraded": bool(reasons),
        "reason": reasons[0] if reasons else "nominal",
        "reasons": reasons,
        "frozen": True,
        "state": _public_state(st),
        "occupancy_version": st["occupancy_version"],
        "occupancy": {_cell_key(c): uid for c, uid in sorted(occupancy.items())},
        "healthy": not conflicts,
    }


def determinism_tactics(seed: Any = 0, **opts: Any) -> Dict[str, Any]:
    """确定性自检：同一种子跑两遍完整管线，摘要必须逐字节一致。"""
    seed_int = int(seed)

    def run() -> Dict[str, Any]:
        st = _new_state(seed=seed_int)
        _begin_round(st, 1)
        move = preview_move(st, "scout", (9, 4))
        st["previews"].append(move)
        knock = occupancy_update(st, "guard", to=(5, 7))
        attack = resolve_attack(st, attackers=["raider"], defender="scout")
        chain = counter_chain_depth(st, attackers=["raider"], defender="scout")
        barrier = turn_barrier(_demo_turn_messages(seed_int or 1))
        carried = battle_seed_carry(_demo_battle(seed_int), spectator_seed=seed_int + 1)
        consistent = preview_resolve_consistent(st)
        return {
            "state": _state_signature(st),
            "move": {k: move[k] for k in ("path", "cost", "zoc_cost")},
            "knockback": {k: knock[k] for k in ("final", "released", "occupied")},
            "attack": {k: attack[k] for k in ("events", "kills", "hp")},
            "chain": {k: chain[k] for k in ("events", "depth_reached", "hp")},
            "barrier": barrier["round_boundaries"],
            "carry": carried["digest"],
            "consistent": consistent["consistent"],
        }

    first = run()
    second = run()
    digest_a = _digest(first)
    digest_b = _digest(second)
    return {
        "deterministic": digest_a == digest_b,
        "seed": seed_int,
        "signature": digest_a,
        "signature_again": digest_b,
        "runs": 2,
        "snapshot": first,
    }
