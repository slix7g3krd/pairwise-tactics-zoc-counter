"""六类修复的行为级测试：验证问题真的被修掉，而不只是返回非 None。"""

import json

import pytest

import game.tactics_zoc as tz
from game.tactics_zoc import (
    DEFAULT_CHAIN_DEPTH,
    DEFAULT_ZOC_EXTRA,
    battle_seed_carry,
    counter_chain_depth,
    determinism_tactics,
    occupancy_update,
    preview_move,
    preview_resolve_consistent,
    resolve_attack,
    turn_barrier,
    validate_tactics,
    zoc_path_cost,
)


def state_with(rows, units, seed=0):
    return tz._new_state(seed=seed, rows=rows, units=units)


def unit(uid, team, x, y, **kw):
    base = {"id": uid, "team": team, "x": x, "y": y, "hp": 20, "max_hp": 20,
            "atk": 5, "def": 1, "move": 8}
    base.update(kw)
    return base


# --- 修复 1：ZOC 路径代价 ---------------------------------------------------

class TestZocPathCost:
    def test_entering_tile_controlled_by_two_enemies_is_charged_once(self):
        rows = ["######", "#....#", "#....#", "#....#", "######"]
        st = state_with(rows, [
            unit("h", 0, 1, 2),
            unit("e1", 1, 3, 1),
            unit("e2", 1, 1, 1),
        ])
        # (2,2) 同时被 e1/e2 控制
        report = zoc_path_cost([(1, 2), (2, 2)], state=st, team=0)
        assert report["zoc_cost"] == pytest.approx(DEFAULT_ZOC_EXTRA)
        assert len(report["zoc_cells"]) == 1

    def test_diagonal_entry_charged_once_not_twice(self):
        rows = ["######", "#....#", "#....#", "#....#", "######"]
        st = state_with(rows, [
            unit("h", 0, 1, 2),
            unit("e1", 1, 2, 1),
            unit("e2", 1, 1, 1),
        ])
        report = zoc_path_cost([(1, 2), (2, 2)], state=st, team=0)
        assert len(report["steps"]) == 1
        assert report["steps"][0]["zoc"] == pytest.approx(DEFAULT_ZOC_EXTRA)
        assert report["zoc_cost"] == pytest.approx(DEFAULT_ZOC_EXTRA)

    def test_extra_only_charged_on_entry_not_on_exit(self):
        rows = ["#######", "#.....#", "#.....#", "#.....#", "#######"]
        st = state_with(rows, [
            unit("h", 0, 1, 2),
            unit("e", 1, 3, 3),
        ])
        # (2,2)(3,2)(4,2) 都在 ZOC 内，但每格只因“进入”计费一次；
        # 离开控制区本身不再收费。
        report = zoc_path_cost([(1, 2), (2, 2), (3, 2), (4, 2)], state=st, team=0)
        assert report["zoc_cost"] == pytest.approx(3 * DEFAULT_ZOC_EXTRA)
        assert report["steps"][-1]["to"] == [4, 2]
        # 在控制区内横向移动不会让同一格重复计费
        repeated = zoc_path_cost([(1, 2), (2, 2), (2, 3), (2, 2)], state=st, team=0)
        assert repeated["zoc_cost"] == pytest.approx(2 * DEFAULT_ZOC_EXTRA)

    def test_pathfinding_prefers_cheaper_route_into_zoc(self):
        rows = [
            "##########",
            "#........#",
            "#........#",
            "#.e......#",
            "#........#",
            "#........#",
            "##########",
        ]
        st = state_with(rows, [unit("h", 0, 1, 3, move=20),
                               unit("e", 1, 4, 4)])
        with_zoc = tz._plan_move(st, tz._find_unit(st["units"], "h"), (7, 3))
        without = tz._plan_move(st, tz._find_unit(st["units"], "h"), (7, 3),
                                zoc_extra=0.0)
        # 有 ZOC 时不应沿着经过敌人身边的直线走：
        # 代价更高（因为绕路），但完全避开控制区
        assert without["zoc_cost"] == pytest.approx(0.0)
        assert with_zoc["cost"] > without["cost"]
        assert with_zoc["zoc_cost"] == pytest.approx(0.0)
        assert with_zoc["path"] != without["path"]

    def test_front_crossing_is_more_expensive_than_detour(self):
        rows = [
            "##########",
            "#........#",
            "#........#",
            "#........#",
            "#........#",
            "#........#",
            "##########",
        ]
        # 敌人排在直线下方一格：正面硬穿会连续进入三格控制区
        wall = [unit(f"e{i}", 1, 4, 4) for i in range(1)]
        st = state_with(rows, [unit("h", 0, 1, 3, move=30)] + wall)
        plan = tz._plan_move(st, tz._find_unit(st["units"], "h"), (7, 3))
        straight = zoc_path_cost([(1, 3), (2, 3), (3, 3), (4, 3), (5, 3),
                                  (6, 3), (7, 3)], state=st, team=0)
        assert straight["feasible"] is True
        assert straight["zoc_cost"] > 0
        assert plan["cost"] < straight["cost"]
        # 绕后路线穿过控制区的次数更少
        assert plan["zoc_cost"] < straight["zoc_cost"]

    def test_impassable_and_occupied_steps_are_infeasible(self):
        rows = ["#####", "#.#.#", "#...#", "#####"]
        st = state_with(rows, [unit("h", 0, 1, 2), unit("b", 1, 3, 2)])
        report = zoc_path_cost([(1, 2), (2, 2), (3, 2)], state=st, team=0)
        assert report["feasible"] is False


# --- 修复 2：反击链深度与去重 ----------------------------------------------

class TestCounterChain:
    def _duel(self, hp_a=40, hp_b=40, **kw):
        st = tz._new_state(units=[
            unit("a", 0, 1, 1, hp=hp_a, max_hp=hp_a, **kw),
            unit("b", 1, 2, 1, hp=hp_b, max_hp=hp_b, **kw),
        ])
        return st

    def test_mutual_counter_terminates_after_one_round_trip(self):
        st = self._duel()
        report = counter_chain_depth(st, attackers=["a"], defender="b", max_depth=64)
        assert report["depth_reached"] == 2
        assert [e["kind"] for e in report["events"]] == ["attack", "counter", "counter"]
        assert report["truncated"] is False

    def test_depth_cap_is_honoured(self):
        st = self._duel()
        report = counter_chain_depth(st, attackers=["a"], defender="b", max_depth=1)
        assert report["depth_reached"] == 1
        assert report["truncated"] is True
        assert all(e["depth"] <= 1 for e in report["events"])

    def test_zero_depth_disables_counters(self):
        st = self._duel()
        report = counter_chain_depth(st, attackers=["a"], defender="b", max_depth=0)
        assert report["counters"] == []

    def test_unit_never_counters_twice(self):
        st = tz._new_state(units=[
            unit("a", 0, 1, 1), unit("b", 0, 1, 2),
            unit("c", 1, 2, 1, hp=200, max_hp=200),
        ])
        report = counter_chain_depth(st, attackers=["a", "b"], defender="c",
                                     max_depth=64)
        attackers_of_counters = [e["attacker"] for e in report["counters"]]
        assert len(attackers_of_counters) == len(set(attackers_of_counters))

    def test_low_hp_unit_is_not_repeatedly_countered(self):
        st = tz._new_state(units=[
            unit("a", 0, 1, 1, atk=9),
            unit("b", 1, 2, 1, hp=4, max_hp=40),
        ])
        report = counter_chain_depth(st, attackers=["a"], defender="b", max_depth=64)
        struck = [e["defender"] for e in report["counters"]]
        assert struck.count("b") <= 1
        assert report["counters"] == []

    def test_attackers_resolved_in_list_order(self):
        st = tz._new_state(units=[
            unit("a", 0, 1, 1), unit("b", 0, 1, 2),
            unit("t", 1, 2, 1, hp=999, max_hp=999),
        ])
        report = counter_chain_depth(st, attackers=["b", "a"], defender="t")
        assert report["order"] == ["b", "a"]

    def test_self_targeting_is_rejected(self):
        st = tz._new_state(seed=0)
        report = counter_chain_depth(st, attackers=["scout"], defender="scout")
        assert report["events"] == []
        assert any(s["reason"] == "self_target" for s in report["skipped"])

    def test_dead_attacker_is_skipped(self):
        st = tz._new_state(units=[
            unit("a", 0, 1, 1, hp=0),
            unit("t", 1, 2, 1),
        ])
        report = counter_chain_depth(st, attackers=["a"], defender="t")
        assert report["events"] == []
        assert any(s["reason"] == "dead_before_attack" for s in report["skipped"])


# --- 修复 3：回合屏障 -------------------------------------------------------

class TestTurnBarrier:
    def test_late_counter_stays_in_original_round(self):
        report = turn_barrier([
            {"round": 1, "kind": "end_turn", "team": 0, "time": 1.0},
            {"round": 1, "kind": "end_turn", "team": 1, "time": 1.1},
            {"round": 2, "kind": "move", "team": 0, "time": 1.15},
            {"round": 1, "kind": "counter", "team": 1, "time": 1.3},
        ])
        assert report["counter_rounds"] == [1]
        assert report["round_boundaries"]["1"] >= 1.3

    def test_next_round_messages_are_held_until_barrier(self):
        # 回合 2 的移动先于回合 1 的反击到达：在回合 1 关闭前不得生效
        report = turn_barrier([
            {"round": 1, "kind": "end_turn", "team": 0, "time": 1.0},
            {"round": 1, "kind": "counter", "team": 1, "time": 1.4},
            {"round": 2, "kind": "move", "team": 0, "time": 1.1},
            {"round": 2, "kind": "end_turn", "team": 0, "time": 1.5},
            {"round": 2, "kind": "end_turn", "team": 1, "time": 1.6},
        ])
        held = [e for e in report["committed"] if e["round"] == 2]
        assert held and all(e["applied_at"] >= 1.6 for e in held)
        # 早到的回合 2 移动被屏障挡住，直到回合 1 关闭才生效
        early = next(e for e in held if e["time"] == 1.1)
        assert early["applied_at"] > early["time"]
        assert report["counter_rounds"] == [1]
        # 回合 1 的反击在回合 2 任何消息生效之前就已结算
        counter_at = next(e["applied_at"] for e in report["committed"]
                          if e["kind"] == "counter")
        assert counter_at <= min(e["applied_at"] for e in held)

    def test_unclosed_round_keeps_later_messages_deferred(self):
        report = turn_barrier([
            {"round": 1, "kind": "end_turn", "team": 0, "time": 1.0},
            {"round": 2, "kind": "move", "team": 1, "time": 1.1},
        ])
        assert report["closed_rounds"] == [1]
        assert report["deferred_count"] == 1
        assert report["deferred"][0]["round"] == 2

    def test_round_stays_open_without_all_submissions(self):
        report = turn_barrier([
            {"round": 1, "kind": "end_turn", "team": 0, "time": 1.0},
        ], total_teams=2)
        assert report["closed_rounds"] == []
        assert report["pending_rounds"] == [1]

    def test_timeout_closure_flags_late_messages(self):
        report = turn_barrier([
            {"round": 1, "kind": "end_turn", "team": 0, "time": 1.0},
            {"round": 1, "kind": "counter", "team": 1, "time": 5.0},
        ], total_teams=2, timeout=1.5)
        assert report["closed_rounds"] == [1]
        assert report["violations"]
        assert report["violations"][0]["reason"] == "late_after_barrier"
        assert report["counter_rounds"] == []

    def test_barrier_time_includes_late_counter(self):
        report = turn_barrier([
            {"round": 1, "kind": "end_turn", "team": 0, "time": 1.0},
            {"round": 1, "kind": "end_turn", "team": 1, "time": 1.0},
            {"round": 1, "kind": "counter", "team": 1, "time": 2.4},
        ])
        assert report["round_boundaries"]["1"] == pytest.approx(2.4)


# --- 修复 4：占位更新 -------------------------------------------------------

class TestOccupancyUpdate:
    def _state(self):
        rows = ["#######", "#.....#", "#.....#", "#.....#", "#.....#", "#######"]
        return state_with(rows, [
            unit("big", 0, 1, 1, footprint=2),
            unit("small", 1, 4, 2),
        ])

    def test_knockback_rebuilds_footprint_occupancy(self):
        st = self._state()
        before = dict(st["occupancy"])
        result = occupancy_update(st, "big", to=(2, 1))
        assert result["occupancy_version"] == 2
        big = tz._find_unit(st["units"], "big")
        assert len(tz._footprint_cells(big)) == 4
        for cell in tz._footprint_cells(big):
            assert st["occupancy"][cell] == "big"
        # 旧占位里被腾出的格子必须释放
        for cell in result["released"]:
            assert tuple(cell) not in st["occupancy"]
        assert set(before) - {tuple(c) for c in result["occupied"]}
        # 体积占用的每一格都更新到新位置
        assert {(2, 1), (3, 1), (2, 2), (3, 2)} == set(tz._footprint_cells(big))

    def test_other_units_cannot_pass_through_new_cells(self):
        st = self._state()
        occupancy_update(st, "big", to=(2, 1))
        plan = tz._plan_move(st, tz._find_unit(st["units"], "small"), (1, 1))
        assert (2, 1) not in plan["path"]
        assert (2, 2) not in plan["path"]
        assert plan["feasible"] is False

    def test_stale_preview_is_marked_and_recomputed(self):
        st = self._state()
        preview = preview_move(st, "small", (1, 3), record=True)
        assert preview["occupancy_version"] == 1
        result = occupancy_update(st, "big", to=(2, 2))
        assert "small" in result["stale_previews"]
        assert st["previews"][0]["stale"] is True
        assert isinstance(st["previews"][0]["path"], list)

    def test_knockback_onto_blocked_cell_slides_to_free_tile(self):
        rows = ["#######", "#..#..#", "#.....#", "#.....#", "#######"]
        st = state_with(rows, [unit("u", 0, 1, 2), unit("wall", 1, 4, 2)])
        result = occupancy_update(st, "u", to=(4, 2), dx=1, dy=0)
        final = tuple(result["final"])
        assert final != (4, 2)
        assert tz._is_passable(st["grid"], final)
        assert tz._find_unit(st["units"], "u")["x"] == final[0]

    def test_occupancy_stays_conflict_free(self):
        st = self._state()
        result = occupancy_update(st, "big", to=(3, 3))
        assert result["healthy"] is True
        assert validate_tactics(st)["valid"] is True


# --- 修复 5：战斗种子携带 ---------------------------------------------------

class TestBattleSeedCarry:
    def test_spectator_local_seed_is_ignored(self):
        battle = tz._demo_battle(3)
        a = battle_seed_carry(battle, spectator_seed=111)
        b = battle_seed_carry(battle, spectator_seed=999)
        assert a["local_seed_ignored"] and b["local_seed_ignored"]
        assert a["digest"] == b["digest"]
        assert a["rolls"] == b["rolls"]

    def test_casualties_identical_for_late_joiners(self):
        battle = tz._demo_battle(11)
        first = battle_seed_carry(battle)
        late = battle_seed_carry(battle, spectator_seed=7)
        assert first["casualties"] == late["casualties"]
        assert late["consistent"] is True

    def test_checkpoint_at_end_yields_no_new_rolls(self):
        battle = tz._demo_battle(5)
        full = battle_seed_carry(battle)
        after = battle_seed_carry(
            {**battle, "checkpoint": {"cursor": full["cursor"], "index": len(battle["checks"])}})
        assert after["rolls"] == []

    def test_checkpoint_midway_matches_suffix_of_full_replay(self):
        battle = tz._demo_battle(9)
        full = battle_seed_carry(battle)
        rng = tz._RNG(battle["seed"])
        for _ in range(2):
            rng.next_u64()
        tail = battle_seed_carry({**battle, "checkpoint": {"cursor": rng.state, "index": 2}})
        assert tail["rolls"] == full["rolls"][2:]
        # 单场战斗的伤亡由战斗携带的种子决定，与检查点无关
        assert tail["casualties"] == full["casualties"]

    def test_record_can_carry_cursor_forward(self):
        battle = tz._demo_battle(6)
        first = battle_seed_carry(battle)
        advanced = battle_seed_carry({**battle, "checkpoint": {
            "cursor": first["cursor"], "index": len(battle["checks"])}})
        assert advanced["rolls"] == []
        assert advanced["cursor"] == first["cursor"]

    def test_different_seeds_differ_in_crit_stream(self):
        a = battle_seed_carry(tz._demo_battle(1))
        b = battle_seed_carry(tz._demo_battle(2))
        assert a["rolls"] != b["rolls"]


# --- 修复 6：预览与结算一致 -------------------------------------------------

class TestPreviewConsistency:
    def test_attack_preview_matches_resolution(self):
        result = preview_resolve_consistent(1)
        assert result["consistent"] is True
        assert result["diff"] == []
        assert result["preview"]["events"] == result["resolved"]["events"]

    def test_attack_preview_repeated_runs_are_identical(self):
        a = preview_resolve_consistent(1)
        b = preview_resolve_consistent(1)
        assert a["digest"] == b["digest"]

    def test_move_preview_matches_replanning(self):
        result = preview_resolve_consistent(1, mode="move")
        assert result["consistent"] is True
        assert result["preview"]["path"] == result["resolved"]["path"]

    def test_preview_does_not_mutate_state(self):
        st = tz._new_state(seed=3)
        before = tz._state_signature(st)
        preview_move(st, "scout", (9, 4))
        preview_resolve_consistent(st)
        assert tz._state_signature(st) == before

    def test_resolve_consumes_seed_and_converges(self):
        st = tz._new_state(seed=4)
        report = resolve_attack(st, attackers=["raider"], defender="scout")
        assert st["rng_state"] == report["seed"]
        # 最终血量等于该单位最后一次被击打后的血量
        last = [e for e in report["events"] if e["defender"] == "scout"][-1]
        assert report["hp"]["scout"] == last["hp_after"]


# --- 管线与确定性 -----------------------------------------------------------

def test_determinism_self_check():
    result = determinism_tactics(42)
    assert result["deterministic"] is True
    again = determinism_tactics(42)
    assert again["signature"] == result["signature"]
    assert determinism_tactics(43)["signature"] != result["signature"]


def test_audit_is_healthy_on_demo_state():
    report = tz.audit_tactics(1)
    assert report["healthy"] is True
    assert report["findings"] == []


def test_fallback_repairs_corrupt_state():
    st = tz._new_state(seed=0)
    st["units"][0]["hp"] = -5
    st["occupancy"] = {("99", "99"): "ghost"}
    result = tz.fallback_tactics(st)
    assert result["degraded"] is True
    assert validate_tactics(result["state"])["valid"] is True


def test_present_and_metrics_are_serialisable():
    payload = tz.present_tactics(1)
    json.dumps(payload)
    metrics = tz.metrics_tactics(1)
    assert metrics["units_alive"] == 5
    assert set(metrics["zoc_coverage"]) == {"0", "1"}
