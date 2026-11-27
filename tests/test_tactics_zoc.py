import pytest
import game.tactics_zoc as mod

CASES = [
    ("zoc_path_cost", (1,)),
    ("counter_chain_depth", (1,)),
    ("turn_barrier", (1,)),
    ("occupancy_update", (1,)),
    ("battle_seed_carry", (1,)),
    ("preview_resolve_consistent", (1,)),
    ("load_tactics", (1,)),
    ("preview_move", (1,)),
    ("resolve_attack", (1,)),
    ("present_tactics", (1,)),
    ("validate_tactics", (1,)),
    ("metrics_tactics", (1,)),
    ("audit_tactics", (1,)),
    ("fallback_tactics", (1,)),
    ("determinism_tactics", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
