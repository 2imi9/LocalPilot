"""
Unit test: Devstral code refiner / open-value proposal -> train.py edit.

For V4, the pipeline is:
  1. Qwen proposes PARAM + exact VALUE (informed by research)
  2. Value is clamped to safe bounds
  3. Old line in train.py is found and replaced with new value

Tests:
  1. Open value proposal produces correct old/new line pairs
  2. Values are clamped to bounds
  3. ADAM_BETAS tuple handling
  4. Discrete params snap to nearest allowed value
  5. Comment preservation
  6. Edge case: param not found in train.py
"""

import os
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Import parameter bounds from V3 runner
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# Copy bounds here to avoid import side effects
SAFE_CONTINUOUS = {
    "EMBEDDING_LR":   {"min": 0.1,  "max": 2.0,   "fmt": "{:.1f}"},
    "UNEMBEDDING_LR": {"min": 0.001,"max": 0.02,   "fmt": "{:.4f}"},
    "MATRIX_LR":      {"min": 0.01, "max": 0.12,   "fmt": "{:.2f}"},
    "SCALAR_LR":      {"min": 0.1,  "max": 1.5,    "fmt": "{:.1f}"},
    "WEIGHT_DECAY":   {"min": 0.01, "max": 0.4,    "fmt": "{:.2f}"},
    "WARMUP_RATIO":   {"min": 0.0,  "max": 0.3,    "fmt": "{:.2f}"},
    "WARMDOWN_RATIO": {"min": 0.2,  "max": 1.0,    "fmt": "{:.1f}"},
    "FINAL_LR_FRAC":  {"min": 0.0,  "max": 0.1,    "fmt": "{:.3f}"},
}

SAFE_DISCRETE = {
    "DEPTH":            [4, 6, 8, 10],
    "ASPECT_RATIO":     [32, 48, 64, 80, 96],
    "HEAD_DIM":         [48, 64, 96, 128],
    "TOTAL_BATCH_SIZE": ["2**17", "2**18", "2**19"],
    "WINDOW_PATTERN":   ['"SSSL"', '"SSLL"', '"SLLL"', '"LLLL"', '"SSSS"'],
}

SAFE_ADAM_BETAS = {
    "beta1": {"min": 0.7, "max": 0.95},
    "beta2": {"min": 0.9, "max": 0.999},
}


def clamp_value(param_name: str, proposed_value: str) -> str:
    """
    Clamp a proposed value to safe bounds. Returns the safe value string.

    For continuous params: clamp to [min, max], format per spec.
    For discrete params: snap to nearest allowed value.
    For ADAM_BETAS: clamp each component.
    """
    proposed_value = proposed_value.strip()

    # --- ADAM_BETAS ---
    if param_name == "ADAM_BETAS":
        m = re.match(r"\(([^,]+),\s*([^)]+)\)", proposed_value)
        if not m:
            return proposed_value  # can't parse, pass through
        b1 = float(m.group(1))
        b2 = float(m.group(2))
        b1 = max(SAFE_ADAM_BETAS["beta1"]["min"],
                 min(SAFE_ADAM_BETAS["beta1"]["max"], b1))
        b2 = max(SAFE_ADAM_BETAS["beta2"]["min"],
                 min(SAFE_ADAM_BETAS["beta2"]["max"], b2))
        return f"({round(b1, 3)}, {round(b2, 3)})"

    # --- Discrete ---
    if param_name in SAFE_DISCRETE:
        options = SAFE_DISCRETE[param_name]
        # Try exact match first
        for opt in options:
            if str(opt).strip('"') == proposed_value.strip('"'):
                return str(opt)
        # For numeric discrete, snap to nearest
        try:
            val = float(proposed_value)
            numeric_opts = []
            for opt in options:
                opt_str = str(opt)
                try:
                    # Handle "2**17" style
                    if "**" in opt_str:
                        opt_val = eval(opt_str)
                    else:
                        opt_val = float(opt_str.strip('"'))
                    numeric_opts.append((abs(opt_val - val), opt_str))
                except (ValueError, SyntaxError):
                    pass
            if numeric_opts:
                numeric_opts.sort()
                return numeric_opts[0][1]
        except ValueError:
            pass
        # Default: return first option
        return str(options[0])

    # --- Continuous ---
    if param_name in SAFE_CONTINUOUS:
        spec = SAFE_CONTINUOUS[param_name]
        try:
            val = float(proposed_value)
        except ValueError:
            return proposed_value  # can't parse
        val = max(spec["min"], min(spec["max"], val))
        return spec["fmt"].format(val)

    return proposed_value


def make_edit(param_name: str, proposed_value: str, hp_lines: list[str]) -> dict | None:
    """
    Given a param name and proposed value, produce an edit dict with
    {desc, old, new, param, value}.

    Returns None if param not found or value unchanged.
    """
    # Find the current line
    old_line = None
    for l in hp_lines:
        if l.startswith(param_name + " = ") or l.startswith(param_name + " ="):
            old_line = l
            break

    if old_line is None:
        return None

    # Extract current value and comment
    eq_pos = old_line.index(" = ")
    comment_pos = old_line.find(" # ", eq_pos)
    if comment_pos == -1:
        cur_val_str = old_line[eq_pos + 3:].strip()
        comment_part = ""
    else:
        cur_val_str = old_line[eq_pos + 3:comment_pos].strip()
        comment_part = old_line[comment_pos:]

    # Clamp proposed value to safe bounds
    safe_value = clamp_value(param_name, proposed_value)

    # Build new line
    new_line = old_line[:eq_pos + 3] + safe_value + comment_part

    if new_line == old_line:
        return None  # no change

    return {
        "desc": f"{param_name}={safe_value} (was {cur_val_str})",
        "old": old_line,
        "new": new_line,
        "param": param_name,
        "value": safe_value,
    }


# ===========================================================================
# Tests
# ===========================================================================

SAMPLE_HP_LINES = [
    "DEPTH = 4",
    "EMBEDDING_LR = 0.6",
    "UNEMBEDDING_LR = 0.005",
    "MATRIX_LR = 0.05",
    "SCALAR_LR = 0.5",
    "WEIGHT_DECAY = 0.1",
    "WARMUP_RATIO = 0.0",
    "WARMDOWN_RATIO = 0.8",
    "FINAL_LR_FRAC = 0.033",
    "TOTAL_BATCH_SIZE = 2**18",
    'WINDOW_PATTERN = "SSSL"',
    "ADAM_BETAS = (0.8, 0.95)",
    "HEAD_DIM = 128",
    "ASPECT_RATIO = 48 # width/depth ratio",
]


def test_continuous_edit():
    """Test continuous param edit with value within bounds."""
    print("\n" + "="*60)
    print("TEST 1: Continuous param edit")
    print("="*60)

    result = make_edit("EMBEDDING_LR", "1.2", SAMPLE_HP_LINES)
    assert result is not None, "FAIL: should produce edit"
    assert result["old"] == "EMBEDDING_LR = 0.6"
    assert result["new"] == "EMBEDDING_LR = 1.2"
    assert result["param"] == "EMBEDDING_LR"
    print(f"  {result['old']} -> {result['new']}")

    # Test clamping: value above max
    result2 = make_edit("EMBEDDING_LR", "5.0", SAMPLE_HP_LINES)
    assert result2 is not None
    assert "2.0" in result2["new"], f"FAIL: should clamp to max 2.0, got {result2['new']}"
    print(f"  {result2['old']} -> {result2['new']} (clamped from 5.0)")

    # Test clamping: value below min
    result3 = make_edit("EMBEDDING_LR", "0.01", SAMPLE_HP_LINES)
    assert result3 is not None
    assert "0.1" in result3["new"], f"FAIL: should clamp to min 0.1, got {result3['new']}"
    print(f"  {result3['old']} -> {result3['new']} (clamped from 0.01)")

    print("  PASS")


def test_discrete_edit():
    """Test discrete param edit."""
    print("\n" + "="*60)
    print("TEST 2: Discrete param edit")
    print("="*60)

    result = make_edit("DEPTH", "6", SAMPLE_HP_LINES)
    assert result is not None
    assert result["new"] == "DEPTH = 6"
    print(f"  {result['old']} -> {result['new']}")

    # Value not in list: should snap to nearest
    snapped = clamp_value("DEPTH", "5")
    assert snapped in ["4", "6"], f"FAIL: should snap to 4 or 6, got {snapped}"
    print(f"  DEPTH=5 snapped to {snapped}")

    # Snap to 7 -> 6 (nearest allowed), different from current 4
    result2 = make_edit("DEPTH", "7", SAMPLE_HP_LINES)
    assert result2 is not None
    assert result2["value"] == "6", f"FAIL: DEPTH=7 should snap to 6, got {result2['value']}"
    print(f"  {result2['old']} -> {result2['new']} (snapped from 7)")

    # TOTAL_BATCH_SIZE
    result3 = make_edit("TOTAL_BATCH_SIZE", "2**19", SAMPLE_HP_LINES)
    assert result3 is not None
    assert "2**19" in result3["new"]
    print(f"  {result3['old']} -> {result3['new']}")

    print("  PASS")


def test_adam_betas_edit():
    """Test ADAM_BETAS tuple handling."""
    print("\n" + "="*60)
    print("TEST 3: ADAM_BETAS edit")
    print("="*60)

    result = make_edit("ADAM_BETAS", "(0.9, 0.99)", SAMPLE_HP_LINES)
    assert result is not None
    assert "(0.9, 0.99)" in result["new"]
    print(f"  {result['old']} -> {result['new']}")

    # Clamping: beta1 too high
    result2 = make_edit("ADAM_BETAS", "(0.99, 0.99)", SAMPLE_HP_LINES)
    assert result2 is not None
    assert "0.95" in result2["new"], f"FAIL: beta1 should clamp to 0.95, got {result2['new']}"
    print(f"  (0.99, 0.99) clamped to {result2['value']}")

    # Clamping: beta2 too low
    result3 = make_edit("ADAM_BETAS", "(0.8, 0.5)", SAMPLE_HP_LINES)
    assert result3 is not None
    assert "0.9" in result3["value"], f"FAIL: beta2 should clamp to 0.9, got {result3['value']}"
    print(f"  (0.8, 0.5) clamped to {result3['value']}")

    print("  PASS")


def test_comment_preservation():
    """Test that inline comments are preserved."""
    print("\n" + "="*60)
    print("TEST 4: Comment preservation")
    print("="*60)

    result = make_edit("ASPECT_RATIO", "64", SAMPLE_HP_LINES)
    assert result is not None
    assert "# width/depth ratio" in result["new"], \
        f"FAIL: comment lost in {result['new']}"
    print(f"  {result['old']} -> {result['new']}")
    print("  PASS")


def test_no_change():
    """Test that same value returns None."""
    print("\n" + "="*60)
    print("TEST 5: No change (same value)")
    print("="*60)

    result = make_edit("EMBEDDING_LR", "0.6", SAMPLE_HP_LINES)
    assert result is None, f"FAIL: should return None for same value, got {result}"
    print("  EMBEDDING_LR=0.6 -> None (no change)")

    result2 = make_edit("DEPTH", "4", SAMPLE_HP_LINES)
    assert result2 is None, f"FAIL: should return None for same value, got {result2}"
    print("  DEPTH=4 -> None (no change)")

    print("  PASS")


def test_param_not_found():
    """Test missing param returns None."""
    print("\n" + "="*60)
    print("TEST 6: Param not found")
    print("="*60)

    result = make_edit("NONEXISTENT_PARAM", "42", SAMPLE_HP_LINES)
    assert result is None, f"FAIL: should return None for missing param"
    print("  NONEXISTENT_PARAM -> None")

    print("  PASS")


def test_real_train_py():
    """Test against actual train.py file."""
    print("\n" + "="*60)
    print("TEST 7: Real train.py")
    print("="*60)

    train_path = ROOT / "train.py"
    if not train_path.exists():
        print("  SKIP: train.py not found")
        return

    content = train_path.read_text(encoding="utf-8")
    hp_lines = [l for l in content.splitlines()
                if re.match(r'^[A-Z_]+ = ', l)][:20]

    print(f"  Found {len(hp_lines)} HP lines in train.py")
    for l in hp_lines[:5]:
        print(f"    {l}")

    # Try editing a real param
    for param in ["EMBEDDING_LR", "WEIGHT_DECAY", "SCALAR_LR"]:
        for l in hp_lines:
            if l.startswith(param + " = "):
                result = make_edit(param, "0.42", hp_lines)
                if result:
                    print(f"  Edit: {result['old'][:50]} -> {result['new'][:50]}")
                    # Verify old line exists in actual file
                    assert result["old"] in content, \
                        f"FAIL: old line not found in train.py: {result['old']}"
                break

    print("  PASS")


if __name__ == "__main__":
    print("Devstral Code Refiner / Open Value Edit Tests")
    print("="*60)

    passed = 0
    failed = 0

    for test_fn in [test_continuous_edit, test_discrete_edit, test_adam_betas_edit,
                    test_comment_preservation, test_no_change, test_param_not_found,
                    test_real_train_py]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
