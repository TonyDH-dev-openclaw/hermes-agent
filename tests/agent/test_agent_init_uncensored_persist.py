import json
from agent.agent_init import _uncensored_mode_persist_disabled


def test_returns_false_when_state_file_missing(tmp_path):
    assert _uncensored_mode_persist_disabled(
        "default", state_path=tmp_path / "mode-state.json",
    ) is False


def test_returns_true_when_profile_is_a_key_in_the_backup_and_kind_is_uncensored(tmp_path):
    state_path = tmp_path / "mode-state.json"
    state_path.write_text(json.dumps({
        "kind": "uncensored",
        "model": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
        "backup": {
            "default": {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"},
            "researcher": {"model": "claude-sonnet-5", "provider": "anthropic"},
        },
    }))
    assert _uncensored_mode_persist_disabled("default", state_path=state_path) is True
    assert _uncensored_mode_persist_disabled("researcher", state_path=state_path) is True


def test_returns_false_for_a_profile_not_in_the_backup(tmp_path):
    state_path = tmp_path / "mode-state.json"
    state_path.write_text(json.dumps({
        "kind": "uncensored", "model": "x",
        "backup": {"default": {"model": "x", "provider": "y"}},
    }))
    assert _uncensored_mode_persist_disabled("reviewer", state_path=state_path) is False


def test_returns_false_when_kind_is_local_even_if_profile_is_in_backup(tmp_path):
    # Regression: /mode local must NOT suppress persistence -- only
    # uncensored carries the "no trace" guarantee. Before this fix, the
    # old file (uncensored-mode-state.json) could ONLY ever contain
    # uncensored's own backup, so this ambiguity didn't exist; now that
    # local and uncensored share one file, kind must be checked, not just
    # profile membership in backup.
    state_path = tmp_path / "mode-state.json"
    state_path.write_text(json.dumps({
        "kind": "local", "model": "qwen/qwen3.5-9b",
        "backup": {"default": {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"}},
    }))
    assert _uncensored_mode_persist_disabled("default", state_path=state_path) is False


def test_fails_safe_to_false_on_malformed_state_file(tmp_path):
    # A malformed/partially-written state file must never crash session
    # init, and must never accidentally suppress persistence for a
    # legitimate session -- fail toward "persist normally", not toward
    # "silently go private".
    state_path = tmp_path / "mode-state.json"
    state_path.write_text("not valid json{{{")
    assert _uncensored_mode_persist_disabled("default", state_path=state_path) is False
