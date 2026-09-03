"""Tests for the post-delegation model restore (2026-09-03).

delegation.model (config.yaml) is a single hardcoded model sharing the same
8GB LM Studio instance as the main conversation's own local/uncensored
model. Loading it via gpu-slot.sh evicts whichever model the parent
conversation had loaded (LM Studio's own unloadPreviousJITModelOnLoad=true),
so a delegate_task call mid-conversation silently unloads the parent's
active model. This restores it synchronously right after the delegation
batch finishes, instead of waiting on gpu-contention-watch.sh's ~1min
periodic reconciliation.
"""

import tools.delegate_tool as dt


class _FakeParent:
    def __init__(self, model):
        self.model = model


def test_restores_parent_model_when_delegation_used_a_different_model():
    """Local mode: parent=qwen, delegation=uncensored -- a real eviction."""
    parent = _FakeParent(model="qwen/qwen3.5-9b")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class _Result:
            stdout = ""
        return _Result()

    dt._restore_parent_model_after_delegation(
        parent,
        {"model": "qwen3.5-9b-uncensored-hauhaucs-aggressive"},
        run=fake_run,
    )

    assert calls == [[dt._GPU_SLOT_SCRIPT, "use", "qwen"]]


def test_no_restore_when_delegation_used_the_same_model_as_parent():
    """Uncensored mode: parent == delegation.model already -- no eviction
    occurred, so restoring would just be a wasted round-trip."""
    parent = _FakeParent(model="qwen3.5-9b-uncensored-hauhaucs-aggressive")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return None

    dt._restore_parent_model_after_delegation(
        parent,
        {"model": "qwen3.5-9b-uncensored-hauhaucs-aggressive"},
        run=fake_run,
    )

    assert calls == []


def test_no_restore_when_delegation_has_no_model_configured():
    parent = _FakeParent(model="qwen/qwen3.5-9b")
    calls = []

    dt._restore_parent_model_after_delegation(
        parent, {}, run=lambda cmd, **kw: calls.append(cmd),
    )

    assert calls == []


def test_no_restore_when_parent_model_is_not_a_known_gpu_slot_role():
    """A cloud-mode parent (e.g. claude-*) has nothing local to restore."""
    parent = _FakeParent(model="claude-sonnet-5")
    calls = []

    dt._restore_parent_model_after_delegation(
        parent,
        {"model": "qwen3.5-9b-uncensored-hauhaucs-aggressive"},
        run=lambda cmd, **kw: calls.append(cmd),
    )

    assert calls == []


def test_restore_swallows_gpu_slot_failures():
    """A failed reload must never raise into the caller -- the delegation
    batch itself already completed successfully by the time this runs."""
    parent = _FakeParent(model="qwen/qwen3.5-9b")

    def fake_run(cmd, **kwargs):
        raise RuntimeError("gpu wedged")

    dt._restore_parent_model_after_delegation(
        parent,
        {"model": "qwen3.5-9b-uncensored-hauhaucs-aggressive"},
        run=fake_run,
    )  # must not raise


def test_finalize_child_results_triggers_restore_when_models_differ(monkeypatch):
    """Integration point: _finalize_child_results (called once per
    delegation batch, both the single-task and batch/async paths) must
    invoke the restore after applying its existing summary/memory/hook/cost
    contracts."""
    parent = _FakeParent(model="qwen/qwen3.5-9b")
    calls = []
    monkeypatch.setattr(
        dt,
        "_restore_parent_model_after_delegation",
        lambda p, cfg, **kw: calls.append((p, cfg)),
    )

    dt._finalize_child_results(
        results=[{"task_index": 0, "summary": "done", "status": "completed"}],
        task_list=[{"goal": "do it"}],
        children=[(0, {"goal": "do it"}, None)],
        parent_agent=parent,
        cfg={"model": "qwen3.5-9b-uncensored-hauhaucs-aggressive"},
    )

    assert calls == [(parent, {"model": "qwen3.5-9b-uncensored-hauhaucs-aggressive"})]
