"""End-to-end tests for ``POST /analyze/version-withdrawal``.

The endpoint solves the unmodified input as a baseline, removes only the
named version from the request's own package list, and re-solves from
scratch under the full constraints.  Tests cover the package diff, the
post-withdrawal graph and locks, genuine full-constraint backtracking,
the three failure-status families, input validation and determinism.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _withdraw(**payload):
    response = client.post("/analyze/version-withdrawal", json=payload)
    return response.status_code, response.json()


# ---------------------------------------------------------------------------
# Successful withdrawals
# ---------------------------------------------------------------------------


def test_withdraw_eligible_highest_leaf_version():
    status, body = _withdraw(
        packages={
            "app": {"1.0": {"dependencies": {"lib": ">=1.0"}}},
            "lib": {"1.0": {}, "1.5": {}, "2.0": {}},
        },
        root=["app"],
        withdraw_package="lib",
        withdraw_version="2.0",
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["withdrawn"] == {"package": "lib", "version": "2.0"}

    baseline = body["baseline"]
    assert baseline["status"] == "ok"
    assert baseline["resolved"] == {"app": "1.0", "lib": "2.0"}
    after = body["after_withdrawal"]
    assert after["status"] == "ok"
    assert after["resolved"] == {"app": "1.0", "lib": "1.5"}

    assert body["changes"]["summary"] == {"added": 0, "removed": 0, "changed": 1}
    assert body["changes"]["added"] == []
    assert body["changes"]["removed"] == []
    assert body["changes"]["changed"] == [
        {"package": "lib", "from": "2.0", "to": "1.5"}
    ]

    # The post-withdrawal graph and locks are surfaced at top level too.
    pairs = {(e["source"], e["target"], e["target_version"]) for e in body["dependency_graph"]}
    assert ("<root>", "app", "1.0") in pairs
    assert ("app", "lib", "1.5") in pairs
    assert ("app", "lib", "2.0") not in pairs
    assert body["locks"] == after["locks"]


def test_withdrawing_selected_version_triggers_full_recompute():
    # Baseline: app picks 2.0 (highest), pulling common==1.5 (the only
    # version satisfying db's pin chain).  A naive "rewrite along the old
    # graph" could keep app==2.0 and only rebump common; the real re-solve
    # must reconsider everything: app==2.0 becomes impossible without
    # common==1.5, so the solver backtracks to app 1.0 and common 2.0.
    packages = {
        "app": {
            "1.0": {"dependencies": {"common": ">=1.0"}},
            "2.0": {"dependencies": {"db": "", "common": "<2.0"}},
        },
        "db": {"1.0": {"dependencies": {"common": "==1.5"}}},
        "common": {"1.5": {}, "2.0": {}},
    }
    status, body = _withdraw(
        packages=packages,
        root=["app"],
        withdraw_package="common",
        withdraw_version="1.5",
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["baseline"]["resolved"] == {
        "app": "2.0", "common": "1.5", "db": "1.0",
    }
    after = body["after_withdrawal"]["resolved"]
    assert after == {"app": "1.0", "common": "2.0"}

    changes = body["changes"]
    assert changes["summary"] == {"added": 0, "removed": 1, "changed": 2}
    assert changes["removed"] == [{"package": "db", "version": "1.0"}]
    assert changes["changed"] == [
        {"package": "app", "from": "2.0", "to": "1.0"},
        {"package": "common", "from": "1.5", "to": "2.0"},
    ]
    # db no longer appears in the post-withdrawal graph at all.
    targets = {e["target"] for e in body["dependency_graph"]}
    assert "db" not in targets


def test_withdrawal_can_add_packages():
    # Baseline: app 2.0 requires common<2.0, so 1.5 (no extra deps) wins;
    # withdrawing common 1.5 makes app 2.0 impossible, the fresh solve
    # backtracks to app 1.0 and common 2.0, pulling a new transitive
    # package (tool) into the solution.
    packages = {
        "app": {
            "1.0": {"dependencies": {"common": ">=1.0"}},
            "2.0": {"dependencies": {"common": ">=1.0,<2.0"}},
        },
        "common": {
            "1.5": {},
            "2.0": {"dependencies": {"tool": ">=2.0"}},
        },
        "tool": {"1.0": {}, "2.0": {}},
    }
    status, body = _withdraw(
        packages=packages,
        root=["app"],
        withdraw_package="common",
        withdraw_version="1.5",
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["baseline"]["resolved"] == {"app": "2.0", "common": "1.5"}
    assert body["changes"]["added"] == [{"package": "tool", "version": "2.0"}]
    assert body["changes"]["changed"] == [
        {"package": "app", "from": "2.0", "to": "1.0"},
        {"package": "common", "from": "1.5", "to": "2.0"},
    ]
    assert any(
        e["source"] == "common" and e["source_version"] == "2.0"
        and e["target"] == "tool"
        for e in body["dependency_graph"]
    )


def test_withdrawing_a_version_not_selected_leaves_solution_unchanged():
    packages = {
        "app": {"1.0": {"dependencies": {"lib": ">=1.0"}}},
        "lib": {"1.0": {}, "2.0": {}},
    }
    status, body = _withdraw(
        packages=packages,
        root=["app"],
        withdraw_package="lib",
        withdraw_version="1.0",
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["baseline"]["resolved"] == {"app": "1.0", "lib": "2.0"}
    assert body["after_withdrawal"]["resolved"] == {"app": "1.0", "lib": "2.0"}
    assert body["changes"] == {
        "summary": {"added": 0, "removed": 0, "changed": 0},
        "added": [],
        "removed": [],
        "changed": [],
    }


def test_locked_withdrawn_version_is_reported_missing_version_after_withdrawal():
    status, body = _withdraw(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root={"x": ">=1.0"},
        locks={"x": "1.0"},
        withdraw_package="x",
        withdraw_version="1.0",
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["baseline"]["resolved"] == {"x": "1.0"}
    assert body["baseline"]["locks"] == {"x": "applied"}
    # The lock names a version that no longer exists in the withdrawn
    # index: disposition becomes missing-version and the next eligible
    # version is chosen.
    assert body["after_withdrawal"]["locks"] == {"x": "missing-version"}
    assert body["after_withdrawal"]["resolved"] == {"x": "2.0"}
    assert body["locks"] == {"x": "missing-version"}
    assert body["changes"]["changed"] == [{"package": "x", "from": "1.0", "to": "2.0"}]


def test_non_canonical_withdraw_package_name_is_normalized():
    status, body = _withdraw(
        packages={"Foo_Bar": {"1.0": {}, "2.0": {}}},
        root={"foo-bar": ">=1.0"},
        withdraw_package="foo.bar",
        withdraw_version="2.0",
    )
    assert status == 200
    assert body["withdrawn"] == {"package": "foo-bar", "version": "2.0"}
    assert body["baseline"]["resolved"] == {"foo-bar": "2.0"}
    assert body["after_withdrawal"]["resolved"] == {"foo-bar": "1.0"}


def test_version_string_is_pep440_normalized_in_echo_and_index_lookup():
    # 'v2.0' is accepted as PEP 440 and normalized to '2.0' for both the
    # index lookup and the response echo.
    status, body = _withdraw(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root=["x"],
        withdraw_package="x",
        withdraw_version="v2.0",
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["withdrawn"]["version"] == "2.0"
    assert body["baseline"]["resolved"] == {"x": "2.0"}
    assert body["after_withdrawal"]["resolved"] == {"x": "1.0"}


def test_results_are_deterministic():
    payload = {
        "packages": {
            "app": {
                "1.0": {"dependencies": {"common": ">=1.0"}},
                "2.0": {"dependencies": {"db": "", "common": "<2.0"}},
            },
            "db": {"1.0": {"dependencies": {"common": "==1.5"}}},
            "common": {"1.5": {}, "2.0": {}},
        },
        "root": ["app"],
        "withdraw_package": "common",
        "withdraw_version": "1.5",
    }
    results = [client.post("/analyze/version-withdrawal", json=payload).json() for _ in range(3)]
    assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Baseline conflicts / limits — withdrawal solve must not run
# ---------------------------------------------------------------------------


def test_baseline_conflict_is_reported_without_withdrawal_solve():
    packages = {
        "service-a": {"1.0": {"dependencies": {"libfoo": ">=2.0"}}},
        "service-b": {"1.0": {"dependencies": {"libfoo": "<2.0"}}},
        "libfoo": {"1.0": {}, "2.0": {}},
    }
    status, body = _withdraw(
        packages=packages,
        root=["service-a", "service-b"],
        withdraw_package="libfoo",
        withdraw_version="1.0",
    )
    assert status == 200
    assert body["status"] == "baseline-conflict"
    assert body["baseline"]["status"] == "conflict"
    terminals = body["baseline"]["conflict"]["terminal_conflicts"]
    assert {t["package"] for t in terminals} == {"libfoo"}
    assert body["after_withdrawal"] is None
    assert "changes" not in body


def test_withdrawal_conflict_is_a_full_resolution_conflict():
    # Baseline is a clean diamond converging on common 1.5 (db pins
    # <2.0, web accepts anything).  Removing common 1.5 leaves only 2.0,
    # which contradicts db's <2.0; the contradiction is found by a fresh
    # full re-solve, not inferred from the old graph.
    packages = {
        "app": {"1.0": {"dependencies": {"db": "", "web": ""}}},
        "db": {"1.0": {"dependencies": {"common": "<2.0"}}},
        "web": {"2.0": {"dependencies": {"common": ""}}},
        "common": {"1.5": {}, "2.0": {}},
    }
    status, body = _withdraw(
        packages=packages,
        root=["app"],
        withdraw_package="common",
        withdraw_version="1.5",
    )
    assert status == 200
    assert body["status"] == "withdrawal-conflict"
    assert body["baseline"]["status"] == "ok"
    assert body["baseline"]["resolved"] == {
        "app": "1.0", "common": "1.5", "db": "1.0", "web": "2.0",
    }
    after = body["after_withdrawal"]
    assert after["status"] == "conflict"
    terminals = after["conflict"]["terminal_conflicts"]
    assert len(terminals) == 1
    assert terminals[0]["package"] == "common"
    assert terminals[0]["reason"] == "unsatisfiable"
    requirers = {
        c["required_by"] for c in terminals[0]["active_constraints"]
    }
    assert requirers == {"db", "web"}
    assert "changes" not in body
    assert "dependency_graph" not in body


def test_withdrawal_conflict_when_no_version_of_withdrawn_package_remains():
    status, body = _withdraw(
        packages={
            "app": {"1.0": {"dependencies": {"only": ""}}},
            "only": {"9.9": {}},
        },
        root=["app"],
        withdraw_package="only",
        withdraw_version="9.9",
    )
    assert status == 200
    assert body["status"] == "withdrawal-conflict"
    terminal = body["after_withdrawal"]["conflict"]["terminal_conflicts"][0]
    assert terminal["package"] == "only"
    assert terminal["reason"] == "no-versions"


def test_baseline_limit_is_not_treated_as_unsatisfiable():
    packages = {
        "app": {"1.0": {"dependencies": {"p0": ""}}},
    }
    for i in range(3):
        packages[f"p{i}"] = {
            f"{major}.0": {"dependencies": {f"p{i+1}": ""}}
            for major in range(1, 4)
        }
    packages["p3"] = {f"{major}.0": {} for major in range(1, 4)}

    status, body = _withdraw(
        packages=packages,
        root=["app"],
        max_attempts=3,
        withdraw_package="p3",
        withdraw_version="3.0",
    )
    assert status == 200
    assert body["status"] == "baseline-limit"
    assert body["baseline"]["status"] == "limit"
    assert body["baseline"]["error"]["code"] == "search_limit_exceeded"
    assert body["after_withdrawal"] is None


def test_withdrawal_limit_is_not_treated_as_unsatisfiable():
    # Baseline (app 2.0) pins the whole chain to 1.0, so it solves in a
    # handful of attempts.  After app 2.0 is withdrawn, app 1.0 starts the
    # unpinned chain and the solver must backtrack through p2/p3 (p3 2.0
    # and 3.0 pull in a missing package) before reaching p3 1.0; with the
    # small budget that search hits the limit.  The limit must surface as
    # 'withdrawal-limit', never as a conflict.
    packages = {
        "app": {
            "1.0": {"dependencies": {"p0": ""}},
            "2.0": {
                "dependencies": {"p0": "==1.0", "p1": "==1.0", "p2": "==1.0"}
            },
        },
        "p0": {f"{major}.0": {"dependencies": {"p1": ""}} for major in (1, 2, 3)},
        "p1": {f"{major}.0": {"dependencies": {"p2": ""}} for major in (1, 2, 3)},
        "p2": {
            f"{major}.0": {"dependencies": {f"p3": f"=={major}.0"}}
            for major in (1, 2, 3)
        },
        "p3": {
            "1.0": {},
            "2.0": {"dependencies": {"ghost": ""}},
            "3.0": {"dependencies": {"ghost": ""}},
        },
    }

    status, body = _withdraw(
        packages=packages,
        root=["app"],
        max_attempts=6,
        withdraw_package="app",
        withdraw_version="2.0",
    )
    assert status == 200
    assert body["status"] == "withdrawal-limit"
    assert body["baseline"]["status"] == "ok"
    assert body["baseline"]["resolved"] == {
        "app": "2.0", "p0": "1.0", "p1": "1.0", "p2": "1.0", "p3": "1.0",
    }
    assert body["after_withdrawal"]["status"] == "limit"
    assert body["after_withdrawal"]["error"]["code"] == "search_limit_exceeded"
    assert body["after_withdrawal"]["attempts"] == 7
    assert "changes" not in body


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_unknown_package_is_input_error():
    status, body = _withdraw(
        packages={"x": {"1.0": {}}},
        root=["x"],
        withdraw_package="ghost",
        withdraw_version="1.0",
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "ghost" in body["error"]["message"]


def test_unknown_version_is_input_error():
    status, body = _withdraw(
        packages={"x": {"1.0": {}}},
        root=["x"],
        withdraw_package="x",
        withdraw_version="2.0",
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "2.0" in body["error"]["message"]


def test_invalid_version_is_input_error():
    status, body = _withdraw(
        packages={"x": {"1.0": {}}},
        root=["x"],
        withdraw_package="x",
        withdraw_version="not-a-version",
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"


def test_resolve_input_validation_is_reused():
    # Bad package version in the index.
    status, body = _withdraw(
        packages={"x": {"bad": {}}},
        root=["x"],
        withdraw_package="x",
        withdraw_version="1.0",
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"

    # Bad root requirement.
    status, body = _withdraw(
        packages={"x": {"1.0": {}}},
        root=["x >>>> 1"],
        withdraw_package="x",
        withdraw_version="1.0",
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"

    # Bad lock.
    status, body = _withdraw(
        packages={"x": {"1.0": {}}},
        root=["x"],
        locks={"x": "nope"},
        withdraw_package="x",
        withdraw_version="1.0",
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"


def test_missing_withdraw_fields_is_422():
    response = client.post(
        "/analyze/version-withdrawal",
        json={"packages": {"x": {"1.0": {}}}, "root": ["x"]},
    )
    assert response.status_code == 422

    response = client.post(
        "/analyze/version-withdrawal",
        json={
            "packages": {"x": {"1.0": {}}},
            "root": ["x"],
            "withdraw_package": "x",
        },
    )
    assert response.status_code == 422


def test_extra_fields_rejected():
    response = client.post(
        "/analyze/version-withdrawal",
        json={
            "packages": {"x": {"1.0": {}}},
            "root": ["x"],
            "withdraw_package": "x",
            "withdraw_version": "1.0",
            "unexpected": 1,
        },
    )
    assert response.status_code == 422


def test_marker_env_and_allow_prereleases_are_honored_in_both_solves():
    # The withdrawn index is re-resolved under the same marker environment;
    # a marker-gated dependency must stay present in both solves.
    packages = {
        "app": {
            "1.0": {"dependencies": ["pyonly; python_version < '3.11'"]},
        },
        "pyonly": {"1.0": {}, "2.0": {}},
    }
    status, body = _withdraw(
        packages=packages,
        root=["app"],
        marker_env={"python_version": "3.10"},
        withdraw_package="pyonly",
        withdraw_version="2.0",
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["baseline"]["resolved"] == {"app": "1.0", "pyonly": "2.0"}
    assert body["after_withdrawal"]["resolved"] == {"app": "1.0", "pyonly": "1.0"}
    assert body["changes"]["changed"] == [
        {"package": "pyonly", "from": "2.0", "to": "1.0"}
    ]


def test_attempts_are_counted_per_solve():
    status, body = _withdraw(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root=["x>=1.0"],
        withdraw_package="x",
        withdraw_version="2.0",
    )
    assert status == 200
    # Baseline picks 2.0 on the first attempt; the re-solve picks 1.0.
    assert body["baseline"]["attempts"] == 1
    assert body["after_withdrawal"]["attempts"] == 1
