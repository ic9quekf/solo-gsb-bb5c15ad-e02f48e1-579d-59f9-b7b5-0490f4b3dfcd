"""Version withdrawal analysis.

Given the same package index, root requirements, locks and resolution
options as ``/resolve`` plus a package name and one of its versions, this
module answers what happens when that version is withdrawn from the
request's own package list:

1. Solve the unmodified input to establish a *baseline*.
2. Remove only the named ``(package, version)`` entry from the in-memory
   index and solve **again from scratch** under the complete dependency
   constraints.  Backtracking triggered by the withdrawal is therefore a
   full re-resolution — never an inference along the old dependency graph.
3. When both solves succeed, diff them: packages added, packages that
   disappear and packages present in both whose version changed, and
   return the post-withdrawal dependency graph together with the lock
   disposition.

Statuses
--------
Every well-formed request returns HTTP 200; the top-level ``status``
discriminates the outcome and each solve is also reported in full:

* ``ok`` — both solves succeeded; ``changes`` holds the diff.
* ``baseline-conflict`` / ``withdrawal-conflict`` — the named solve has no
  solution and carries the structured ``/resolve``-style conflict body.
* ``baseline-limit`` / ``withdrawal-limit`` — the named solve exhausted
  ``max_attempts``.  Hitting the search limit is not treated as
  unsatisfiability, so a conflict is never reported in that case.

When the baseline does not succeed the withdrawal solve is not run.
All lists are sorted canonically; identical input always yields identical
output.
"""

from __future__ import annotations

from typing import Any, Optional

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .resolver import (
    PackageIndex,
    ResolutionInputError,
    RootEntry,
    resolve,
)
from .serialize import conflict_summary


def validate_withdrawal_target(
    index: PackageIndex,
    withdraw_package: str,
    withdraw_version: str,
) -> tuple[str, Version]:
    """Validate and normalize the withdrawal target against the index.

    The package must be present in the supplied index and the version must
    be a valid PEP 440 version offered by that package; otherwise the
    request is an input error (HTTP 400).
    """
    try:
        version = Version(withdraw_version)
    except InvalidVersion as exc:
        raise ResolutionInputError(
            f"withdraw_version {withdraw_version!r}: not a valid PEP 440 version: {exc}"
        )
    package = canonicalize_name(withdraw_package)
    records = index.versions.get(package)
    if records is None:
        raise ResolutionInputError(
            f"withdraw_package {withdraw_package!r}: package not present in the supplied index"
        )
    if not any(existing == version for existing, _record in records):
        raise ResolutionInputError(
            f"package {package!r}: version {str(version)!r} is not present "
            "in the supplied index"
        )
    return package, version


def index_without_version(
    index: PackageIndex, package: str, version: Version
) -> PackageIndex:
    """Return a copy of the index with one version entry removed.

    Only the named ``(package, version)`` record disappears; every other
    package and the package's other versions are kept byte-for-byte.  The
    original index is never mutated.
    """
    versions = {name: records for name, records in index.versions.items()}
    versions[package] = [
        (existing, record)
        for existing, record in index.versions[package]
        if existing != version
    ]
    return PackageIndex(versions=versions)


def _solve_view(result: Any, max_attempts: int) -> dict[str, Any]:
    """Render one solve outcome like the ``/resolve`` body."""
    view: dict[str, Any] = {
        "status": result.status,
        "attempts": result.attempts,
    }
    if result.status == "ok":
        view["resolved"] = result.chosen
        view["dependency_graph"] = result.edges
        view["locks"] = result.lock_notes
        return view
    view["locks"] = result.lock_notes
    view["conflict"] = conflict_summary(result)
    if result.status == "limit":
        view["error"] = {
            "code": "search_limit_exceeded",
            "message": f"resolution exceeded {max_attempts} candidate evaluations",
        }
    return view


def _diff_chosen(
    baseline: dict[str, str], after: dict[str, str]
) -> dict[str, Any]:
    """Compare two successful solutions, sorted by canonical package name."""
    baseline_names = set(baseline)
    after_names = set(after)

    added = [
        {"package": name, "version": after[name]}
        for name in sorted(after_names - baseline_names)
    ]
    removed = [
        {"package": name, "version": baseline[name]}
        for name in sorted(baseline_names - after_names)
    ]
    changed = [
        {"package": name, "from": baseline[name], "to": after[name]}
        for name in sorted(baseline_names & after_names)
        if baseline[name] != after[name]
    ]
    return {
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
        },
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def run_version_withdrawal(
    index: PackageIndex,
    entries: list[RootEntry],
    locks: dict[str, Version],
    *,
    allow_prereleases: bool,
    marker_env: Optional[dict[str, str]],
    max_attempts: int,
    withdraw_package: str,
    withdraw_version: str,
) -> dict[str, Any]:
    """Run the baseline + withdrawal solves and render the public body."""
    package, version = validate_withdrawal_target(
        index, withdraw_package, withdraw_version
    )
    roots = [entry.dep for entry in entries]

    common = dict(
        allow_prereleases=allow_prereleases,
        marker_env=marker_env,
        max_attempts=max_attempts,
    )

    baseline_result = resolve(index, roots, locks, **common)
    baseline_view = _solve_view(baseline_result, max_attempts)

    body: dict[str, Any] = {
        "allow_prereleases": allow_prereleases,
        "withdrawn": {"package": package, "version": str(version)},
        "baseline": baseline_view,
        "after_withdrawal": None,
    }

    if baseline_result.status == "conflict":
        body["status"] = "baseline-conflict"
        return body
    if baseline_result.status == "limit":
        body["status"] = "baseline-limit"
        return body

    # The baseline succeeded: re-resolve from the modified index under the
    # complete constraints, so withdrawal-induced backtracking is a genuine
    # fresh search rather than a rewrite of the baseline graph.
    withdrawn_index = index_without_version(index, package, version)
    after_result = resolve(withdrawn_index, roots, locks, **common)
    body["after_withdrawal"] = _solve_view(after_result, max_attempts)

    if after_result.status == "conflict":
        body["status"] = "withdrawal-conflict"
        return body
    if after_result.status == "limit":
        body["status"] = "withdrawal-limit"
        return body

    body["status"] = "ok"
    body["changes"] = _diff_chosen(baseline_result.chosen, after_result.chosen)
    # The post-withdrawal dependency graph and lock disposition are the
    # primary artifacts; surface them at top level as well as inside
    # ``after_withdrawal`` for direct consumption.
    body["dependency_graph"] = after_result.edges
    body["locks"] = after_result.lock_notes
    return body
