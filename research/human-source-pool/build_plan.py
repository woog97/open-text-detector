"""Validate the researched source registry and apportion planning quotas.

This does not download training text or certify any document as human.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def apportion(total, weights):
    denominator = sum(weights.values())
    ideals = {key: Fraction(total * value, denominator) for key, value in weights.items()}
    counts = {key: int(value) for key, value in ideals.items()}
    order = sorted(ideals, key=lambda key: (-(ideals[key] - counts[key]), key))
    for key in order[:total - sum(counts.values())]:
        counts[key] += 1
    assert sum(counts.values()) == total
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=100_000)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    registry = json.loads((ROOT / "source-registry.json").read_text())
    categories, sources = registry["categories"], registry["sources"]
    assert len({x["id"] for x in sources}) == len(sources)
    assert sum(x["weight_units"] for x in categories) == registry["category_weight_denominator"] == 999
    cat_ids = {x["id"] for x in categories}
    for source in sources:
        assert source["category"] in cat_ids
        assert 0 <= source["within_category_percent"] <= 100
        assert source["admitted_records"] == 0
        assert source["eligible_records_measured"] is None
        assert source["rights_evidence_urls"] and source["group_keys"] and source["remaining_checks"]
    category_counts = apportion(args.size, {x["id"]: x["weight_units"] for x in categories})
    plan_rows, status_counts = [], Counter()
    for category in categories:
        members = [x for x in sources if x["category"] == category["id"] and x["within_category_percent"]]
        assert sum(x["within_category_percent"] for x in members) == 100
        quotas = apportion(category_counts[category["id"]], {x["id"]: x["within_category_percent"] for x in members})
        for source in members:
            row = dict(source_id=source["id"], category=category["id"], planned_passages=quotas[source["id"]],
                       rights_evidence_status=source["rights_evidence_status"], actual_eligible_passages=None)
            plan_rows.append(row)
            status_counts[source["rights_evidence_status"]] += row["planned_passages"]
    assert sum(x["planned_passages"] for x in plan_rows) == args.size
    evidence = json.loads((ROOT / "evidence-manifest.json").read_text())
    for record in evidence:
        if record["status"] == "retrieved":
            body = (ROOT / record["path"]).read_bytes()
            assert len(body) == record["bytes"]
            assert hashlib.sha256(body).hexdigest() == record["sha256"]
    plan = dict(status="planning_quotas_only",planned_total=args.size,actual_admitted_total=0,
                rounding="Largest remainder, category then source; ties ordered by identifier",
                source_shortfall_policy="Keep quota unfilled; never silently renormalize or repeat records",
                category_counts=category_counts,source_quotas=plan_rows,planned_counts_by_rights_evidence_status=dict(status_counts))
    (ROOT / "sampling-plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    lines = ["# Human source registry", "", "Researched October 1, 2026. All allocations below are proposals. No training passages have been admitted.", "",
             "Read README.md for the category design and SAMPLING.md for admission and splitting rules. An open-subset designation is source-level evidence, not a completed document audit.", ""]
    quota_index = {x["source_id"]: x["planned_passages"] for x in plan_rows}
    for category in categories:
        lines += ["## " + category["name"], ""]
        for source in [x for x in sources if x["category"] == category["id"]]:
            lines += ["### " + source["name"], "",
                f"**Allocation:** {source['within_category_percent']}% within category; {quota_index.get(source['id'],0):,} passages in the {args.size:,}-passage planning example. **Rights evidence:** {source['rights_evidence_status']}. **ID:** `{source['id']}`.", "",
                source["source_findings"], "",
                "**Selection:** " + source["proposed_selection"], "",
                "**Access:** [source]("+source["origin_url"]+"); [data or retrieval route]("+source["access_url"]+").", "",
                "**Evidence:** " + "; ".join(f"[source {i+1}]({url})" for i,url in enumerate(source["rights_evidence_urls"])) + ".", "",
                "**Group before splitting:** " + ", ".join(source["group_keys"]) + ".", "",
                "**Remaining checks:** " + " ".join(source["remaining_checks"]), ""]
            if source["alternatives"]:
                lines += ["**Related route:** " + "; ".join(f"[alternative {i+1}]({url})" for i,url in enumerate(source["alternatives"])) + ".", ""]
    (ROOT / "SOURCE_REGISTRY.md").write_text("\n".join(lines))
    validation = dict(registry_entries=len(sources),weighted_sources=len(plan_rows),category_count=len(categories),
                      paper_weight_sum=99.9,within_category_weight_sums_valid=True,planned_total=args.size,
                      actual_admitted_total=0,evidence_files_verified=sum(x["status"]=="retrieved" for x in evidence),
                      evidence_failed=sum(x["status"]!="retrieved" for x in evidence),
                      registry_sha256=hashlib.sha256((ROOT/"source-registry.json").read_bytes()).hexdigest(),
                      plan_sha256=hashlib.sha256((ROOT/"sampling-plan.json").read_bytes()).hexdigest())
    (ROOT / "validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print(json.dumps({"validation":validation,"category_counts":category_counts,"rights_status_counts":status_counts},indent=2))


if __name__ == "__main__":
    main()
