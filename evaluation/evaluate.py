import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.core import SupportAgent


def evaluate_case(agent: SupportAgent, case: Dict[str, Any]) -> Dict[str, Any]:
    case_id = case["id"]
    category = case.get("category", "general")
    messages = case.get("messages", [])
    expect = case.get("expect", {})

    history: List[Dict[str, str]] = []
    final_res = None

    for msg in messages:
        user_text = msg["content"]
        final_res = agent.chat(user_text, history=history)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": final_res.answer})

    answer_lower = final_res.answer.lower() if final_res else ""
    failures = []

    # 1. Check must_include
    for item in expect.get("must_include", []):
        if item.lower() not in answer_lower:
            failures.append(f"Missing required phrase: '{item}'")

    # 2. Check must_not_include
    for item in expect.get("must_not_include", []):
        if item.lower() in answer_lower:
            failures.append(f"Contains forbidden phrase: '{item}'")

    # 3. Check must_include_concepts (substring or conceptual check)
    for concept in expect.get("must_include_concepts", []):
        tokens = [t.lower() for t in concept.split() if len(t) > 3]
        if not any(t in answer_lower for t in tokens):
            failures.append(f"Missing concept: '{concept}'")

    # 4. Check forbidden sources as authority
    for forbidden in expect.get("forbidden_sources_as_authority", []):
        if forbidden in final_res.sources or forbidden in final_res.answer:
            failures.append(f"Cited forbidden source: '{forbidden}'")

    # 5. Check handoff expectations
    if "handoff" in expect:
        expected_handoff = expect["handoff"]
        if final_res.handoff_recommended != expected_handoff:
            failures.append(
                f"Handoff mismatch: expected {expected_handoff}, got {final_res.handoff_recommended}"
            )

    passed = len(failures) == 0
    return {
        "id": case_id,
        "category": category,
        "passed": passed,
        "failures": failures,
        "sources": final_res.sources,
        "handoff": final_res.handoff_recommended,
    }


def run_all():
    agent = SupportAgent()
    root = Path(__file__).parent

    files_to_run = [root / "visible-cases.json"]
    custom_path = root / "custom-cases.json"
    if custom_path.exists():
        files_to_run.append(custom_path)

    all_cases = []
    for f in files_to_run:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            all_cases.extend(data.get("cases", []))

    print(f"\nRunning evaluation on {len(all_cases)} test cases...\n" + "=" * 65)

    results_by_cat: Dict[str, List[bool]] = {}
    case_results = []

    for c in all_cases:
        res = evaluate_case(agent, c)
        case_results.append(res)
        cat = res["category"]
        results_by_cat.setdefault(cat, []).append(res["passed"])

        status = "✅ PASS" if res["passed"] else "❌ FAIL"
        print(f"[{status}] {res['id']} ({cat})")
        if not res["passed"]:
            for f in res["failures"]:
                print(f"       -> {f}")

    print("\n" + "=" * 65)
    print("CATEGORY BREAKDOWN:")
    print("-" * 65)
    total_passed = 0
    total_count = 0
    for cat, scores in results_by_cat.items():
        passed = sum(scores)
        count = len(scores)
        total_passed += passed
        total_count += count
        pct = (passed / count) * 100
        print(f"• {cat:<25}: {passed}/{count} ({pct:.1f}%)")

    overall_pct = (total_passed / total_count) * 100
    print("-" * 65)
    print(f"TOTAL SCORE: {total_passed}/{total_count} ({overall_pct:.1f}%)\n")


if __name__ == "__main__":
    run_all()