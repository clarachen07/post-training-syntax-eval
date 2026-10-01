import argparse
import csv
import json

from .common import ROOT, config, digest, read_jsonl, write_json

STAGES = ["base", "sft", "dpo", "final"]


def select_prompt():
    means = []
    for index, candidate in enumerate(["P0", "P1", "P2", "P3"]):
        metrics = [json.loads((ROOT / f"reports/dev128/{candidate}/{s}.json").read_text())["with_punct"]["metrics"] for s in STAGES]
        means.append({"candidate": candidate, "task1_upos": sum(x["task1_upos"] for x in metrics) / 4, "task3_las": sum(x["task3_las"] for x in metrics) / 4, "index": index})
    winner = max(means, key=lambda x: (x["task1_upos"], x["task3_las"], -x["index"]))
    spec = json.loads((ROOT / "manifests/prompts.json").read_text())[winner["candidate"]]
    result = {"selected": winner["candidate"], "criterion": "mean strict Task1 UPOS, then mean strict LAS, then P0/P1/P2/P3 order", "split": "dev128", "candidates": means, "prompt_spec_sha256": digest(spec)}
    write_json(ROOT / "manifests/selection.json", result)
    return result


def paired_ci(candidate):
    import numpy as np
    rows = {s: read_jsonl(ROOT / f"reports/test/{candidate}/{s}.sentences.jsonl") for s in STAGES}
    ids = [r["sent_id"] for r in rows["base"]]
    assert all([r["sent_id"] for r in rows[s]] == ids for s in STAGES)
    counts = {s: {k: np.array([r["counts"].get(k, 0) for r in rows[s]], dtype=np.float64) for k in ["gold", "task1_upos", "task3_uas", "task3_las"]} for s in STAGES}
    denom = counts["base"]["gold"]
    rng = np.random.default_rng(42)
    samples = {(a, b, k): [] for a, b in zip(STAGES[:-1], STAGES[1:]) for k in ["task1_upos", "task3_uas", "task3_las"]}
    # Chunking avoids allocating 10000 x 2077 all at once.
    for start in range(0, 10000, 100):
        chosen = rng.integers(0, len(ids), size=(100, len(ids)))
        d = denom[chosen].sum(axis=1)
        for (a, b, k), values in samples.items():
            values.extend(((counts[b][k] - counts[a][k])[chosen].sum(axis=1) / d * 100).tolist())
    result = []
    for (a, b, k), values in samples.items():
        result.append({"from": a, "to": b, "metric": k, "delta_percentage_points": (counts[b][k].sum() - counts[a][k].sum()) / denom.sum() * 100, "ci95": np.quantile(values, [0.025, 0.975]).tolist()})
    write_json(ROOT / f"reports/test/{candidate}/paired-bootstrap.json", {"seed": 42, "replicates": 10000, "comparisons": result})
    return result


def report(candidate, phase):
    base = ROOT / "reports/test" / candidate
    summaries = {s: json.loads((base / f"{s}.json").read_text()) for s in STAGES}
    fields = ["stage", "task1_UPOS_strict", "task1_UPOS_compat_extension", "task3_UPOS_strict", "UPOS_author_F1", "UAS_strict", "LAS_strict", "LAS_subtypes_strict", "UAS_author_F1", "LAS_author_F1", "task1_complete_rate", "task3_complete_rate", "truncated_rate", "valid_tree_rate"]
    table = []
    for s, summary in summaries.items():
        m = summary["with_punct"]["metrics"]
        a = summary["author"]["with_punct"]["metrics"]
        d = summary["with_punct"]["diagnostic_counts"]
        n = d["sentences"]
        table.append(dict(zip(fields, [s, m["task1_upos"], summary["with_punct"]["task1_author_extension"]["f1"], m["task3_upos"], a["UPOS"], m["task3_uas"], m["task3_las"], m["task3_las_subtypes"], a["UAS"], a["LAS"], d.get("task1_table_complete", 0) / n, d.get("task3_table_complete", 0) / n, d.get("truncated", 0) / n, d.get("valid_tree", 0) / n])))
    with (base / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    title = "原文主实验" if candidate == "P0" else f"开发集选优补充实验 {candidate}"
    text = [f"# EWT r2.15：{title} — {'UPOS' if phase == 'upos' else '依存分析'}", "", "每模型 2,077 句、25,094 词；冻结权重，BF16，单次生成三步表。", "", "数值以百分比表示；严格口径保留固定 gold 分母，作者口径包含其恢复规则。", ""]
    if phase == "upos":
        names = ["stage", "task1_UPOS_strict", "task1_UPOS_compat_extension", "task3_UPOS_strict", "UPOS_author_F1", "task1_complete_rate", "truncated_rate"]
    else:
        names = ["stage", "UAS_strict", "LAS_strict", "LAS_subtypes_strict", "UAS_author_F1", "LAS_author_F1", "task3_complete_rate", "valid_tree_rate"]
    text += ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for row in table:
        text.append("| " + " | ".join(row[k] if k == "stage" else f"{row[k] * 100:.2f}" for k in names) + " |")
    if phase == "dependency":
        ci = paired_ci(candidate)
        text += ["", "## 相邻阶段差值（百分点；按句配对 bootstrap 95% CI）", "", "| 比较 | 指标 | 差值 | 95% CI |", "| --- | --- | --- | --- |"]
        for x in ci:
            lo, hi = x["ci95"]
            text.append(f"| {x['from']} → {x['to']} | {x['metric']} | {x['delta_percentage_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    text += ["", "## 解释限制", "", "Base→SFT 同时改变权重与调用外壳，不能解释为纯 SFT 效应。三个 Tülu 阶段模板一致；行为成绩不直接证明内部句法表征变化。冻结推理不等于模型从未接触 EWT，本实验不作无污染结论。", "", "原文主实验与开发集优化补充实验分别报告；P3 含 train 示例，属于 few-shot。所有截断和格式失败均保留，未解决的基础设施失败不能标记为完成。", "", "每模型 JSON 包含不含标点的辅助成绩、作者恢复次数及适配异常；sentences.jsonl 包含逐句严格计数和错误诊断。"]
    (base / f"{phase}.md").write_text("\n".join(text) + "\n")
    print(f"Report ready: {base / (phase + '.md')}", flush=True)
    return table


def comparison(candidate):
    if candidate == "P0":
        write_json(ROOT / "reports/optimized-comparison.json", {"selected": "P0", "reused_original": True, "delta": 0})
        return
    rows = []
    for stage in STAGES:
        p0 = json.loads((ROOT / f"reports/test/P0/{stage}.json").read_text())["with_punct"]["metrics"]
        opt = json.loads((ROOT / f"reports/test/{candidate}/{stage}.json").read_text())["with_punct"]["metrics"]
        rows.append({"stage": stage, **{k + "_delta_pp": (opt[k] - p0[k]) * 100 for k in ["task1_upos", "task3_uas", "task3_las"]}})
    write_json(ROOT / "reports/optimized-comparison.json", {"selected": candidate, "comparisons": rows})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true")
    p.add_argument("--candidate", default="P0")
    p.add_argument("--phase", choices=["upos", "dependency"], default="upos")
    args = p.parse_args()
    if args.select:
        print(json.dumps(select_prompt(), indent=2))
    else:
        report(args.candidate, args.phase)


if __name__ == "__main__":
    main()
