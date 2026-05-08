import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data/intent")
REPORTS_DIR = os.path.join(BASE_DIR, "reports/dataset_analysis")
os.makedirs(REPORTS_DIR, exist_ok=True)

OUR_INTENTS = [
    "inform", "question", "directive", "commissive",
    "greeting", "farewell", "gratitude", "complaint", "confirmation"
]

COLORS = [
    "#7F77DD", "#1D9E75", "#D85A30", "#BA7517",
    "#378ADD", "#D4537E", "#639922", "#E24B4A", "#888780"
]

def compute_stats(name, df):
    texts   = df["text"].tolist()
    intents = df["intent"].tolist()
    dist    = Counter(intents)
    lengths = [len(str(t).split()) for t in texts]

    stats = {
        "dataset":             name,
        "total_examples":      len(df),
        "unique_texts":        int(df["text"].nunique()),
        "intent_distribution": {
            k: {
                "count":      dist.get(k, 0),
                "percentage": round(dist.get(k, 0) / len(df) * 100, 2)
            }
            for k in OUR_INTENTS
        },
        "text_length": {
            "min_words":    int(min(lengths)),
            "max_words":    int(max(lengths)),
            "avg_words":    round(sum(lengths) / len(lengths), 2),
            "median_words": int(sorted(lengths)[len(lengths) // 2])
        },
        "class_balance": {
            "most_common":     dist.most_common(1)[0][0],
            "least_common":    dist.most_common()[-1][0],
            "imbalance_ratio": round(
                dist.most_common(1)[0][1] / max(dist.most_common()[-1][1], 1), 2
            )
        },
        "sample_examples": {
            intent: df[df["intent"] == intent]["text"].iloc[0]
            if intent in df["intent"].values else "N/A"
            for intent in OUR_INTENTS
        }
    }

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total examples  : {stats['total_examples']:,}")
    print(f"  Unique texts    : {stats['unique_texts']:,}")
    print(f"  Avg word length : {stats['text_length']['avg_words']}")
    print(f"  Imbalance ratio : {stats['class_balance']['imbalance_ratio']}x  "
          f"(most: {stats['class_balance']['most_common']} / "
          f"least: {stats['class_balance']['least_common']})")
    print(f"\n  Intent distribution:")
    for intent, color in zip(OUR_INTENTS, COLORS):
        count = dist.get(intent, 0)
        pct   = count / len(df) * 100
        bar   = "█" * int(pct / 2)
        print(f"    {intent:<14}: {count:>6,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Sample examples:")
    for intent in OUR_INTENTS:
        sample = stats["sample_examples"][intent]
        print(f"    {intent:<14}: {str(sample)[:55]}")

    return stats

def plot_distributions(stats_list, filename, title):
    n    = len(stats_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, stats in zip(axes, stats_list):
        counts = [stats["intent_distribution"][i]["count"] for i in OUR_INTENTS]
        bars   = ax.bar(OUR_INTENTS, counts, color=COLORS, edgecolor="white")
        ax.set_title(stats["dataset"], fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_xlabel("Intent")
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(counts) * 0.01,
                        f"{count:,}", ha="center", va="bottom", fontsize=7)
        ax.tick_params(axis="x", rotation=35)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")

def plot_comparison_table(stats_list, filename):
    fig, ax = plt.subplots(figsize=(16, len(stats_list) * 1.2 + 2))
    ax.axis("off")

    rows = []
    for s in stats_list:
        row = [s["dataset"], f"{s['total_examples']:,}",
               f"{s['class_balance']['imbalance_ratio']}x"]
        for intent in OUR_INTENTS:
            pct = s["intent_distribution"][intent]["percentage"]
            cnt = s["intent_distribution"][intent]["count"]
            row.append(f"{cnt:,} ({pct}%)")
        rows.append(row)

    cols = ["Dataset", "Total", "Imbalance"] + [i.capitalize() for i in OUR_INTENTS]
    table = ax.table(cellText=rows, colLabels=cols,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 2.2)

    for j in range(len(cols)):
        table[0, j].set_facecolor("#3C3489")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            table[i, j].set_facecolor("#F1EFE8" if i % 2 == 0 else "white")

    plt.title("Dataset Comparison — 9 Intent Classes",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")

def plot_text_length_distribution(datasets_dict, filename):
    n    = len(datasets_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, datasets_dict.items()):
        lengths = [len(str(t).split()) for t in df["text"]]
        mean    = sum(lengths) / len(lengths)
        ax.hist(lengths, bins=30, color="#7F77DD", edgecolor="white", alpha=0.85)
        ax.set_title(f"{name}\nText Length Distribution", fontsize=10)
        ax.set_xlabel("Word count")
        ax.set_ylabel("Frequency")
        ax.axvline(mean, color="#D85A30", linestyle="--",
                   label=f"Mean: {mean:.1f} words")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")

def plot_intent_heatmap(datasets_dict, filename):

    import numpy as np

    names = list(datasets_dict.keys())
    data  = []
    for df in datasets_dict.values():
        dist  = Counter(df["intent"].tolist())
        total = len(df)
        data.append([round(dist.get(i, 0) / total * 100, 1) for i in OUR_INTENTS])

    matrix = pd.DataFrame(data, index=names, columns=OUR_INTENTS)

    fig, ax = plt.subplots(figsize=(12, len(names) * 1.2 + 2))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Percentage (%)")

    ax.set_xticks(range(len(OUR_INTENTS)))
    ax.set_xticklabels([i.capitalize() for i in OUR_INTENTS], rotation=35, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(OUR_INTENTS)):
            ax.text(j, i, f"{matrix.values[i, j]:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="black" if matrix.values[i, j] < 50 else "white")

    ax.set_title("Intent Distribution Heatmap (% per dataset)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")

if __name__ == "__main__":
    print("=" * 60)
    print("  DATASET ANALYSIS  (9 intents)")
    print("=" * 60)

    dataset_files = {
        "DailyDialog": "dailydialog_intent.csv",
        "CLINC150":    "clinc150_intent.csv",
        #"Banking77":   "banking77_intent.csv",
    }

    datasets  = {}
    all_stats = []

    for name, filename in dataset_files.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df             = pd.read_csv(path).dropna()
            datasets[name] = df
            stats          = compute_stats(name, df)
            all_stats.append(stats)
        else:
            print(f"\n  Skipping {name} — {filename} not found")

    if not datasets:
        print("No CSVs found. Run load_datasets.py first.")
        exit()

    parts = []
    for split in ["train", "val", "test"]:
        p = os.path.join(DATA_DIR, f"combined_{split}.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p))

    if parts:
        combined_df    = pd.concat(parts).dropna()
        combined_stats = compute_stats("Combined", combined_df)
        all_stats.append(combined_stats)
        datasets["Combined"] = combined_df

    print("\nGenerating plots...")

    plot_distributions(
        [s for s in all_stats if s["dataset"] != "Combined"],
        os.path.join(REPORTS_DIR, "individual_distributions.png"),
        "Intent Distribution — Individual Datasets"
    )

    if "Combined" in datasets:
        plot_distributions(
            [combined_stats],
            os.path.join(REPORTS_DIR, "combined_distribution.png"),
            "Intent Distribution — Combined Dataset"
        )

    plot_distributions(
        all_stats,
        os.path.join(REPORTS_DIR, "all_distributions_comparison.png"),
        "Intent Distribution — All Datasets"
    )

    plot_comparison_table(
        all_stats,
        os.path.join(REPORTS_DIR, "comparison_table.png")
    )

    plot_text_length_distribution(
        {k: v for k, v in datasets.items() if k != "Combined"},
        os.path.join(REPORTS_DIR, "text_length_distributions.png")
    )

    plot_intent_heatmap(
        datasets,
        os.path.join(REPORTS_DIR, "intent_heatmap.png")
    )

    report = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_intents":       9,
        "intent_classes":    OUR_INTENTS,
        "datasets_analyzed": list(datasets.keys()),
        "individual_stats":  {s["dataset"]: s for s in all_stats
                              if s["dataset"] != "Combined"},
        "combined_stats":    combined_stats if "Combined" in datasets else {},
        "splits": {
            split: len(pd.read_csv(os.path.join(DATA_DIR, f"combined_{split}.csv")))
            for split in ["train", "val", "test"]
            if os.path.exists(os.path.join(DATA_DIR, f"combined_{split}.csv"))
        }
    }

    report_path = os.path.join(REPORTS_DIR, "dataset_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ JSON report saved")

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for s in all_stats:
        print(f"\n  {s['dataset']:<15} | "
              f"Total: {s['total_examples']:>8,} | "
              f"Imbalance: {s['class_balance']['imbalance_ratio']:>6}x")

    print(f"\n  Saved to reports/dataset_analysis/:")
    for f in ["individual_distributions.png", "combined_distribution.png",
              "all_distributions_comparison.png", "comparison_table.png",
              "text_length_distributions.png", "intent_heatmap.png",
              "dataset_analysis_report.json"]:
        print(f"    {f}")