#!/usr/bin/env python3
"""
Generate final experiment report
=================================
汇总:
  - experiment_config.json
  - comparison_results.json
  - 生成图像统计
  - Missing Boundary 分布
"""
import argparse, os, json, csv
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mb_dir', type=str, required=True,
                        help='MissingBoundary workspace root')
    parser.add_argument('--output', type=str, default='report.md')
    args = parser.parse_args()

    report = []
    report.append("# Missing Boundary Guided Generation — 实验报告\n")

    # 1. Config
    config_path = os.path.join(args.mb_dir, 'experiment_config.json')
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        report.append("## 实验配置\n")
        report.append(f"- **类别**: {config['category']}")
        report.append(f"- **检测器**: {config['detector']}")
        report.append(f"- **生成器**: {config['generator']}")
        report.append(f"- **特征层**: {config['feature_layers']}")
        report.append(f"- **M(x)**: α={config['alpha']}*Gap + β={config['beta']}*PB")
        report.append(f"- **Gap k-neighbors**: {config['gap_k_neighbors']}")
        report.append(f"- **高缺失阈值**: P{config['percentile_threshold']}")
        report.append(f"- **BN prompt**: `{config['boundary_normal_prompt']}`")
        report.append(f"- **BD prompt**: `{config['blind_defect_prompt']}`")
        report.append(f"- **Random 种子**: {config['random_selection_seed']}\n")

        report.append("### 数据集\n")
        report.append("| 数据集 | 组成 |")
        report.append("|--------|------|")
        for name, ds in config['datasets'].items():
            report.append(f"| {name} | {ds['description']} |")
        report.append("")

    # 2. Comparison results
    eval_dir = os.path.join(args.mb_dir, 'outputs', 'evaluation')
    result_path = os.path.join(eval_dir, 'comparison_results.json')
    if os.path.isfile(result_path):
        with open(result_path) as f:
            results = json.load(f)
        report.append("## 评估结果\n")
        report.append("| Method | Image AUROC | Pixel AUROC | Image AP | Pixel AP |")
        report.append("|--------|------------|------------|----------|----------|")
        # Match by keyword to handle suffixed names like 'Baseline (700ep)'
        def find(name_keyword):
            for k, v in results.items():
                if name_keyword in k:
                    return v
            return None
        for keyword in ['Baseline (700ep)', 'Baseline (200ep)', 'Random Aug', 'Missing Boundary']:
            m = find(keyword)
            if m:
                report.append(f"| {keyword} | {m.get('Image AUROC', 0):.4f} | "
                              f"{m.get('Pixel AUROC', 0):.4f} | "
                              f"{m.get('Image AP', 0):.4f} | "
                              f"{m.get('Pixel AP', 0):.4f} |")
        report.append("")

    # 3. Generated samples stats
    gen_dir = os.path.join(args.mb_dir, 'outputs', 'generated')
    report.append("## 生成样本统计\n")
    for sub in ['boundary_normal', 'blind_defect', 'random_anomalies']:
        img_dir = os.path.join(gen_dir, 'bottle', sub, 'image')
        n = len(glob(os.path.join(img_dir, '*.png'))) if os.path.isdir(img_dir) else 0
        report.append(f"- **{sub}**: {n} 张")
    report.append("")

    # 4. Missing Boundary distribution
    mb_csv = os.path.join(args.mb_dir, 'outputs', 'features', 'bottle', 'missing_boundary',
                          'missing_boundary.csv')
    if os.path.isfile(mb_csv):
        report.append("## Missing Boundary Top-10 (测试集)\n")
        report.append("| 图像 | Score | PB | Gap | M |")
        report.append("|------|-------|----|----|----|")
        rows = []
        with open(mb_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == 'test':
                    rows.append(row)
        rows.sort(key=lambda r: float(r['M']), reverse=True)
        for r in rows[:10]:
            report.append(f"| {r['img_name']} | {float(r['anomaly_score']):.4f} | "
                          f"{float(r['pb']):.4f} | {float(r['gap_mean']):.4f} | {float(r['M']):.4f} |")
        report.append("")

    # 5. Findings
    findings_path = os.path.join(args.mb_dir, 'analysis_findings.md')
    if os.path.isfile(findings_path):
        report.append("\n## 中间发现\n")
        report.append("详见 `analysis_findings.md`\n")

    report_text = '\n'.join(report)
    with open(args.output, 'w') as f:
        f.write(report_text)
    print(f"Report saved to {args.output}")
    print(report_text[:3000])


if __name__ == '__main__':
    main()
