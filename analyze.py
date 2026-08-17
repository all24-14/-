# -*- coding: utf-8 -*-
"""
客服工单趋势分析工具
====================
对附件中的 50 条客服工单数据进行分析，输出：
  1) 各分析维度的结构化指标（打印到控制台 + 写入 output/report.md）
  2) 可视化图表（写入 output/*.png）

分析维度：
  1. 时间趋势 —— 每日工单量、各类别随时间变化
  2. 类型分布 —— 问题分类占比
  3. 严重程度 —— 优先级分布、满意度分布
  4. 处理效率 —— 各分类平均/最长处理时长
  5. 渠道分布 —— 来源渠道占比
  6. 解决状态 —— 未解决工单分布及风险
  7. 关联分析 —— 分类 × 满意度 × 处理时长的交叉关系

用法：
    python analyze.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # 无界面后端，便于命令行/CI 运行
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "task5_tickets.json")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# 中文字体（Windows 微软雅黑 / 黑体），保证图表中文不乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 统一配色
COLORS = {
    "支付问题": "#E15759",
    "退款退货": "#F28E2B",
    "物流查询": "#4E79A7",
    "投诉": "#B07AA1",
    "账号问题": "#76B7B2",
    "商品咨询": "#59A14F",
}
PRIORITY_COLOR = {"高": "#E15759", "中": "#F28E2B", "低": "#59A14F"}


def load_data() -> pd.DataFrame:
    """加载并清洗工单数据。"""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M")
    df["date"] = df["created_at"].dt.date
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.day_name()
    return df


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_category_distribution(df: pd.DataFrame) -> None:
    """图1：问题分类分布。"""
    counts = df["category"].value_counts()
    colors = [COLORS.get(c, "#888888") for c in counts.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_title("工单问题分类分布")
    ax.set_xlabel("问题分类")
    ax.set_ylabel("工单数量")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_category_distribution.png"), dpi=150)
    plt.close(fig)


def plot_daily_trend(df: pd.DataFrame) -> None:
    """图2：每日工单量趋势（含类别堆叠）。"""
    daily = df.groupby(["date", "category"]).size().unstack(fill_value=0)
    daily.index = pd.to_datetime(daily.index)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    total = daily.sum(axis=1)
    axes[0].plot(total.index, total.values, marker="o", color="#4E79A7", linewidth=2)
    axes[0].fill_between(total.index, total.values, alpha=0.15, color="#4E79A7")
    axes[0].set_title("每日工单总量趋势")
    axes[0].set_ylabel("工单数量")

    order = df["category"].value_counts().index.tolist()
    daily = daily[order]
    daily.plot.area(ax=axes[1], color=[COLORS[c] for c in order], alpha=0.8)
    axes[1].set_title("每日工单类别构成（堆叠）")
    axes[1].set_ylabel("工单数量")
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)
    axes[1].set_xlabel("日期")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_daily_trend.png"), dpi=150)
    plt.close(fig)


def plot_priority_satisfaction(df: pd.DataFrame) -> None:
    """图3：优先级分布 + 各优先级满意度。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    pr = df["priority"].value_counts().reindex(["高", "中", "低"])
    axes[0].bar(pr.index, pr.values, color=[PRIORITY_COLOR[p] for p in pr.index])
    axes[0].set_title("优先级分布")
    axes[0].set_ylabel("工单数量")

    sat = df.groupby("priority")["satisfaction"].mean().reindex(["高", "中", "低"])
    axes[1].bar(sat.index, sat.values, color=[PRIORITY_COLOR[p] for p in sat.index])
    axes[1].axhline(df["satisfaction"].mean(), color="gray", linestyle="--", linewidth=1)
    axes[1].text(0.02, df["satisfaction"].mean() + 0.05, f"总体均值 {df['satisfaction'].mean():.2f}",
                 color="gray", fontsize=9)
    axes[1].set_title("各优先级平均满意度")
    axes[1].set_ylabel("平均满意度")
    axes[1].set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_priority_satisfaction.png"), dpi=150)
    plt.close(fig)


def plot_resolution_time(df: pd.DataFrame) -> None:
    """图4：各分类平均处理时长（突出异常长的退款退货）。"""
    res = df.groupby("category")["resolution_time_hours"].mean().sort_values()
    colors = [COLORS.get(c, "#888888") for c in res.index]
    # 高亮处理时长最长的分类
    colors[-1] = "#D62728"
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(res.index, res.values, color=colors)
    ax.bar_label(bars, fmt="%.1fh", padding=3)
    ax.set_title("各分类平均处理时长（小时）")
    ax.set_xlabel("平均处理时长（小时）")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_resolution_time.png"), dpi=150)
    plt.close(fig)


def plot_satisfaction_by_category(df: pd.DataFrame) -> None:
    """图5：各分类平均满意度。"""
    sat = df.groupby("category")["satisfaction"].mean().sort_values()
    colors = [COLORS.get(c, "#888888") for c in sat.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sat.index, sat.values, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.axhline(df["satisfaction"].mean(), color="gray", linestyle="--", linewidth=1,
               label=f"总体均值 {df['satisfaction'].mean():.2f}")
    ax.legend()
    ax.set_title("各分类平均满意度（1-5 分）")
    ax.set_ylabel("平均满意度")
    ax.set_ylim(0, 5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_satisfaction_by_category.png"), dpi=150)
    plt.close(fig)


def plot_channel_distribution(df: pd.DataFrame) -> None:
    """图6：来源渠道分布。"""
    counts = df["channel"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].pie(counts.values, labels=counts.index, autopct="%.0f%%",
                colors=["#4E79A7", "#F28E2B", "#59A14F"],
                startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("来源渠道占比")

    cross = pd.crosstab(df["channel"], df["priority"]).reindex(columns=["高", "中", "低"])
    cross.plot.bar(ax=axes[1], color=[PRIORITY_COLOR[p] for p in ["高", "中", "低"]], stacked=True)
    axes[1].set_title("各渠道 × 优先级")
    axes[1].set_ylabel("工单数量")
    axes[1].set_xlabel("渠道")
    axes[1].legend(title="优先级")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_channel_distribution.png"), dpi=150)
    plt.close(fig)


def plot_unresolved(df: pd.DataFrame) -> None:
    """图7：未解决工单分布及风险。"""
    unresolved = df[~df["is_resolved"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    labels = ["已解决", "未解决"]
    vals = [df["is_resolved"].sum(), (~df["is_resolved"]).sum()]
    axes[0].pie(vals, labels=labels, autopct="%.0f%%", colors=["#59A14F", "#E15759"],
                startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("工单解决状态")

    if len(unresolved):
        uc = unresolved["category"].value_counts()
        axes[1].barh(uc.index[::-1], uc.values[::-1],
                     color=[COLORS.get(c, "#E15759") for c in uc.index[::-1]])
        axes[1].set_title(f"未解决工单分类（共 {len(unresolved)} 条）")
        axes[1].set_xlabel("未解决工单数量")
        axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_unresolved.png"), dpi=150)
    plt.close(fig)


def plot_satisfaction_distribution(df: pd.DataFrame) -> None:
    """图8：满意度评分分布。"""
    dist = df["satisfaction"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E15759" if s <= 2 else "#F28E2B" if s == 3 else "#59A14F" for s in dist.index]
    bars = ax.bar([str(int(s)) + "分" for s in dist.index], dist.values, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_title("满意度评分分布")
    ax.set_xlabel("满意度")
    ax.set_ylabel("工单数量")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "08_satisfaction_distribution.png"), dpi=150)
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame) -> None:
    """图9：汇总仪表盘（2×2 一屏总览，便于作为运行结果截图）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("客服工单趋势分析 —— 汇总仪表盘", fontsize=16, fontweight="bold")

    # 左上：分类分布
    counts = df["category"].value_counts()
    axes[0, 0].bar(counts.index, counts.values,
                   color=[COLORS.get(c, "#888888") for c in counts.index])
    axes[0, 0].set_title("问题分类分布")
    axes[0, 0].set_ylabel("工单数量")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # 右上：每日趋势
    daily = df.groupby("date").size()
    daily.index = pd.to_datetime(daily.index)
    axes[0, 1].plot(daily.index, daily.values, marker="o", color="#4E79A7", linewidth=2)
    axes[0, 1].fill_between(daily.index, daily.values, alpha=0.15, color="#4E79A7")
    axes[0, 1].set_title("每日工单量趋势")
    axes[0, 1].set_ylabel("工单数量")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # 左下：处理时长
    res = df.groupby("category")["resolution_time_hours"].mean().sort_values()
    colors = [COLORS.get(c, "#888888") for c in res.index]
    colors[-1] = "#D62728"
    axes[1, 0].barh(res.index, res.values, color=colors)
    axes[1, 0].set_title("各分类平均处理时长（小时）")
    axes[1, 0].set_xlabel("小时")

    # 右下：满意度
    sat = df.groupby("category")["satisfaction"].mean().sort_values()
    axes[1, 1].bar(sat.index, sat.values,
                   color=[COLORS.get(c, "#888888") for c in sat.index])
    axes[1, 1].axhline(df["satisfaction"].mean(), color="gray", linestyle="--",
                       label=f"总体 {df['satisfaction'].mean():.2f}")
    axes[1, 1].set_title("各分类平均满意度")
    axes[1, 1].set_ylabel("满意度")
    axes[1, 1].set_ylim(0, 5)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "09_dashboard.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 异常检测
# ---------------------------------------------------------------------------
def detect_anomalies(df: pd.DataFrame) -> list:
    """识别值得主管关注的异常信号，返回带判断依据的列表。"""
    anomalies = []

    # 1. 退款退货处理时长异常 + 退款时效被反复投诉
    refund = df[df["category"] == "退款退货"]
    refund_mean = refund["resolution_time_hours"].mean()
    others_mean = df[df["category"] != "退款退货"]["resolution_time_hours"].mean()
    refund_slow = df[df["description"].str.contains(
        "审核中|钱还没退|没退|什么时候给|还在处理中", na=False)]
    if refund_mean > others_mean * 3:
        ids = "、".join(refund_slow["ticket_id"].tolist())
        anomalies.append({
            "title": "退款退货处理时长显著异常、时效被反复投诉",
            "detail": f"退款退货平均处理 {refund_mean:.1f} 小时，是其他分类均值({others_mean:.1f}h)的 "
                      f"{refund_mean / others_mean:.1f} 倍，最长单笔达 {refund['resolution_time_hours'].max():.0f} 小时；"
                      f"且描述中反复出现「审核一周」「钱还没退」「还在处理中」等时效抱怨（{ids}）。"
                      "退款流程存在明显瓶颈，客户资金被长时间占用。",
        })

    # 2. 支付类工单占比最高，且「重复扣款/多扣款」反复出现
    payment = df[df["category"] == "支付问题"]
    dup = df[df["description"].str.contains(
        "重复扣款|扣了两次|两个都扣|多扣|金额不对|又出现", na=False)]
    if len(dup) >= 3:
        ids = "、".join(dup["ticket_id"].tolist())
        anomalies.append({
            "title": "支付类工单占比最高，「重复扣款/多扣款」反复出现",
            "detail": f"支付问题共 {len(payment)} 条（占 {len(payment) / len(df) * 100:.0f}%），为第一大分类；"
                      f"其中 {len(dup)} 条涉及重复扣款/多扣款（{ids}），且 T046 明确提到「上个月也有过」，"
                      "说明这是反复发生的系统性支付缺陷（疑似回调幂等性问题），涉及资金安全，风险最高。",
        })

    # 3. 投诉类满意度极低
    complaint = df[df["category"] == "投诉"]
    if len(complaint) and complaint["satisfaction"].mean() <= 1.5:
        anomalies.append({
            "title": "投诉类工单满意度垫底（全部 1 分）",
            "detail": f"{len(complaint)} 条投诉平均满意度 {complaint['satisfaction'].mean():.2f}，"
                      "内容集中在客服态度差、响应慢（等待 40 分钟）、机器人无效应答，说明服务体验存在系统性短板。",
        })

    # 4. 未解决工单风险
    unresolved = df[~df["is_resolved"]]
    if len(unresolved):
        high = unresolved[unresolved["priority"] == "高"]
        anomalies.append({
            "title": "未解决工单集中为高优先级退款/支付问题",
            "detail": f"共 {len(unresolved)} 条未解决，其中 {len(high)} 条为高优先级；"
                      f"平均满意度仅 {unresolved['satisfaction'].mean():.2f}，"
                      "多为退款/支付类资金问题，存在客户流失与投诉升级风险。",
        })

    # 5. 物流查询满意度偏低且处理慢
    logistics = df[df["category"] == "物流查询"]
    if len(logistics) and logistics["satisfaction"].mean() < df["satisfaction"].mean():
        anomalies.append({
            "title": "物流查询满意度偏低、处理偏慢",
            "detail": f"物流类平均满意度 {logistics['satisfaction'].mean():.2f}，低于总体均值 "
                      f"{df['satisfaction'].mean():.2f}；平均处理 {logistics['resolution_time_hours'].mean():.1f} 小时，"
                      "物流信息不透明是主要诱因。",
        })

    return anomalies


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame, anomalies: list) -> str:
    """生成结构化 Markdown 趋势报告。"""
    lines = []
    lines.append("# 客服工单趋势分析报告\n")
    lines.append(f"**数据范围**：{df['created_at'].min():%Y-%m-%d} ~ {df['created_at'].max():%Y-%m-%d}，共 {len(df)} 条工单\n")

    lines.append("## 一、总览指标\n")
    lines.append(f"- 工单总量：{len(df)} 条")
    lines.append(f"- 平均满意度：{df['satisfaction'].mean():.2f} / 5")
    lines.append(f"- 平均处理时长：{df['resolution_time_hours'].mean():.1f} 小时")
    lines.append(f"- 解决率：{df['is_resolved'].mean() * 100:.0f}%（{df['is_resolved'].sum()}/{len(df)}）\n")

    lines.append("## 二、类型分布\n")
    lines.append("| 分类 | 数量 | 占比 | 平均满意度 | 平均处理时长(h) |")
    lines.append("|------|------|------|------------|-----------------|")
    for cat, row in df.groupby("category").agg(
        n=("ticket_id", "count"),
        sat=("satisfaction", "mean"),
        rt=("resolution_time_hours", "mean"),
    ).sort_values("n", ascending=False).iterrows():
        lines.append(f"| {cat} | {int(row['n'])} | {row['n'] / len(df) * 100:.0f}% | "
                     f"{row['sat']:.2f} | {row['rt']:.1f} |")
    lines.append("")

    lines.append("## 三、时间趋势\n")
    lines.append("| 日期 | 工单量 |")
    lines.append("|------|--------|")
    for date, n in df.groupby("date").size().items():
        lines.append(f"| {date} | {n} |")
    lines.append("")

    lines.append("## 四、优先级与渠道\n")
    lines.append(f"- 优先级：高 {int((df['priority'] == '高').sum())}、中 {int((df['priority'] == '中').sum())}、"
                 f"低 {int((df['priority'] == '低').sum())}")
    lines.append(f"- 渠道：在线 {int((df['channel'] == '在线').sum())}、电话 {int((df['channel'] == '电话').sum())}、"
                 f"邮件 {int((df['channel'] == '邮件').sum())}\n")

    lines.append("## 五、异常信号\n")
    for i, a in enumerate(anomalies, 1):
        lines.append(f"### {i}. {a['title']}\n")
        lines.append(f"{a['detail']}\n")

    lines.append("## 六、结论与建议\n")
    lines.append("1. **优先修复支付系统重复扣款缺陷**：多笔重复/多扣款反复发生，涉及资金安全，风险最高。")
    lines.append("2. **优化退款流程时效**：退款处理时长显著异常且高频被投诉，建议设 SLA 阈值与自动催办。")
    lines.append("3. **提升客服服务质量**：投诉类满意度垫底，重点改善响应速度与人工兜底。")
    lines.append("4. **物流信息透明化**：物流类满意度偏低，建议主动推送物流异常预警。\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data()

    print("=" * 60)
    print("客服工单趋势分析")
    print("=" * 60)
    print(f"工单总量: {len(df)} | 平均满意度: {df['satisfaction'].mean():.2f} | "
          f"解决率: {df['is_resolved'].mean() * 100:.0f}%\n")

    # 生成图表
    plot_category_distribution(df)
    plot_daily_trend(df)
    plot_priority_satisfaction(df)
    plot_resolution_time(df)
    plot_satisfaction_by_category(df)
    plot_channel_distribution(df)
    plot_unresolved(df)
    plot_satisfaction_distribution(df)
    plot_dashboard(df)

    # 异常检测 + 报告
    anomalies = detect_anomalies(df)
    report = build_report(df, anomalies)
    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("发现异常信号:\n")
    for i, a in enumerate(anomalies, 1):
        print(f"  [{i}] {a['title']}")
        print(f"      {a['detail']}\n")

    print(f"图表已输出到: {OUT_DIR}")
    print(f"报告已输出到: {report_path}")


if __name__ == "__main__":
    main()
