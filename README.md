# POI 决策网关 · Jev demo

一个面向 **TikTok Local Services FDE 岗位**的 MVP：用 [Jev](https://www.typesafe.ai)（TypeSafe AI 的 System One 决策模型）解决 **POI 数据治理**里两个最高频、最耗人的问题——**POI 判重**与**商家提交审核预筛**。

> 核心洞察：POI 治理的本质是「海量、高频、结构化判断」，而不是文本生成。Jev 直接输出**校准过的概率 + 置信度**，天然支撑「置信度分级漏斗」——高置信自动处理、低置信升级人工，把人工成本砍掉一大截。

## 快速开始

```bash
pip install -r requirements.txt
python run.py
# 打开 http://127.0.0.1:8000
```

默认跑在**模拟模式**（内置一个确定性 mock，无需 API key 即可完整演示）。要接入真实 Jev：

```bash
export TYPESAFE_API_KEY="你的 key"   # Windows: set TYPESAFE_API_KEY=...
python run.py
```

拿到 key 后无需改任何代码——后端会自动从 mock 切换到真实 API（`POST https://api.typesafe.ai/v1/systemone`）。

## 项目结构

```
app/
├── config.py          # 环境变量配置（key / model / 置信度阈值 / 成本模型）
├── jev_client.py      # Jev API 封装 + 确定性 mock（无 key 也能跑）
├── synthetic_data.py  # 合成 POI 判重对 + 商家提交样本（带隐藏 ground truth）
├── decisions.py       # state 构造 + score 问题设计 + 置信度路由
├── main.py            # FastAPI 应用 + 评测统计
└── static/            # 纯 HTML/CSS/JS 前端（无构建步骤）
```

## 决策设计

每条样本只问 Jev 一个 `score` 问题（5 级有序 rubric），一次调用同时拿到**位置（方向）+ 置信度**：

| 流水线 | score 问题 | 路由规则 |
|---|---|---|
| POI 判重 | 「A 与 B 是同一物理地点的可能性？」<br>`definitely different → definitely same` | score ≥ 3 自动合并 · score ≤ 1 自动判重 · 否则升级人工 |
| 审核预筛 | 「这条商家提交从明确拒到明确过？」<br>`clearly reject → clearly approve` | score ≥ 3 自动通过 · score ≤ 1 自动拒绝 · 否则升级人工 |

置信度低于阈值（默认 0.65）一律升级人工——这就是「分级漏斗」：Jev 只对高置信样本做自动决策，模糊尾巴交给人。

## 评测与价值（/api/stats）

合成数据带隐藏 ground truth，所以能算出硬指标。当前 demo 的典型结果：

| 指标 | 判重 | 审核 |
|---|---|---|
| 自动决策覆盖率 | ~68% | ~74% |
| 自动决策准确率 | 100% | 100% |
| 全自动（无分级）准确率 | ~89% | ~87% |
| 升级人工比例 | ~32% | ~26% |

成本对比（全量）：**全人工 $25.5 → Jev 分级漏斗 $7.5，省 ~70%**。

这几个数字正好讲清楚 JD 第 4 条要的「用数据证明价值」：分级后自动决策的准确率显著高于无脑全自动，且只有一小部分模糊样本需要人。

## 面试可讲的点

1. **为什么用 Jev 而不是纯 LLM**：判重/分类是高频小判断，LLM 贵且慢、输出要 parse；Jev 输出结构化 + 校准概率，几百倍便宜、几十倍快，输出 token 免费。
2. **置信度分级漏斗**：Jev 处理大头 + 低置信升级（LLM 或人工），是 FDE 该有的架构判断，不是「Jev 替代 LLM」。
3. **Jev 的边界**：计数、日期比较、多步推理、文本生成这些不交给 Jev（比如坐标距离是在代码里用 haversine 算好再写进 state 的）。
4. **端到端闭环**：数据管道（合成数据）→ 模型调用 → 路由 → 轻量前端 → 评测 dashboard，覆盖 JD 第 2/3 条的完整链路。

## 下一步可扩展

- 场景 3 多语言归一（把判重问题改造成跨语言实体对齐）
- 场景 5 异常挖掘（对存量语料跑 `noul`/`score` 批量打风险分）
- 场景 6 用 Jev 当廉价 judge 搭评测回归
