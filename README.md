# POI 数据治理决策平台 · Laya + LLM 分级决策

面向 **TikTok Local Services 反作弊/风控（FDE）** 岗位的端到端 demo：用「快而准的决策模型（Laya）+ 慢而深的 LLM（DeepSeek）」两级决策，解决本地生活 POI 治理里两个最高频、最耗人的判断——**内容审核预筛** 与 **POI 判重**。

> 核心洞察：POI 治理的本质是海量、高频、结构化的「小判断」，不是文本生成。System 1（Laya）直接输出校准概率 + 置信度，只对高置信样本自动决策，模糊尾巴升级人工——这就是**置信度分级漏斗**。

## 三级决策流水线

一次商家提交走三步：

```
商家提交
  │
  ▼
① 内容审核（Laya · System 1）
   描述文本 → moderation 预设 → approve / reject（+ 每项违规信号 p）
  │
  ▼
② POI 判重（blocking → LLM · System 2）
   品牌/地理分块 → DeepSeek 判断 → new / merge / uncertain
  │
  ▼
③ 置信度路由
   auto_reject / auto_merge / auto_create / escalate（升级人工）
  │
  ▼
   SQLite 持久化（places 写库 + audit 决策日志）
```

## 为什么这样设计

1. **决策模型 ≠ 通用 LLM**：审核/判重是高频小判断，Laya 输出结构化（倾向 + 置信度），比 LLM 快、便宜几个数量级，且免解析。
2. **分级而非全自动**：全自动必然错落地（把两家店合错、把垃圾店放进池都是真损失）；分级让高置信自动、低置信交人。
3. **证据与判断分离**：距离（haversine）、名称相似度、品牌/分类/电话匹配在代码里算好当证据；模型只「综合证据下判断」，不负责算数。（见 `DESIGN.md` §2）
4. **System-2 兜底难题**：同一品牌不同分店（same brand, different branch）是确定性特征层分不开的，交给 LLM 用世界知识判断，且只对分块候选短名单跑，绝不全库。

## 快速开始

```bash
pip install -r requirements.txt     # 轻依赖（fastapi/uvicorn/openai）
# 重依赖（Laya 的 torch/transformers + Overture 的 pyarrow/shapely）需另行安装，见 requirements.txt 注释

# 可选：配置 DeepSeek key（不配则判重自动降级为离线规则）
copy .env.example .env              # 然后填入 DEEPSEEK_API_KEY

python run.py                       # 打开 http://127.0.0.1:8000
# Windows 一键启动：双击 start.bat
```

后端：内容审核固定用 **Laya**（`convaiinnovations/laya`，多语 mmBERT 子目录，自托管、Apache-2.0、与 Jev 兼容）；判重用 **DeepSeek**（有 `DEEPSEEK_API_KEY`）或离线规则（无 key 自动降级，状态栏判重后端显示 `rules (offline)`）。

## 可解释性 / 归因（亮点）

每个决策都能说清「为什么」：

- **审核**：`spam / toxic / harassment / threat` 四项违规概率逐条展示，取最高 p 驱动动作。
- **判重**：每个候选 POI 显示 `name / distance / brand / category / phone` 的证据权重（与 `feature_hint` 一致，条形之和等于模型先验）；merge 时给出「why same — evidence」面板。

这让「模型做了个决定」变成「模型给了哪些证据、各占多少」——正是风控场景人审与申诉所需。

## 持久化 + 实时索引

- Overture 真实 POI（`overture_london.parquet`，GeoParquet）作为只读种子。
- SQLite（`poi.db`，stdlib `sqlite3` 零依赖）：`places` 存自动创建的新店，`audit` 存每一次决策（时间戳 + 审核/判重/路由结果）。
- **判重实时扫 `places` 表**：刚 `auto_create` 的店，下一秒就进入判重索引——连续提交重复店会被抓成 merge，而不是各存一条。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 后端/模型/判重后端/置信度阈值/库状态 |
| GET | `/api/examples` | 前端预填样例（重复 / 垃圾 / 新店） |
| GET | `/api/audit` | 决策日志（最近 N 条） |
| POST | `/api/submit` | 跑一条完整三级流水线 |

## 项目结构

```
app/
├── chain.py          # 编排：content gate → dedup → route（canonical）
├── content_gate.py   # Laya moderation 预设 → approve/reject + 每项信号
├── llm_judge.py      # System-2 判重 judge（DeepSeek，OpenAI 兼容）
├── pipeline.py       # find_matches 分块（品牌相等 / 地理 ≤0.3km，topk=5）
├── blocking.py       # 批量候选对分块（O(n²)→分块，离线上量）
├── features.py       # 确定性证据 + explain() 归因
├── models.py         # 字段读取工具
├── overture.py       # Overture GeoParquet 摄取
├── store.py          # SQLite 持久化（places + audit）
├── config.py         # 环境变量配置（backend / 阈值 / 成本模型）
└── static/           # 纯 HTML/CSS/JS 前端（无构建）
```

根目录另有开发期脚本：`run_*.py`（各环节 CLI）、`eval_laya*.py` / `probe_*.py` / `tune_threshold.py`（阈值标定与预设探针）、`fine_tune.py`（Laya 微调）、`synthetic_data.py` / `perturb.py`（合成对生成）。

## 已知局限（诚实的边界）

- **审核域错配**：多语 `spam` 预设是在用户「帖子」上训练的，用在「商家 listing 描述」上会过触发——药房、书店、理发店等正常的促销性描述可能被误判为 spam（尤其中文商户文案）。商家 listing 本身就是广告，「spam」是问错的问题；正确解法是 listing 专用 rubric 或改用英文 checkpoint。这正是校准/域理解要抓的点。
- **判重分块依赖品牌/坐标**：无品牌且无坐标的提交会漏进分块（blocking 召回下限），生产需补电话/地址分块键。
- **置信度阈值写死 0.65**：生产应像 `tune_threshold.py` 那样用评测集标定，而非拍脑袋。

## 面试可讲的点

1. **两级决策（System-1 + System-2）**：Laya 吃高频文本判断、LLM 吃需要世界知识的判重，各取所长，成本/延迟分层。
2. **置信度分级漏斗**：高置信自动、低置信升级，是风控该有的架构判断，不是「模型替代人」。
3. **决策模型的边界**：计数、坐标换算、字符串匹配不交给模型——代码里算好当证据喂进去。
4. **端到端闭环 + 可解释**：接入 → 决策 → 路由 → 写库 → 前端 dashboard，每个决定都能追溯到证据。

## 下一步可扩展

- 电话/地址纳入分块键（提升无坐标提交的召回）
- 对抗样本（同形字 / 全角 / 品牌仿冒）鲁棒性
- 校准曲线（reliability diagram / ECE）检验置信度
- 升级队列 + 人审回流闭环
- 速度/行为信号（批量提交、异常频率）接入风控维度
