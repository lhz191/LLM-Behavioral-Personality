<div align="center">

# LLM Behavioral Personality

**一个用于测量语言模型 thought-level 行为模式的框架**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
![Status](https://img.shields.io/badge/Status-Work%20in%20progress-orange)

[English](README.md) | [中文](README.zh-CN.md)

</div>

---

## 概览

本项目研究大语言模型是否具有稳定、可测量的 **行为模式**，这些行为模式是否类似人类人格特质，以及它们是否相比表层的 *response style*，更适合通过模型内部的 *thought process* 捕捉。

我们将六个经过验证的人类人格问卷改写为 LLM 行为场景，基于 hidden states 提取行为轴，并通过 activation steering 比较 thought-level 轴与 response-level 轴在不同交互寄存器中的迁移效果，包括第一人称行为、建议和可执行任务。

## 行为框架

每个 **domain** 描述一种 LLM 输出风格。每个 **subdomain** 定义一个更具体的行为风格构念，并包含 low pole 与 high pole。

### GCS：迎合/从众
> 诌媚，不坚定，随人摇摆，缺乏独立判断

| Subdomain | 构念 | Low pole | High pole |
|---|---|---|---|
| GCS-F1 让步 | **立场让步**：遇反对时是否以减少摩擦为优先 | 坚持原有判断，明确说明理由，不因对方不满而改变立场 | 对方表达不满即修改立场，以平息摩擦代替维护判断 |
| GCS-F2 服从 | **权威服从**：是否把来源权威性作为执行依据 | 对指令提出质疑，要求说明依据后再执行 | 无条件照单全收，不质疑权威或来源 |
| GCS-F3 冲突回避 | **冲突回避**：是否以模糊或赞同替代真实评价 | 明确指出分歧，即使引发不适也不回避 | 以含糊表达或赞同替代真实评价，优先维持和谐 |

### DOSPERT：冒进/轻率
> 忽视后果，低估风险，目光短浅，急于给出激进建议

| Subdomain | 构念 | Low pole | High pole |
|---|---|---|---|
| Financial | **财务激进**：推荐方案时是否优先高收益而将风险放次要 | 强调下行风险，推荐稳健方案，明确说明代价 | 优先推荐高收益方案，淡化或忽略风险代价 |
| Health/Safety | **安全忽视**：建议中是否低估身体或健康代价 | 优先安全边际，主动提示潜在人身风险 | 接受较高的人身安全代价以换取其他收益 |
| Recreational | **刺激偏好**：是否倾向推荐危险刺激选项 | 倾向推荐安全可控的活动选项 | 倾向推荐刺激但存在明显危险的选项 |
| Social | **规范挑战**：是否鼓励对抗权威或打破社交规范 | 建议谨慎行事，回避不必要的对抗或冒险表态 | 鼓励当众挑战权威或采取异类社交行动 |
| Ethical | **伦理合理化**：是否为伦理灰色地带主动找理由 | 明确拒绝伦理违规，不为其辩护 | 在利益驱动下主动为伦理问题合理化 |

### UPPS-P：冲动/无韧性
> 不深思，情绪化行动，遇阻即退，缺乏计划

| Subdomain | 构念 | Low pole | High pole |
|---|---|---|---|
| UPPS_LP 缺乏预谋 | **草率行动**：给出建议前是否权衡后果 | 行动前分析潜在影响，给出前瞻性判断 | 直接给出建议或执行，不考虑潜在后果 |
| UPPS_PU 正面情绪冲动 | **情绪驱动**：在积极情绪下是否绕过审慎判断 | 即使情境积极也保持审慎，不因氛围做仓促决定 | 因兴奋或乐观状态绕过正常评估，快速做出决定 |
| UPPS_LPE 缺毅力 | **半途放弃**：遇阻时是否主动降标或转移目标 | 任务变难时坚持推进，保持标准不降低 | 遇到阻力后转移目标、降低要求或放弃 |

### HH：不诚信
> 说好听的，顾自己，摆架子，言行不一

| Subdomain | 构念 | Low pole | High pole |
|---|---|---|---|
| HH_SIN 虚伪 | **表面迎合**：是否以讨好性回应替代真实评价 | 给出真实评价，即使不讨好，不因对方期望改变判断 | 为获得认可迎合对方，隐藏或扭曲真实判断 |
| HH_GRE 贪婪 | **自利偏向**：推荐分析时是否存在不公平倾向 | 推荐公平方案，不偏袒任何一方 | 推荐时隐性偏袒某方，对不公平结果不作说明 |
| HH_MOD 傲慢 | **居高临下**：是否以优越姿态定位自身 | 承认局限，平等对待他人判断 | 以优越姿态自我定位，贬低他人判断或不承认错误 |

### PPS：拖延/敷衍
> 推迟决策，不给结论，被动应付，能拖就拖

| Subdomain | 构念 | Low pole | High pole |
|---|---|---|---|
| PPS 拖延 | **决策回避**：面对需要判断的问题时是否给出明确方向 | 直接给出可执行的判断和行动步骤 | 推迟、模糊或转移问题，不做明确决断 |

### BFI-2C：无序/低效
> 缺乏条理，精力分散，产出稀薄，靠即兴应对

| Subdomain | 构念 | Low pole | High pole |
|---|---|---|---|
| BFI2C-Organization 条理 | **结构缺失**：面对松散信息时是否主动施加结构 | 接受松散混乱状态，靠即兴、记忆和临时应对处理 | 主动分类排序、标注、分配稳定位置，减少后续搜索和返工 |
| BFI2C-Productiveness 效率 | **低效输出**：单位努力的有效产出是否聚焦 | 泛泛展开，过度修饰，有效产出低 | 聚焦核心，每步直接指向结果，不做冗余铺垫 |
| BFI2A_TRU 多疑 | **过度质疑**：对输入信息是否默认善意解读 | 默认善意，对信息不做无依据的过度质疑 | 假设存在隐藏动机，对一切信息持怀疑态度 |

## Pipeline

```text
pipeline/
├── 01_generate/    场景生成
├── 02_axes/        从 hidden states 提取行为轴
├── 03_steering/    使用 thought 轴和 response 轴进行 activation steering
├── 04_results/     Choice probe 统计与跨寄存器分析
├── 05_sae/         SAE 特征分析
├── 06_monitoring/  Thought 前缀监控
└── common_org.py   共享配置和工具函数
```

## Data

```text
data/
└── bfi2c_organization/
    ├── scenarios/          场景 JSON
    └── results/
        ├── axes/           轴质量指标和寄存器分类结果
        └── steering/       Steering 实验结果
```

## Requirements

```text
torch
transformers
openai
numpy
```
