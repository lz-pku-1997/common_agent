# LLM Judge 人工校准记录表

这张表用于人工抽查 `answer_quality.py` 的模型裁判是否合理。对照任务、实际工具返回、文件前后状态和最终回答；裁判分数不是客观真值。

## 评分口径

三项各给整数0、1或2分，分别使用下表中的依据：

| 维度及依据 | 0分 | 1分 | 2分 |
|---|---|---|---|
| correctness：事实正确性；裁判用工具结果和文件状态核实真实情况 | 关键事实错误或操作未成功却声称成功 | 核心结论正确但个别次要事实不准确 | 回答所述事实符合实际结果，未知事项如实说明 |
| completeness：回答完整性；对照任务及expected_points必答清单 | 未回应核心要求 | 回应部分内容但遗漏必答要点 | 覆盖全部必答要点 |
| coherence：表达连贯性；看最终回答自身 | 难以理解或严重自相矛盾 | 基本能懂但有歧义或组织不清 | 意思清楚、前后一致、容易理解 |

`expected_points`只列要求交代的内容，不充当三项评分的混合标签。遗漏内容只扣完整性；事实说错只扣正确性；如果同时造成表达歧义或回答自身矛盾，才按表达问题另扣连贯性。例如没提保存位置属于完整性问题；保存失败却声称成功属于正确性问题。事实材料给裁判核实答案使用，不按回答有没有展示依据评分；任务要求文件出处时，才在完整性里检查出处。

三项相加，每题满分6分，不设及格线。总分只作摘要，不能用表达好或内容全掩盖正确性0分。工具失败但如实解释可以得到高分；空回答按零分。明确标注的推测和合理建议不当作已发生的事实。不要因为表达风格不同或使用同义表达扣分，不要求用户未要求的步骤复述。

## 抽查记录

从 JSON 报告抽查原始回答，将人工判断与模型裁判并排记录：

```text
报告文件：
样例 ID：
回答模型 / 裁判模型：
人工评分：correctness __/2 / completeness __/2 / coherence __/2
人工总分：__/6
模型裁判评分与理由：
是否同意裁判：是 / 否
分歧原因 / rubric 修改建议：
```

如修改了 rubric 或参考要点，保存为新的数据版本并重新运行；不要覆盖原始报告。少量人工抽查只能用于发现明显偏差，不能宣称已经证明裁判与人类高度一致。

## 选择依据

2026-10-12查阅以下三家官方指标清单。这是官方定义覆盖核对，不是使用量排名；相关指标的定义和量表各家不同。本项目按四道文件任务明确数据来源和0/1/2档位。

- [AWS Bedrock指标](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-metrics.html)：列出Faithfulness、Completeness、Coherence，以及Correctness、Relevance、FollowingInstructions等。
- [Microsoft Foundry指标](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-metrics-built-in)：列出Groundedness、Response Completeness和Coherence，并提供Relevance、Fluency等。
- [Google模型评分模板](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/metrics-templates)：包含Groundedness、Coherence模板；问答质量模板包含Completeness。

本项目选择Correctness、Completeness、Coherence：评价回答事实是否正确、有没有答全、表达是否连贯。Groundedness强调回答与给定上下文的支持关系；本项目用实际执行记录核实回答事实，采用Correctness名称更直接。相关性、指令遵循、简洁性也常见，但本次没有另增评分项。旧报告中的factual/responsive/clear或groundedness是旧Rubric，保留原始字段，不与新分数直接当作同一指标对比。
