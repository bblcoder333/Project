# Experimental Results Workbook

The file `GitHub_Master_Results_Final.xlsx` contains the experimental measurements, intermediate evidence, and metric calculations used to evaluate the proposed five-phase metamorphic testing pipeline. The workbook provides traceability from per-screen, phase-level measurements to the five primary metrics reported in the paper:

- Energy Consumption (EC)
- Carbon Emissions (CE)
- Metamorphic Relation Coverage (MRC)
- Deterministic Intervention Rate (DIR)
- Scheduling Savings Ratio (SSR)

The evaluation covers the same fixed set of 500 ENRICO UI screenshots across five VLM–SLM configurations. The common evaluation run for each configuration is used for the cross-model results reported in the manuscript.

## Evaluated model configurations

| Configuration | Vision-language model | Small language model |
| --- | --- | --- |
| Qwen2-VL + Qwen2.5 | Qwen/Qwen2-VL-7B-Instruct | Qwen/Qwen2.5-7B-Instruct |
| Phi-3.5-Vision + Phi-3.5-Mini | microsoft/Phi-3.5-vision-instruct | microsoft/Phi-3.5-mini-instruct |
| InternVL2 + InternLM2.5 | OpenGVLab/InternVL2-8B | internlm/internlm2_5-7b-chat |
| InternVL2 + Mistral | OpenGVLab/InternVL2-8B | mistralai/Mistral-7B-Instruct-v0.3 |
| InternVL2 + Llama-3.1 | OpenGVLab/InternVL2-8B | meta-llama/Meta-Llama-3.1-8B-Instruct |

The complete model mapping is also provided in the `Model_Configurations` worksheet.

## Workbook organization

The five pipeline phases are: UI Perception, Source Test Case Generation, Metamorphic Relation Generation, MR Prioritization, and Execution Scheduling.

| Worksheet | Contents |
| --- | --- |
| `Qwen_Final` | Per-screen EC and CE measurements for Qwen2-VL + Qwen2.5 across all five phases. Each row is one evaluated ENRICO screen. |
| `Phi_Final` | The corresponding five-phase per-screen EC and CE measurements for Phi-3.5-Vision + Phi-3.5-Mini. |
| `InternVL_InternLM` | Five-phase per-screen EC and CE measurements for InternVL2 + InternLM2.5. |
| `InternVL_Mistral` | Five-phase per-screen EC and CE measurements for InternVL2 + Mistral. |
| `InternVL_Llama` | Five-phase per-screen EC and CE measurements for InternVL2 + Llama-3.1. |
| `Model_Configurations` | The exact VLM and SLM identifiers used in each configuration, with model size information and links to the corresponding measurement worksheets. |
| `Phase1-3_Summary` | Pipeline progression through the first three phases for the 500 screens: usable Phase 1 UI descriptions, screens producing at least one source test case, total source test cases, screens producing at least one final MR, and the resulting MRC. A compact view of how successfully each configuration progresses from UI perception to MR generation. |
| `Metric_Definitions` | The five evaluation metrics and the formulas used in the workbook, with interpretation notes (below). |
| `MR_Coverage_Proof` | Screen-level evidence for MRC. Each of the 500 screens has a binary indicator per configuration: 1 = at least one final Phase 3 MR is available, 0 = none. The sum of the indicators gives MRC. |
| `Phase5_Raw` | Detailed Phase 5 scheduling evidence for DIR and SSR: screen identifier, number of MRs entering scheduling, model-proposed scheduling decisions, deterministic clamping, fallback information, final execution tiers and frequencies, and projected optimized and unoptimized recurring execution cost. |
| `Metrics_Calculation` | The formula-based calculations deriving the five primary metrics. Phase-level EC and CE are aggregated from the model worksheets; MRC, DIR and SSR are computed from the supporting evidence sheets. This is the main calculation trace from raw evidence to reported results. |
| `Five_Metric_Summary` | The consolidated results for all five configurations, corresponding to the primary results reported in the manuscript. |
| `Repeatability_Check` | A separate three-run repeatability analysis for Qwen2-VL + Qwen2.5 (see below). |

## Metrics

| Metric | Meaning |
| --- | --- |
| EC | Measured generation energy across Phases 1–5 |
| CE | Corresponding measured carbon emissions |
| MRC | Proportion of evaluated screens for which Phase 3 produces at least one final validated MR |
| DIR | Proportion of checked Phase 5 scheduling proposals modified by deterministic scheduling constraints |
| SSR | Projected reduction in recurring MR execution cost produced by the final execution schedule |

SSR is a projection, not a measurement: it uses an assumed execution cost of 0.0005 kWh per MR execution (set in `stages/optimization_reduction.py`), so it describes relative savings from scheduling rather than measured device energy. EC and CE are measured with CodeCarbon.

## Repeatability check

The `Repeatability_Check` worksheet provides a three-run analysis for Qwen2-VL + Qwen2.5. Runs 1, 2 and 3 contain per-screen EC and CE measurements for all five phases under the same experimental conditions, as an illustrative check of run-to-run measurement stability.

The analysis is supplementary and is not used to average, replace or modify the primary cross-model results. The other four configurations remain part of the common cross-model evaluation in their own worksheets, so the Qwen experiment should be read as a stability check rather than an additional model-comparison criterion.

## Relationship between the workbook and the paper

The workbook supports the reported results at three levels:

1. **Raw evidence:** per-screen phase measurements, MR-coverage indicators and Phase 5 scheduling records.
2. **Metric calculation:** `Metrics_Calculation` applies the definitions in `Metric_Definitions` to the raw evidence.
3. **Reported results:** `Five_Metric_Summary` consolidates EC, CE, MRC, DIR and SSR for direct comparison with the manuscript.

This allows every reported value to be traced back to its measurements and intermediate evidence.

## Reproducibility notes

- The same fixed set of 500 ENRICO screenshots is used for all evaluated configurations.
- Each screenshot is processed through the five pipeline phases.
- Screen identifiers are retained to support traceability across phases and result files.
- Energy and carbon are measured using CodeCarbon 3.2.8.
- MRC is calculated from final Phase 3 MR availability.
- DIR and SSR are derived from the Phase 5 scheduling evidence.
- The Qwen three-run experiment is provided separately as supplementary repeatability evidence.
- Missing measurements are not replaced with assumed or synthetic values. The only assumed quantity is the per-MR execution cost used for the projected SSR.
- Prompts, generated artifacts, screen identifiers and implementation files in this repository provide further support for reproducing and inspecting the pipeline.
