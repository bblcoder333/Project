# Sample outputs (Qwen2-VL + Qwen2.5)

One representative result per phase, all for the same input screen: ENRICO screen **11794**, a login screen of a news app (topic label `login`). The files come from Run 1 of the Qwen2-VL + Qwen2.5 configuration. They are shown unedited so reviewers can follow one screen through the whole pipeline. Each phase reads only the previous phase's output. The prompts that produced them are in [`../PROMPTS.md`](../PROMPTS.md).

| Phase | File | Prompt in `PROMPTS.md` | What the output shows |
| --- | --- | --- | --- |
| Input | `qwen/input_screenshot.jpg` | n/a | The screenshot given to the VLM. |
| 1. UI Perception | `qwen/phase1_ui_description.json` | Phase 1 | A text description of the screen in the fixed A to F structure plus a recap. |
| 2. Source Test Generation | `qwen/phase2_test_cases.csv` | Phase 2 | Seven source test cases. |
| 3. MR Generation | `qwen/phase3_metamorphic_relations.csv` | Phase 3 | Four metamorphic relations (MRs) derived from the test cases. |
| 4. MR Prioritization | `qwen/phase4_prioritization.csv` | Phase 4 | One decision and a reason per MR. |
| 5. Execution Scheduling | `qwen/phase5_execution_schedule.csv` | Phase 5 | An execution tier and cycle frequency per MR. |

## Phase 1: UI Perception

The VLM turns the screenshot into text, which is the only information later phases receive. For this screen it identifies a login page, lists the app bar with back arrow and "Log In" title, the logo, the email and password fields, the orange "Log In" button and the "Forgot your password?" link, and adds colors, typography estimates and accessibility notes (it flags low-contrast field text).

*Observation:* the password field in the screenshot is masked, but the description reports a readable password value, which Phase 2 then reuses as test data. This shows how a Phase 1 error can propagate to later phases. The results workbook reports the number of usable Phase 1 descriptions separately (`Phase1-3_Summary`).

## Phase 2: Source Test Generation

The LLM writes manual test cases using only elements named in the description. The seven cases here are: four functional (the "Log In" button, valid email and password entry, the "Forgot your password?" link, the back arrow), one negative (invalid email format), and two accessibility cases (contrast of the field text and touch-target size of the "Log In" button). Every case has at least one action step, apart from the accessibility cases, which are allowed "Verify" steps by the prompt's rules.

## Phase 3: MR Generation

Each test case is matched to a pattern and turned into follow-up MRs. The negative test (invalid email) produces one `VALIDATION_CONSISTENCY` MR: change the invalid value to a different invalid value and expect the same validation error. The three test cases that are a single tap (Log In, Forgot password, back arrow) each produce a `ROBUSTNESS` MR: repeat the tap and expect no crash or inconsistent state. Four MRs in total, across two categories.

## Phase 4: MR Prioritization

Each MR receives one decision. The validation MR is `high_priority_keep` because it has a computable expected result (an invalid input must stay invalid). The three robustness MRs are behavior-stability checks, which the prompt caps at `keep`: the first is kept, one is marked `reduce_repetitions` by the deterministic coverage rule ("exact duplicate ROBUSTNESS MR on the same screen"), and one is `partial_sampling` as the same pattern on a different element. Rows are not in MR-number order.

## Phase 5: Execution Scheduling

Each MR gets an execution tier within the tiers allowed for its Phase 4 decision, plus a cycle frequency:

| MR | Phase 4 decision | Tier | Runs every |
| --- | --- | --- | --- |
| MR-01 | high_priority_keep | always | cycle |
| MR-02 | keep | sampled | 2nd cycle |
| MR-04 | reduce_repetitions | optional | 10th cycle |
| MR-03 | partial_sampling | sampled | 5th cycle |

Here every LLM proposal was already within the allowed tiers, so the deterministic clamping step changed nothing (a clamped row's reasoning would state it, as in the Phase 5 evidence in the results workbook). With the assumed cost of 0.0005 kWh per MR execution, running all four MRs every cycle costs 0.0020 kWh, and this schedule costs 0.0009 kWh per cycle on average, a projected saving of 55% for this screen. Across all screens, SSR is computed from the summed full-execution and scheduled costs.
