# AIAA 3102 — Final Project: General Algorithm SuperAgent

## 1. Project Overview

In this Final Project, each group of **2–4 students** is required to design and implement a **General Algorithm SuperAgent**.

Given a previously unseen task, its corresponding data, and a specified objective (classification in this Final Project), the Agent should autonomously carry out the entire algorithm development process. This includes analyzing the task and data, designing solutions, running experiments, refining its approach based on experimental results, and ultimately producing a runnable classification model.

Results included in the final report must be generated through complete, autonomous runs of the SuperAgent. During these runs, students are not allowed to manually modify code, decide what to try next, or select the final model.

You will be provided with an API key and a usage quota to use an existing LLM on the HKUST(GZ) API platform for this project (but you may also use other API providers).

## 2. Tasks

Groups will first have time to develop their SuperAgents. After an initial development period, the course will release several classification tasks for further testing and improvement.

The tasks will involve binary or multiclass classification using tabular, text, or image data. Data formats, evaluation data splits, and metrics will be announced with each task.

All tasks must be handled by the same SuperAgent. The Agent may select suitable methods based on the task, data, and experimental results, but must not use task names, dataset names, or other task identity information to invoke manually prepared, task-specific solutions.

After both sections of this course (L01 and L02) have completed their Final Presentations, one previously unreleased classification task (the **Unseen Task**) will be provided to evaluate the submitted SuperAgent on a new task.

## 3. SuperAgent Design

Each group is free to design its own SuperAgent implementation.

During execution, the Agent may autonomously select its system architecture, algorithms, and tools, generate or modify code, and organize model training and experiments.

The SuperAgent should use the current task, data characteristics, and previous experimental results to decide what to try next, and conduct further experiments to evaluate its changes. Its solution choices, modifications, and execution results should be recorded in the execution logs.

## 4. Resource Constraints

SuperAgents may only use **Qwen or DeepSeek models officially released on or before October 21, 2026, with official standard API pricing**. These models may be accessed through any API provider or deployed locally.

For each evaluation task included in the final report, the **model usage budget is limited to RMB 20**. Usage costs will be calculated using the models' official standard input and output token prices as of October 21, 2026, and actual token consumption, without considering cache discounts, platform promotions, or free credits. Locally deployed models are subject to the same accounting rules.

Each group must accurately record all agents' model calls, input and output token usage, and calculated costs, and submit the corresponding execution logs. **The course will conduct spot checks of these records.** Dishonest reporting of model usage will result in grade deductions.

## 5. Evaluation and Grading

Final grades will be based on the submitted SuperAgent, experimental results on the released and unseen tasks, source code and execution logs, project report, and Final Presentation.

| Component                                          | Weight   | Assessment                                                                                                                            |
| -------------------------------------------------- | --------:| ------------------------------------------------------------------------------------------------------------------------------------- |
| Performance — Released Tasks                       | 20%      | Task completion and final results on the released tasks, including valid partial results from incomplete tasks.                       |
| Performance — Unseen Task                          | 10%      | Task completion and final results on the unseen task, including valid partial results from incomplete tasks.                          |
| General SuperAgent Design & Autonomous Development | 30%      | Generality, system design, autonomous experiment planning, and feedback-driven improvement.                                           |
| Execution, Reproducibility & Report                | 10%      | Successful end-to-end execution, compliance with resource constraints, reproducibility, and report quality.                           |
| Final Presentation                                 | 30%      | Presentation of the system design and experimental results, and group members’ understanding of the submitted system and experiments. |
| **Total**                                          | **100%** |                                                                                                                                       |

Groups must report results according to the evaluation data splits, metrics, and requirements announced for each task. The SuperAgent may design its own internal validation procedures, but **the designated test set must be used only for final evaluation, not for training, hyperparameter tuning, or solution selection**.

## 6. Submission Requirements and Schedule

Each group must submit:

1. **Complete SuperAgent source code and a unified execution entry point**.
2. **`requirements.txt` and `README.md`**, documenting dependencies, environment setup, and execution instructions.
3. **The final classification models or runnable prediction solutions generated for each released task, together with the corresponding result files**. Final solutions must include the preprocessing steps and code required for prediction.
4. **`report.pdf`**, describing the system design, main experiments, results, and limitations.
5. **Execution logs and experimental records**, including candidate solutions, experimental results, key decisions, and resource usage.
6. **An AI tool usage statement**.

All reported results must be generated using the same final submitted version of the SuperAgent and must be consistent with the submitted solutions and execution logs. For incomplete tasks, groups should explain the results obtained, where execution failed, and the reasons for failure. The course may inspect the code and reproduce selected results.

The final project schedule is as follows:

| Stage                      | Schedule                                                                                                                                                                                                                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Final Submission**       | Submit the SuperAgent source code, report, and Released Tasks results by **23:59 on the evening before the first class of the final week**, according to each section's class schedule. The submitted SuperAgent version is then frozen.                                                 |
| **Final Presentation**     | Groups present during the final week, according to their section's class schedule.                                                                                                                                                                                                       |
| **Unseen Task Evaluation** | Once both sections (L01 and L02) have completed their presentations, the course will release one previously unreleased classification task. Groups will have **24 hours** to run their submitted SuperAgent and submit predictions and execution logs. No additional report is required. |

The Unseen Task must be completed using the SuperAgent version frozen at Final Submission. No human intervention is allowed during execution. The same resource constraints apply.

## 7. AI Policy

AI tools may be used to assist with project development, including system design, coding, debugging, and testing.

All group members are responsible for understanding and explaining the submitted SuperAgent, its key design choices, and experimental results, and are accountable for their submitted work.
