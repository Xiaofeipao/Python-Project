# AIAA 3102 — Final Project: General Algorithm SuperAgent (Preview)

**This is a preview of the Final Project. The full assignment requirements will be released within one week.**

## 1. Project Overview

In this Final Project, each group is required to design and implement a **General Algorithm SuperAgent**.

Given a previously unseen task, its corresponding data, and a specified objective (classification in this Final Project), the Agent should autonomously carry out the entire algorithm development process. This includes analyzing the task and data, designing solutions, running experiments, refining its approach based on experimental results, and ultimately producing a runnable classification model.

Results included in the final report must be generated through complete, autonomous runs of the SuperAgent. During these runs, students are not allowed to manually modify code, decide what to try next, or select the final model.

You will be provided with an API key and a usage quota to use an existing LLM in HKUST(GZ) API platform for this project (But this doesn't mean you can only use it). More details will be released in the **full assignment requirements** .

## 2. Tasks

Groups will first have time to develop their SuperAgents. After an initial development period, the course will release several classification tasks for further testing and improvement.

The tasks will involve binary or multiclass classification using tabular, text, or image data. Data formats, evaluation data splits, and metrics will be announced with each task.

All tasks must be handled by the same SuperAgent. The Agent may select suitable methods based on the task, data, and experimental results, but must not use task names, dataset names, or other task identity information to invoke manually prepared, task-specific solutions.

## 3. SuperAgent Design

Each group is free to design its own SuperAgent implementation.

During execution, the Agent may autonomously select its system architecture, algorithms, and tools, generate or modify code, and organize model training and experiments.

The SuperAgent should use the current task, data characteristics, and previous experimental results to decide what to try next, and conduct further experiments to evaluate its changes. Its solution choices, modifications, and execution results should be recorded in the execution logs.

## 4. Resource Constraints

Specific resource constraints, including permitted models/APIs and usage budgets, will be announced in the full assignment requirements.

Results included in the final report must be generated autonomously by the SuperAgent within these limits, with corresponding execution and resource usage records.

## 5. Evaluation and Grading

Final grades will be based on the submitted SuperAgent, experimental results on the released tasks, source code and execution logs, project report, and Final Presentation.

| Component                                          | Weight   | Assessment                                                                                                                            |
| -------------------------------------------------- | --------:| ------------------------------------------------------------------------------------------------------------------------------------- |
| Performance                                        | 30%      | Task completion and final results, including valid partial results from incomplete tasks.                                             |
| General SuperAgent Design & Autonomous Development | 30%      | Generality, autonomous experimentation, and the ability to revise and improve solutions using experimental feedback.                  |
| Execution, Reproducibility & Report                | 10%      | Autonomous execution, compliance with resource constraints, reproducibility of experiments and results, and report quality.           |
| Final Presentation                                 | 30%      | Presentation of the system design and experimental results, and group members’ understanding of the submitted system and experiments. |
| **Total**                                          | **100%** |                                                                                                                                       |

Groups must report results according to the evaluation data splits, metrics, and requirements announced for each task. The SuperAgent may design its own internal validation procedures, but **the designated test set must be used only for final evaluation, not for training, hyperparameter tuning, or solution selection**.

## 6. Submission Requirements

Each group must submit:

1. **Complete SuperAgent source code and a unified execution entry point**.
2. **`requirements.txt` and `README.md`**, documenting dependencies, environment setup, and execution instructions.
3. **The final classification models or runnable prediction solutions generated for each task, together with the corresponding result files**. Final solutions must include the preprocessing steps and code required for prediction.
4. **`report.pdf`**, describing the system design, main experiments, results, and limitations.
5. **Execution logs and experimental records**, including candidate solutions, experimental results, key decisions, and resource usage.
6. **An AI tool usage statement**.

All reported results must be generated using the same final submitted version of the SuperAgent and must be consistent with the submitted solutions and execution logs. For incomplete tasks, groups should explain the results obtained, where execution failed, and the reasons for failure. The course may inspect the code and reproduce selected results.

## 7. AI Policy

AI tools may be used to assist with project development, including system design, coding, debugging, and testing.

All group members are responsible for understanding and explaining the submitted SuperAgent, its key design choices, and experimental results, and are accountable for their submitted work.
