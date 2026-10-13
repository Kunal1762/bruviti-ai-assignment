# AI.md — Use of AI Assistance

I used AI tools as development assistants while building and evaluating the fault-classification service. I reviewed the suggestions, ran the code and tests, and made the final decisions myself.

## Antigravity

I used Antigravity to iterate on the synthetic complaint dataset and classifier evaluation. This included reviewing evaluation results, identifying weak classes and difficult examples, adjusting scenario variety and data-generation choices, and checking how those changes affected the measured Macro-F1. The goal was to build a more useful evaluation set—not simply to maximize a score. I documented limitations such as synthetic-data bias and the need to keep the test set independent.

I evolved the dataset iteratively rather than relying on a single generation. I started by evaluating the initial synthetic complaint dataset, then identified gaps in class coverage, ambiguous complaints, and difficult edge cases. I refined the data-generation approach to produce a larger and more diverse dataset, including variations in complaint wording and examples that could be confused across fault categories.

I also focused on making the evaluation more meaningful by checking class-level performance and adding harder examples, such as vague descriptions, overlapping symptoms, and error-code-only complaints. I kept scenario-level separation between training, development, and test splits to reduce the risk of similar generated examples leaking across splits, and ensured that metadata such as `scenario_id` was not used as a model feature.

The iteration was guided by evaluation results, not just the overall Macro-F1 score. I treated the results cautiously because repeatedly inspecting test performance during dataset development can make the final score optimistic.


## ChatGPT

I used ChatGPT to help review the dataset and data-generation logic, reason about split leakage and scenario coverage, identify gaps in hard-case evaluation, and plan improvements to the classifier and rule engine. It also helped clarify the repository documentation, including `DATA.md` and `NOTES.md`, and gave practical Git guidance.

## Claude 

I used Claude (Sonnet 5.5, via the Claude chat interface) as a development assistant for the following.

- **First synthetic-data generator.** Claude wrote an early `data.py` that generated labelled complaints from phrase banks, with CSV output, noise (typos, casing, agent prefixes, irrelevant detail), a validator for safety and gas wording, and a train/held-out split. It was later superseded by the scenario-based `data_diverse.py`, which I developed with the other tools above. The early version informed how I thought about rule coverage and held-out evaluation.
- **Dataset design discussion.** Claude helped me decide which columns belong in the training file and which are evaluation-only bookkeeping, and why a scenario or group identifier is needed to avoid leakage.
- **Frontend.** Claude drafted the single-page UI (`index.html`, with styling separated into `index.css`) and the FastAPI changes that serve it from the same process.
- **Code review.** Claude reviewed `api.py`, `classifier1.py` and `rules1.py` against the brief and flagged risks. 

## GitHub Copilot

I used GitHub Copilot to review project setup, update dependencies and README instructions, add screenshot links, and trace the UI's periodic health checks.

<!-- ## Responsibility

AI assistance supported analysis, debugging, documentation, and iteration. I remained responsible for choosing what to change, integrating the work, validating behavior with tests, and ensuring that the final submission reflects the assignment requirements. -->

## Responsibility

AI assistance supported analysis, debugging, documentation, and iteration. I remained responsible for choosing what to change, integrating the work, validating behavior with tests, and ensuring that the final submission reflects the assignment requirements.

Specifically, I did the following myself:

- Set up the repository and project structure, and decided which parts of the AI-drafted work to keep, replace or drop. For example, my scenario-based `data_diverse.py` superseded the early generator Claude wrote.
- Reviewed the generated data by hand and found its problems. Examples were the missing `scenario_id` for grouping related examples, and product–symptom mismatches such as gas smells paired with unrelated motor faults and cooling-fan complaints on microwaves. I then directed the fixes.
- Decided what goes into each data file: realistic input and label columns for training, and extra bookkeeping columns only for slicing evaluation results.
- Made scope calls, including dropping multilingual and code-mixed examples and removing the Hinglish example from the UI.
- Directed the frontend structure, requiring styling to be separated from the HTML into its own CSS file.
- Cut the API tests down to what the brief and the graded behaviour actually require.
- Integrated the generator, classifier, rule engine, API and UI into one service, and wrote the final documentation in my own words.