# Model Evaluation

## Goal

I need to choose a model for the Research Agent based on how well it handles research papers, extracts facts, and breaks down information without losing meaning, exaggerating claims, or manufacturing information.

The model should help me understand difficult research without changing what the paper actually says. I care about accuracy and the quality of the explanation.

## What I Care About Most

The biggest issue is trust. Some wording can be paraphrased, but certain facts cannot simply change during explanation. For example, numbers, equations, procedures, experimental conditions, datasets, and reported results cannot simply be changed.

There are four failures I am evaluating for:

- **Fabrication or mutation:** The model invents information or changes a fact
  from the paper. I want to distinguish between something that contradicts the
  paper and something the paper never said at all.

- **Failure to abstain:** The model answers a question the paper does not
  provide enough information to answer. If I explicitly ask it to infer or
  estimate something, that should be clearly separated from what the authors
  actually reported.

- **Claim distortion:** The model makes a claim stronger than the paper does.
  A hypothesis should not become a result, correlation should not become
  causation, and a suggestion or possible future direction should not become a
  demonstrated finding.

- **Loss of important context:** The model gives a fact correctly but removes
  a condition that changes how it should be understood. This could be the
  dataset, population, baseline, experimental condition, metric, or another
  important qualifier.

I also want to know whether a model can explain jargon and methodology in plain language, connect information across a paper, preserve numerical details, identify limitations, and actually help me understand what I am reading  without losing research quality.

## How I Will Test It

Each model gets the same paper and the same questions.

The questions will cover:

- facts stated directly in the paper
- explanation of difficult concepts or methods
- questions that require connecting information from different sections
- questions whose answers are deliberately not in the paper
- interpretation of findings, uncertainty, limitations, and future work

I will create the expected answers before looking at the model responses. Where necessary, I will also record where the evidence appears in the paper, details that should not change, and acceptable variations.

I also want each model to analyse the full paper. It would help me see which model is better at actually explaining a paper and not just finding facts.

## Running the Comparison

The models will receive the same papers and prompts and be called explicitly. So, if a request fails or a model is rate-limited, I want to know instead of another model silently taking its place.

For each run, I will keep the raw response, requested model, returned model,
HTTP status, and latency. Availability also matters when choosing a model.

## First Test

I will start with three different papers:

1. an accessible Agentic AI paper
2. a technically dense AI or ML systems paper
3. a technical paper outside AI, probably energy or materials engineering

Each paper will answer a set of questions and do a full paper analysis. I will run the complete test with two models first. That gives me a chance to find problems with the evaluation before running it across all the candidate models.