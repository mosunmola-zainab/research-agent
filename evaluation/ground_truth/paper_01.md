# Paper 1 Ground Truth

## Paper

Peykani et al. (2026), "A Holistic Review of Agentic AI Frameworks, Applications, and Research Trajectories."

## What I Want to Learn

I want to understand what Agentic AI actually is, what makes a system agentic, what components and architectures these systems have, how they are built and evaluated, what people are currently doing with them, what the major problems are, and where the research is going.

## Questions

### 1. WebArena performance

**Question:** What performance gap does the paper report between humans
and the best GPT-4-based agent on WebArena?

**Expected:** Humans achieve about 78% task success compared with about 14% for the best GPT-4-based agent.

Don't lose:
- 78% = humans
- 14% = GPT-4-based agent
- the numbers are approximate

### 2. OSWorld evaluation errors

**Question:** The paper says agent performance can be underestimated on benchmarks such as OSWorld. By how much, and what causes the problem?

**Expected:** Performance can be underestimated by up to 28% because of problems with the evaluation infrastructure, such as outdated website versions and deprecated CSS selectors.

Don't lose:
- up to 28%
- this is underestimation, not a 28% drop in the agent's actual ability
- the problem comes from the evaluation infrastructure

### 3. Benchmark task scale

**Question:** How many tasks do OdysseyBench and OSWorld contain, and how is OdysseyBench's task set divided?

**Expected:** OdysseyBench has 602 tasks: 300 realistic tasks and 302 difficult composite tasks. OSWorld has 369 real computer tasks.

Don't lose:
- OdysseyBench: 602 total = 300 realistic + 302 difficult composite
- OSWorld: 369

### 4. Length-Filtered Voting

**Question:** According to Figure 5, by what percentage does Length-Filtered Voting improve peak accuracy over normal voting?

**Expected:** The paper does not provide enough information to calculate a percentage. Figure 5 is a conceptual diagram and does not give precise numerical accuracy values.

Don't lose:
- do not invent or estimate a percentage
- the figure is conceptual, not quantitative

### 5. Agentic AI architecture

**Question:** What are the six core components of the Agentic AI architecture proposed in the paper, and what does each one do?

**Expected:** The six components are perception, role specification, memory, planning and reasoning, action and tool use, and online learning. Together, they cover how an agent takes in information, understands its role, keeps useful information, decides what to do, acts through tools, and improves from feedback and experience.

Don't lose:
- all six components
- they are connected parts of the proposed architecture, not six types of agents

### 6. G-Memory

**Question:** How does G-Memory organize long-term memory for multi-agent systems?

**Expected:** G-Memory uses three levels: an insight graph, a query graph, and an interaction graph. They organize knowledge at different levels, from reusable insights to task/query information and interactions between agents.

Don't lose:
- insight graph
- query graph
- interaction graph
- this is for multi-agent long-term memory

### 7. Memory management

**Question:** What five memory management policies does the paper describe for handling limited context and sensitive information?

**Expected:** Storage, summarization, deletion, re-indexing, and sensitive data purging.

Don't lose:
- all five policies
- sensitive data purging is specifically about privacy/security

