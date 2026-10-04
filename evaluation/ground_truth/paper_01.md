# Paper 1 Ground Truth

## Paper

Peykani et al. (2026), "A Holistic Review of Agentic AI Frameworks, Applications, and Research Trajectories."

## What I Want to Learn

I want to understand what Agentic AI actually is, what makes a system agentic, what components and architectures these systems have, how they are built and evaluated, what people are currently doing with them, what the major problems are, and where the research is going.

## Questions

### 1. WebArena performance

**Question:** What performance gap does the paper report between humans and the best GPT-4-based agent on WebArena?

**Expected:** Humans achieve about 78% task success compared with about 14% for the best GPT-4-based agent. The numbers are approximate.

### 2. OSWorld evaluation errors

**Question:** The paper says agent performance can be underestimated on benchmarks such as OSWorld. By how much, and what causes the problem?

**Expected:** Performance can be underestimated by up to 28% because of problems with the evaluation infrastructure, such as outdated website versions and deprecated CSS selectors. This is an evaluation error, not a 28% reduction in the agent's actual ability.

### 3. Benchmark task scale

**Question:** How many tasks do OdysseyBench and OSWorld contain, and how is OdysseyBench's task set divided?

**Expected:** OdysseyBench has 602 tasks: 300 realistic tasks and 302 difficult composite tasks. OSWorld has 369 real computer tasks.

### 4. Length-Filtered Voting

**Question:** According to Figure 5, by what percentage does Length-Filtered Voting improve peak accuracy over normal voting?

**Expected:** The paper does not provide enough information to calculate a percentage. Figure 5 is a conceptual diagram and does not give precise numerical accuracy values. A percentage should not be estimated.

### 5. Agentic AI architecture

**Question:** What are the six core components in the Agentic AI architecture presented in the paper, and what does each one do?

**Expected:** The six components are perception, role specification, memory, planning and reasoning, action and tool use, and online learning. Together, they cover how an agent takes in information, understands its role, keeps useful information, decides what to do, acts through tools, and improves from feedback and experience.


### 6. G-Memory

**Question:** How does G-Memory organize long-term memory for multi-agent systems?

**Expected:** G-Memory uses three levels: an insight graph, a query graph, and an interaction graph to organize long-term interactions.

### 7. Memory management

**Question:** What five memory management policies does the paper describe for handling limited context and sensitive information?

**Expected:** Storage, summarization, deletion, re-indexing, and sensitive data purging.


### 8. Agentic RAG

**Question:** Explain the difference between traditional RAG and Agentic RAG to someone who does not know what either means.

**Expected:** Traditional RAG retrieves information and gives it to the model to help answer a question. In Agentic RAG, retrieval is more dynamic. The agent can decide what information it needs, retrieve it, assess what it found, and search again if needed.


### 9. Evaluating Agentic AI

**Question:** Why does the paper argue that checking whether an agent got the final answer right is not enough to evaluate it?

**Expected:** Because the process also matters. Evaluation should look at things such as tool use, memory and context management, planning, and safe interaction, not just whether the final answer was correct. The agent's behaviour should also be observable and logged.


### 10. Adding more agents

**Question:** According to the paper, does adding more agents necessarily make an Agentic AI system perform better? Explain why or why not.

**Expected:** No. Adding more agents can increase complexity, reduce efficiency, or lead to conflicting behaviour. Communication and coordination between agents also become harder as the system grows.

### 11. Research trajectory

**Question:** What does the paper see as the main priorities for Agentic AI in the short, medium, and long term?

**Expected:** Short term (1–2 years) focuses on foundations such as memory, evaluation, coordination, and security. Medium term (3–5 years) moves toward better reasoning with smaller models, online learning, safer multi-agent systems, and auditing. Long term (more than 5 years) includes autonomous scientific systems, value alignment, accountability, and sustainability.

The autonomous scientific systems are a future research direction, not something the paper claims is already mature.

### 12. Evidence behind the WebArena result

**Question:** Did the authors of this paper run the WebArena experiment that produced the 78% human and 14% GPT-4-based agent results? Where do those numbers come from?

**Expected:** No. This is a review paper, and the authors are reporting results from earlier WebArena research. They cite that work for the roughly 78% human performance and 14% performance of the best GPT-4-based agent.