# Research Agent Architecture

## 1. What I am building

I want an agent that helps me actually understand research papers.

The main job is to de-jargon a paper and explain it in a way a layman
can follow: who did the work, what they did, why they did it, how it
works, what they found, and why it matters.

I still want the technical depth. The agent should help me understand
the methodology, results, major claims, evidence, limitations, and open
questions instead of giving me a dressed-up abstract.

So there are really two levels of explanation:

1. A plain-language explanation for someone without specialist
   knowledge.
2. A deeper technical breakdown for when I want to interrogate the
   paper properly.

The finished analysis goes to both Notion and Obsidian. Notion is where
I submit papers and read the results. Obsidian gives me a local Markdown
copy of my research library and lets me build links between papers over
time.

## 2. Inputs

I want to eventually be able to give the agent:

- a local PDF
- an arXiv URL
- a direct PDF URL
- a DOI or publication URL
- a paper title that it can resolve through scholarly search

For development, starting with PDFs and common paper links is enough.
The architecture should not make the other input types difficult to add
later.

## 3. What I want back

Each processed paper should produce a research note with:

- paper title, authors, year, venue, DOI/arXiv ID where available
- TL;DR
- Who
- What
- Why
- How
- methodology
- key results and important numbers
- jargon decoder
- what it means for a layman
- major claims and the evidence behind them
- where important conclusions came from
- limitations the authors acknowledge
- limitations the agent identifies
- technical implications
- research implications
- practical implications
- open questions
- connections to papers already in my library
- important references or useful papers to read next, when there is a
  real reason to recommend them

The agent should simplify the language without changing the technical
meaning.

## 4. What the agent needs to do

### Get and read the paper

The agent needs to retrieve the paper, collect basic metadata, extract
its text, and preserve enough structure to know where information came
from.

It should be able to go back to a particular section or passage when it
needs evidence. I do not want the model relying on one giant prompt
containing the whole paper every time it has a question.

### Explain the paper

The explanation should answer the basic questions first:

- Who did the research and what or who was studied?
- What problem were they trying to solve?
- Why does the problem matter?
- What did they actually do?
- How does the method work?
- What did they find?
- Why should anyone care?

Technical terms should be explained when they matter rather than simply
replaced with vague language.

### Check claims against evidence

For important claims, I want something closer to:

**Claim:** Their method reduces inference latency.

**Evidence:** 18% reduction compared with baseline X on benchmark Y.

**Support:** Supported under the conditions they tested.

**Caveat:** Only evaluated on...

The agent should keep these different sources of information separate:

- `AUTHOR-STATED`
- `DIRECTLY-EVIDENCED`
- `AGENT-INFERENCE`
- `EXTERNAL-CONTEXT`

If the agent is making an inference, it should say so.

### Look at limitations

I want two separate categories:

1. limitations the authors themselves acknowledge
2. limitations the agent notices from the method, assumptions,
   experiments, evidence, or scope

The second category cannot just be the model inventing criticism. It
needs a reason grounded in the paper.

### Research outside the paper when needed

The agent can use outside sources when the paper is not enough. Examples
include understanding an unfamiliar concept, checking a benchmark,
looking at important prior work, verifying metadata, or following a
citation that is important to understanding the paper.

It should not search the web just because it has a search tool.

### Connect papers

As the library grows, the agent should search papers I have already
processed and point out useful connections such as:

- shared concepts
- similar methods
- extensions of earlier work
- disagreements or contradictory findings
- different approaches to the same problem

I care more about meaningful connections than keyword similarity.

## 5. Tools

These are the capabilities I expect the agent to have. They do not have
to become one Python file per tool.

### Paper tools

- `fetch_paper`: retrieve a paper from a supported source
- `metadata_lookup`: get title, authors, year, venue, DOI, arXiv ID,
  etc.
- `parse_paper`: extract text and useful document structure
- `search_paper`: retrieve passages relevant to a question or claim
- `inspect_section`: pull a larger section when a few retrieved chunks
  are not enough

### Research tools

- `scholarly_search`: find papers, metadata, prior work, or cited
  research
- `web_search`: get outside context when scholarly sources or the
  paper are not enough
- `inspect_reference`: look into an important cited work

### Research library tools

- `search_research_library`: search papers and notes I have already
  processed
- `get_research_note`: retrieve a previous note when a proper
  comparison needs more context

### Notion tools

- `find_notion_paper`
- `create_notion_page`
- `update_notion_page`

### Obsidian tools

- `find_obsidian_note`
- `write_obsidian_note`
- `update_obsidian_note`
- `link_related_notes`

A tool call should have a reason. If the evidence is weak, the agent
should retrieve more rather than fill the gap with a confident guess.
Tool failures also need to be visible to the workflow.

## 6. RAG and research memory

There are two retrieval problems here.

The first is the paper I am reading right now. After parsing it, the
paper can be split into chunks and indexed with metadata such as paper
ID, section, page, and chunk ID. `search_paper` can then retrieve the
parts needed for a particular question.

The second is my growing research library. Processed papers and notes
should also be searchable so the agent can find related ideas, methods,
findings, limitations, datasets, and research questions from things I
have already read.

These are related but not the same job.

### Notion, Obsidian and the retrieval index

They each have a different role:

- **Notion:** where I submit papers, add instructions, see status, and
  read the full analysis
- **Obsidian:** my local Markdown research vault and linked copy of
  the research library
- **Retrieval index:** machine-searchable memory used by the agent

A processed paper may therefore exist as:

```text
original paper
    |
    +--> full Notion page
    +--> full Obsidian note
    +--> searchable representation in the retrieval index
```

They should all be tied to the same paper identifier.

The embedding model and vector store are still TBD. I want to test them
rather than choose one because it is popular. Persistent local storage
is preferred.

## 7. How I interact with it

Notion is the main interface.

I want a research database where I can add a paper, optionally tell the
agent what I care about, and see its status.

Useful fields include:

- paper title
- URL, DOI, or other identifier
- status
- optional instructions
- topics
- date added
- date processed
- error information if something fails

Status can start simple:

- `To Process`
- `Processing`
- `Complete`
- `Failed`

The optional instructions are important. I might say:

> Focus on the memory architecture.

or:

> Explain Figure 4 carefully. I do not understand it.

or:

> Compare this with the MemGPT paper in my library.

Those instructions add to the normal paper analysis rather than
replacing it.

When a run finishes successfully, the full analysis goes to Notion and
Obsidian, and the research index is updated.

During development I also want to be able to run the agent directly from
Python or the command line so I can test pieces without involving Notion
every time.

How Notion triggers a run is still TBD. Polling, a scheduled check, a
manual trigger, and an event/webhook are all possible. I want to choose
that during implementation rather than pretend it is already settled.

## 8. LangGraph workflow

LangGraph will orchestrate the workflow.

I do not want every step to be "agentic." If code can do something
predictably, code should do it. The model should make decisions where
judgement is actually useful, such as deciding that evidence is missing,
outside context is needed, or another paper is genuinely relevant.

The rough flow is:

```text
  Paper request + optional instructions
                    |
                    v
           Identify the paper
                    |
                    v
       Check if it already exists
                    |
                    v
        Acquire paper + metadata
                    |
                    v
          Save original locally
                    |
                    v
               Parse paper
                    |
                    v
      Build current-paper retrieval
                    |
                    v
          Initial understanding
                    |
                    v
            Analyze the paper
                    |
                    v
          Is evidence missing?
                    |
            +-------+-------+
           yes              no
            |               |
            v               |
      retrieve more         |
            |               |
            +-------+-------+
                    |
                    v
          Need outside context?
                    |
            +-------+-------+
           yes              no
            |               |
            v               |
    external research       |
            |               |
            +-------+-------+
                    |
                    v
         Search research library
                    |
                    v
         Find useful connections
                    |
                    v
             Verify analysis
                    |
            +-------+-------+
        gap found         good
            |               |
            v               |
     retrieve again         |
                    +-------+
                    |
                    v
           Final research note
                    |
            +-------+-------+
            v               v
         Notion         Obsidian
            |               |
            +-------+-------+
                    |
                    v
              update index
                    |
                    v
                Complete
```

The retrieval loop needs a stopping condition. The agent should search
again because it has a specific unresolved question, not because more
information exists somewhere on the internet.

Before publishing the note, I want a verification pass that checks the
important things: major claims have evidence, important numbers can be
traced back to the paper, inference is labelled, outside context is
labelled, limitations are separated properly, the layman explanation is
faithful to the paper, and my paper-specific instructions were actually
addressed.

## 9. State and structured data

LangGraph needs a state object that carries information through the run.

I expect it to include things like:

- paper ID and metadata
- original input and my instructions
- local paper path
- parsed paper information
- claims and evidence
- methodology and results
- jargon explanations
- layman explanation
- limitations
- implications and open questions
- outside research
- connections to previous papers
- provenance
- verification results
- errors
- final research note

That list is not a schema yet. I would rather create fields as the
implementation needs them than design a giant state object in advance.

Pydantic makes sense for important structured outputs from the model.
Likely models include `PaperMetadata`, `Claim`, `Evidence`,
`Limitation`, `ResearchConnection`, and the final `ResearchNote`.

Every paper also needs a stable internal ID so its PDF, Notion page,
Obsidian note, vector records, and processing history do not drift
apart. DOI or arXiv ID can help when available, but not every paper will
have one.

## 10. Model layer

The model should be replaceable.

For now, I want to test open-weight models through OpenRouter. Later I
may switch providers, rent a GPU, or host a model myself. That should
not require rebuilding the LangGraph workflow.

### Choosing the model

I do not want to pick the model from vibes or a leaderboard. We will run
a small model bake-off using actual tasks from this project.

Things I care about include:

- understanding the paper correctly
- extracting claims and evidence
- keeping numbers accurate
- identifying sensible limitations
- de-jargoning without mangling the meaning
- producing a genuinely useful layman explanation
- structured output
- tool use
- following my instructions
- not inventing unsupported conclusions
- latency and cost

### What the model should and should not do

The model is useful for language understanding and judgement: explaining
methods, reasoning about evidence, identifying important claims,
de-jargoning, finding possible limitations, and deciding when more
context is needed.

Ordinary code should handle things like downloading files, checking
whether a file exists, storing files, schema validation, and updating
statuses.

Important model outputs should use structured schemas where that helps
us catch bad output.

Prompts also need to keep paper evidence, outside information, library
information, and the model's own inference distinct.

## 11. Evaluation and tracing

I want an evaluation harness early because we are testing models and
will probably change prompts, retrieval, chunking, parsers, and
embeddings as we build.

The detailed harness can live in `docs/EVALUATION.md`.

At minimum, I want to test:

- factual accuracy
- claims against evidence
- important numbers
- hallucinations
- provenance
- de-jargoning
- layman comprehension
- limitation quality
- structured output
- tool use
- instruction following
- latency and cost

For a small benchmark set of papers, I can manually verify important
facts such as the research question, method, datasets, major results,
numbers, claims, and author-stated limitations. That gives us something
better than asking one model to grade another model and calling it
truth.

Some things will still need human judgement. "Would a layman actually
understand this?" is one of them.

For tracing, LangSmith can help me inspect LangGraph runs, model calls,
tool calls, retrieval, latency, and failures. I mainly want it so that
when the agent does something weird, I can see what happened instead of
staring at the final answer and guessing.

Evaluation tells me whether the system is good. Tracing helps me
understand what it did.

## 12. Reliability, privacy and current technology choices

### Failure handling

This workflow touches files, model APIs, paper sources, Notion,
Obsidian, and the retrieval index, so failures are normal.

I want sensible handling for things like failed downloads, bad PDFs,
missing metadata, rate limits, malformed model output, weak retrieval,
and failed writes.

Temporary failures can be retried a limited number of times.
Deterministic failures should not be retried forever.

If one destination fails after the analysis is complete, successful work
should not disappear. For example, if Obsidian is written successfully
and Notion fails, keep the analysis and record that Notion still needs
to be synced.

I also want to avoid paying to repeat expensive steps after a
recoverable failure. How we persist LangGraph checkpoints is still TBD.

### Privacy

Secrets stay in environment variables and out of Git, logs, and research
notes.

The original papers and Obsidian vault are local by default.

For now, using OpenRouter means paper content sent to the model leaves
my laptop. Notion is also a cloud service. So the first version is not a
fully local/private system.

Longer term, I want the option to run:

```text
local paper
    |
local parsing
    |
local retrieval
    |
self-hosted model
    |
local Obsidian vault
```

without changing the whole application.

For a paper I do not want in Notion, the system should eventually allow
local-only storage.

### Current stack

| Part                   | Current choice                                |
| ---------------------- | --------------------------------------------- |
| Workflow orchestration | LangGraph                                     |
| LLM utilities          | LangChain where it actually helps             |
| Model provider         | OpenRouter                                    |
| Model                  | TBD after model bake-off                      |
| Structured validation  | Pydantic                                      |
| Tracing                | LangSmith                                     |
| Paper parser           | TBD after testing                             |
| Scholarly search       | TBD after testing                             |
| Embedding model        | TBD after testing                             |
| Vector store           | TBD after testing, persistent local preferred |
| Main interface         | Notion                                        |
| Local research vault   | Obsidian                                      |
| Original paper storage | Local filesystem                              |

This is the current architecture, not a promise that every choice is
permanent. If testing shows that a component is a bad fit, I want to be
able to replace it without redesigning the entire agent.
