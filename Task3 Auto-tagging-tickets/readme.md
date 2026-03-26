# Task 3: Auto Tagging Support Tickets Using LLM

## Objective
Automatically classify free-text customer support tickets into predefined categories
using prompt engineering with an LLM, outputting the **top 3 most probable tags**
per ticket and comparing zero-shot vs few-shot performance.

## Dataset
- 350 support tickets (`ticket_id`, `ticket_text`, `true_tags`)
- Categories: `billing`, `technical`, `account`, `general`
- Multi-label tickets normalized to primary tag for evaluation
- 150 tickets sampled for evaluation

## Methodology
**Model:** LLaMA 3.1 8B Instant via Groq API (`llama-3.1-8b-instant`)

**Zero-Shot:** Category descriptions only, no examples — tests pure instruction-following.

**Few-Shot:** Six labeled examples (1-2 per category) embedded in the prompt as
in-context reference points, no fine-tuning required.

**Output:** Strict JSON `{"top_3_tags": ["tag1", "tag2", "tag3"]}`, temperature 0.0,
with retry logic and fallback defaults.

## Results
- Zero-Shot Top-1 / Top-3 Accuracy: 28.67% / 78.67%
- Few-Shot Top-1 / Top-3 Accuracy: 27.33% / 79.33%

Few-shot edges out zero-shot on Top-3 (79.33% vs 78.67%), meaning in-context examples
help the model surface the correct tag in its top 3 even when not ranking it first.
Correct tag appeared in top 3 for ~4 out of every 5 tickets.

## Key Observations
- ~79% Top-3 accuracy directly satisfies the task objective
- 119/150 top-1 predictions were identical across both approaches, showing the small
  model struggles to leverage few-shot examples effectively
- Larger models and fine-tuning would yield stronger zero-shot vs few-shot separation

## Skills Demonstrated
- Prompt engineering, zero-shot and few-shot learning
- Multi-label prediction and top-k ranking
- LLM API integration (Groq), JSON output parsing
- Evaluation with accuracy and confusion matrices