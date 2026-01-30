# LLM Prompt Compression - Experimental Spike
Status: Completed

## The Problem
Large language model (LLM) API costs scale with token usage, and many applications involve processing lengthy prompts with substantial context. This creates cost and latency challenges, particularly for applications requiring frequent LLM calls with extensive background information.

## The Hypothesis
We can compress LLM input prompts using semantic and rule-based methods while maintaining acceptable output quality and semantic similarity.

## What we aim to discover
- Can we reduce input token counts through compression without significantly degrading output quality?
- How do different compression methods (rule-based vs. semantic) compare in terms of compression ratio and output quality?
- Does changing prompt structure affect compression effectiveness?
- What is the relationship between compression rate and output quality?
- What is the relationship between compression rate and output length?

## If it works
- Reduce API costs for LLM applications by 20-60% through input token compression
- Decrease latency for LLM processing
- Enable longer context windows by reducing token overhead
- Provide a scalable solution for cost-sensitive LLM applications

## Assumptions
- Simple stopword removal can maintain semantic meaning while reducing tokens
- Transformer-based semantic compression (LLMLingua-2) can intelligently identify non-essential tokens
- Output quality can be objectively measured using established metrics (METEOR, BERTScore)
   - BERTScore being semantic sumiliarity assumes answers are correct
- Preserving system instructions and queries while compressing content may improve results (Oreo Method)

## Outcomes
Both compression methods successfully reduced input tokens with varying quality trade-offs:


### Quality Metrics (Mean Scores)
**METEOR Scores:**
- Caveman (Normal): 0.48 (±0.08)
- Caveman (Oreo): 0.54 (±0.10)
- LLMLingua-2 (Normal): 0.42 (±0.08)
- LLMLingua-2 (Oreo): 0.47 (±0.12)

**BERTScore:**
- Caveman (Normal): 0.87 (±0.03)
- Caveman (Oreo): 0.88 (±0.03)
- LLMLingua-2 (Normal): 0.86 (±0.04)
- LLMLingua-2 (Oreo): 0.87 (±0.03)

### Output Token Lengths
Compressed prompts often produce **different length outputs** than uncompressed prompts:

**Mean Output Ratios (Compressed Output / Control Output):**
- **Caveman (Normal)**: 107% of control (+7% longer responses)
- **Caveman (Oreo)**: 98% of control (-2% shorter responses)
- **LLMLingua-2 (Normal)**: 111% of control (+11% longer responses)
- **LLMLingua-2 (Oreo)**: 103% of control (+3% longer responses)

**Control Output Tokens:** Mean 80.6 tokens (±10.2)

This counterintuitive finding suggests that LLMs may compensate for compressed or less clear prompts by providing more verbose explanations. The Oreo method produces outputs closest to control length, with Caveman-Oreo even producing slightly shorter outputs while maintaining quality.


### Key Findings
1. **Oreo method consistently outperforms normal compression** across both compression methods and quality metrics
2. **Caveman compression shows higher quality retention** (METEOR +12-15%, BERTScore +1-2%) compared to LLMLingua-2
3. **LLMLingua-2 achieves greater compression** (4-5% more reduction) at the cost of quality
4. **Compression rate vs. quality trade-off**: Analysis of LLMLingua-2 at various rates (20-100% of original) shows diminishing returns beyond 40% compression
5. **Output token variation**: Compressed prompts sometimes generate longer outputs, suggesting LLMs may compensate for lost context

## Shortcomings
- Limited to 10 test prompts across diverse domains - larger sample size needed for statistical significance
- Quality metrics (METEOR, BERTScore) may not fully capture semantic correctness or task completion
- No cost-benefit analysis comparing token savings against potential quality degradation
- Testing performed on single LLM model (GPT-4) - results may vary across different models
- No evaluation of task-specific performance (e.g., did the LLM actually answer the question correctly?)
- Caveman method uses fixed stopword list - could be optimized per use case
- LLMLingua-2 compression rate (0.8) chosen somewhat arbitrarily for main tests
- No testing on extremely long prompts (>1000 tokens) where compression benefits would be most significant
- Output quality assessment doesn't account for factual accuracy, only similarity to control responses

## Running the Notebook
The main analysis is in [compressors_testing.ipynb](compressors_testing.ipynb).

### Prerequisites
- Python environment with Jupyter notebook support
- OpenAI API key with GPT-4 access
- Sufficient compute resources for LLMLingua-2 model (GPU recommended for faster inference)

### Models and Methods Under Test
1. **Caveman Compression** - Rule-based stopword removal approach
   - Simple baseline using predefined stopword set
   - Removes articles, auxiliary verbs, prepositions, and common non-essential words
   - Fast execution, no model required

2. **LLMLingua-2** - Semantic compression using transformer model
   - Uses microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
   - Token-level filtering based on importance scores
   - Configurable compression rate (0.2-1.0)

3. **Prompt Structures**
   - **Normal**: Compress entire prompt including instructions
   - **Oreo**: Preserve system instructions and query, compress only content


### Key Metrics Calculated
- **Input Compression Ratio**: test_tokens / control_tokens
- **Output Compression Ratio**: test_output_tokens / control_output_tokens
- **METEOR Score**: Metric for Evaluation of Translation with Explicit ORdering (0-1, higher is better)
- **BERTScore**: Contextual embedding-based similarity (0-1, higher is better)

## Future Work
- Evaluate task-specific performance metrics (accuracy, completeness, correctness)
- Test across multiple LLM models (Claude, Gemini, Llama) to assess generalizability
- Implement adaptive compression: adjust rate based on prompt characteristics
- Develop domain-specific stopword lists for Caveman method
- Cost-benefit analysis: quantify savings vs. quality impact
- Test on production workloads with real user prompts
- Investigate other compression options like SCOPE
