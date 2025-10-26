# codebase-to-puzzles




<p align="center">
<img width="600" height="370" alt="image" src="https://github.com/user-attachments/assets/ef6478f5-9c7b-49a9-8388-803a2b1cef97" />
</p>

This is a super simple LLM "agent" that analyzes a GitHub repository and generates hands-on coding puzzles to help you understand how the code works. The idea came to me when I was trying to study for interviews and I felt I wanted to learn concepts and techniques used in good open-source codebases. While there are tools [codebase-2-tutorial](https://code2tutorial.com/), reading a tutoprial is a **passive** form of learning. True learning is hands-on and happens when code something yourself. This wil lgenerate you coding puzzles that you can use to practice for interviews or just to get a stronger foundatoinal understanding of tools.



## 🚀 Getting Started

1. Clone this repository
   ```bash
   git clone https://github.com/mukhal/codebase-to-puzzles
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up LLM in [`utils/call_llm.py`](./utils/call_llm.py) by providing credentials. To do so, you can put the values in a `.env` file. By default, you can use the AI Studio key with this client for Gemini Pro 2.5 by setting the `GEMINI_API_KEY` environment variable. If you want to use another LLM, you can set the `LLM_PROVIDER` environment variable (e.g. `XAI`), and then set the model, url, and API key (e.g. `XAI_MODEL`, `XAI_URL`,`XAI_API_KEY`). If using Ollama, the url is `http://localhost:11434/` and the API key can be omitted.
   You can use your own models. We highly recommend the latest models with thinking capabilities (Claude 3.7 with thinking, O1). You can verify that it is correctly set up by running:
   ```bash
   python utils/call_llm.py
   ```

5. Generate coding puzzles from a codebase by running the main script:
    ```bash
    # Analyze a GitHub repository
    python main.py --repo https://github.com/username/repo --include "*.py" "*.js" --max-concepts 5

    ```

    - `--repo` or `--dir` - Specify either a GitHub repo URL or a local directory path (required, mutually exclusive)
    - `-n, --name` - Project name (optional, derived from URL/directory if omitted)
    - `-t, --token` - GitHub token (or set GITHUB_TOKEN environment variable)
    - `-o, --output` - Output directory (default: ./output)
    - `-i, --include` - Files to include (e.g., "`*.py`" "`*.js`")
    - `-e, --exclude` - Files to exclude (e.g., "`tests/*`" "`docs/*`")
    - `-s, --max-size` - Maximum file size in bytes (default: 100KB)
    - `--max-concepts` - Maximum number of concepts to generate puzzles for to identify (default: 5)
    - `--include-hints` - Whether to include solution hints in the generated puzzles
    - `--puzzle-count` - Number of puzzles to generate per concept
    - `--no-cache` - Disable LLM response caching (default: caching enabled)

The application will crawl the repository, analyze the codebase structure, generate coding puzzles in the specified language, and save the output in the specified directory (default: ./output).

## Examples
Here are some example generated puzzles! 

### Codebase: [nanogpt](https://github.com/karpathy/nanoGPT)
```python
# Implement the **scaled dot-product causal self-attention** mechanism.
# Given query, key, and value tensors, you must compute attention scores,
# apply a causal mask so that each position only attends to previous tokens,
# normalize with softmax, and produce the final weighted sum.
# *Understanding the mask and scaling is key to GPT’s context-building.*
# # The heart of GPT models is the Transformer block, especially the causal self-attention.
# Think of it as a group discussion: each token (word or character) “earns” a score when
# “talking” to every other earlier token. Only past tokens are heard (causal mask), so
# the model can’t peek into the future. This mechanism lets the model build context
# gradually and generate coherent text. Understanding it is key to modifying, extending,
# or debugging the model’s core reasoning engine.
# import torch
import math
import torch

def causal_self_attention(q, k, v):
    """
    Compute causal self-attention.
    q, k, v: tensors of shape (batch, seq_len, dim)
    returns: attention output of same shape.
    """
    # TODO: implement scaled dot-product attention with a causal mask.
    seq_len = q.shape[1]
    dim = q.shape[2]
    
    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
    mask = ~torch.tril(torch.ones(seq_len, seq_len)).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ v
    return output
    
    raise NotImplementedError

def test_causal_self_attention():
    torch.manual_seed(0)
    batch, seq_len, dim = 1, 4, 2
    # create a simple input where q=k=v are increasing numbers
    x = torch.arange(batch*seq_len*dim, dtype=torch.float).view(batch, seq_len, dim)
    out = causal_self_attention(x, x, x)
    # position 0 cannot attend to anything but itself
    assert torch.allclose(out[:,0,:], x[:,0,:], atol=1e-6), f"pos0 mismatch: {out[:,0,:]} vs {x[:,0,:]}"
    # shape must be preserved
    assert out.shape == x.shape
    print("causal_self_attention test passed")

if __name__ == "__main__":
    test_causal_self_attention()
```

```python
# Create a **sampling function** that:
# 1. Scales logits by `1/temperature`.
# 2. Applies **top-k filtering** (set other logits to -inf).
# 3. Converts to probabilities with softmax.
# 4. Samples a token index with `torch.multinomial`.
# *This is the core of GPT’s text generation step.*
# # Once trained, the model generates text token by token. You provide a prompt,
# then repeatedly sample the next token from a probability distribution shaped
# by temperature (randomness) and top-k filtering (only the k most likely tokens).
# This is like rolling a weighted die where you control how many faces are allowed
# and how fair the die is. Understanding sampling lets you tweak creativity vs.
# coherence during text generation and integrate custom decoding strategies.
# import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=None):
    """
    Sample one token index from `logits`.
    """
    # TODO: implement sampling:
    #   - scale logits: logits = logits / temperature
    #   - if top_k: find kth value, mask below it to -inf
    #   - probs = softmax(logits)
    #   - return torch.multinomial(probs, num_samples=1)
    raise NotImplementedError

def test_sample_next_token():
    torch.manual_seed(0)
    logits = torch.tensor([1.0, 5.0, 0.5])
    counts = {i:0 for i in range(3)}
    for _ in range(1000):
        idx = sample_next_token(logits, temperature=1.0, top_k=None).item()
        counts[idx] += 1
    # token 1 should be chosen most often
    assert counts[1] > counts[0] and counts[1] > counts[2]
    # with top_k=1, always pick the highest logit
    idx = sample_next_token(logits, temperature=1.0, top_k=1)
    assert idx.item() == 1
    print("sample_next_token tests passed")

if __name__ == "__main__":
    test_sample_next_token()
```


## Acknowledgmens

This is largely based on [Pocket Flow](https://github.com/The-Pocket/PocketFlow). 






