<h1 align="center">Turns Codebase into Coding Puzzles with AI</h1>

*This is a super simple LLM "agent" that analyzes a GitHub repository and generates hands-on coding puzzles to help you understand how the code works. The idea came to me when I was trying to study for interviews and I felt I wanted to learn concepts and techniques used in good open-source codebases. While there are tools [codebase-2-tutorial](https://code2tutorial.com/), reading a tutoprial is a **passive** form of learning. True learning is hands-on and happens when code something yourself. This wil lgenerate you coding puzzles that you can use to practice for interviews or just to get a stronger foundatoinal understanding of tools.

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
    - `--max-concepts` - Maximum number of concepts to generate puzzles for to identify (default: 10)
    - `--no-cache` - Disable LLM response caching (default: caching enabled)

The application will crawl the repository, analyze the codebase structure, generate coding puzzles in the specified language, and save the output in the specified directory (default: ./output).


## Acknowledgmens

This is based on [Pocket Flow](https://github.com/The-Pocket/PocketFlow), a 100-line LLM framework. It crawls GitHub repositories and analyzes code to extract key concepts and patterns. The AI then generates educational coding puzzles that challenge developers to implement or understand core functionality, making complex codebases accessible through hands-on practice.






