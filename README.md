# 🌐 Browser_Ai V0.5

**Browser_Ai is an asynchronous, Playwright-backed automation bridge that transforms public AI ChatBot web interfaces into a local OpenAI-compatible API server.**

**Use public LLM reasoning and context windows within tools like Continue, Cursor, or custom LLM scripts without the need for a public-facing API key.**

---
>**DISCLAIMER: All tools in use for Browser_Ai are publicly available, openly sourced, tools are are beholden to their respective license agreements. All LLM interfaces accessed by Playwright are to be used in accordance with their respective Terms of Service, and with respect to relevant regulations. Use Browser_Ai at your own discretion.**
---

## Current Support

* Continue VSCode AI-powered extension for agentic work and side-car chatting
* Google Gemini Flash is the currently supported browser LLM backend

## ✨ Key Features

- **OpenAI-Compatible API**: Implements `/v1/chat/completions` and `/v1/models` for seamless drop-in integration.
- **Smart Context Reordering**: Automatically moves large code blocks to the end of user prompts to prioritize instruction clarity.
- **High-Speed Injection**: Uses clipboard-level injection for large payloads, bypassing slow character-by-character typing.
- **Session Management**: Intelligent handling of browser lifecycles, idle timeouts, and concurrent request locking.
- **Markdown Preservation**: A custom DOM walker reconstructs structured Markdown (code fences, headings, etc.) directly from the browser page's UI.
- **Flexible Rendering**: Support for both Headless (silent) and Visible (interactive) browser modes.

## 🛠 Prerequisites

- Python 3.8+
- Playwright Browsers: Specifically Chromium.

## 📦 Installation

### Clone the Repository
```bash
git clone https://github.com/VOID1793/browser_ai.git
cd browser_ai
```

### Install Dependencies

```Bash
pip3 install -e ./browser_ai/src
playwright install chromium
```

## 🚀 Usage

### 1. Start the API Server

Run the server to provide an OpenAI-compatible endpoint at `http://localhost:8000`.

```Bash
browser-ai serve --backend gemini --visible
```

### 2. Interactive CLI

You can also chat directly with the backend in your terminal:

```Bash
browser-ai chat --backend gemini "<your prompt here>"
```

### 3. Integration

Point your favorite LLM client (Continue) to:

- Base URL: `http://localhost:8000/v1`
- Model: `gemini` [Currently Implemented]

## 🛠 Project Structure

- `browser_ai/src/backends/`: Contains the browser automation logic for different LLMs (e.g. Gemini)
- `browser_ai/src/tunables/server.py`: FastAPI implementation of the OpenAI-compatible API.
- `browser_ai/src/tunables/cli.py`: Command-line interface logic.
- `browser_ai/src/pyproject.toml`: Project dependencies and entry points.
## 📜 License

### MIT