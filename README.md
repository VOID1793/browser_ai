# 🌐 Browser_Ai V0.6

**Browser_Ai is an asynchronous, Playwright-backed automation bridge that transforms public AI ChatBot web interfaces into a local OpenAI-compatible API server.**

**Use public LLM reasoning and context windows within tools like Continue, Cursor, or custom LLM scripts without the need for a public-facing API key.**

---
>**DISCLAIMER: All tools in use for Browser_Ai are publicly available, openly sourced, tools are are beholden to their respective license agreements. All LLM interfaces accessed by Playwright are to be used in accordance with their respective Terms of Service, and with respect to relevant regulations. Use Browser_Ai at your own discretion.**
---

## Current Support

* Continue VSCode AI-powered extension features:
    * File Creation
    * File Editing
    * File Context Chatting
    * Sidecar Chat
* Free and Ephemeral (Non-Signed-In LLM Sessions) with:
    * Google Gemini Flash

## Coming Soon

* Generalized Agent Framework
* Open-WebUI Support
* Multi-port serving
* Dockerized Deployment
* Free and Ephemeral (Non-Signed-In LLM Sessions) with:
    * ChatGPT
    * Perplexity

## ✨ Key Features

- **OpenAI-Compatible API**: Implements `/v1/chat/completions` and `/v1/models` for seamless drop-in integration.
- **Smart Context Reordering**: Automatically moves large code blocks to the end of user prompts to prioritize instruction clarity.
- **High-Speed Injection**: Uses clipboard-level injection for large payloads, bypassing slow character-by-character typing.
- **Session Management**: Intelligent handling of browser lifecycles, idle timeouts, and concurrent request locking.
- **Markdown Preservation**: A custom DOM walker reconstructs structured Markdown (code fences, headings, etc.) directly from the browser page's UI.
- **Flexible Rendering**: Support for both Headless (silent) and Visible (interactive) browser modes.

## Example Architecture for Basic Chat

```plaintext
       [ LLM Client ]           [ GeminiBackend ]             [ Playwright/Browser ]         [ Gemini Web UI ]
      (e.g., Continue)      (gemini.py Selectors)          (Targeting CSS DOM)         (gemini.google.com)
             |                        |                             |                           |
             |--- 1. POST Prompt ---->|                             |                           |
             |                        |-- 2. Check CONSENT_SELECTORS|                           |
             |                        |   ("Accept all" / "I agree")|                           |
             |                        |---------------------------->|----- 3. Clear Dialogs --->|
             |                        |                             |                           |
             |                        |-- 4. Find INPUT_SELECTORS --|                           |
             |                        |   (rich-textarea[content])  |                           |
             |                        |---------------------------->|---- 5. Focus & Type ----->|
             |                        |                             |                           |
             |                        |-- 6. Find SEND_SELECTORS ---|                           |
             |                        |   (button[aria-label=Send]) |                           |
             |                        |---------------------------->|---- 7. Click Send ------->|
             |                        |                             |                           |
             |                        |                             | <--- 8. Rendering Mat-Spinner
             |                        |-- 9. Poll GENERATING_SEL. --|                           |
             |                        |   (Wait for spinner to hide)|                           |
             |                        |                             | <--- 10. <model-response> |
             |                        |-- 11. Extract RESPONSE_SEL -|          populated        |
             |                        |    (Scrape text from DOM)   |                           |
             |<--- 12. Final Text ----|                             |                           |
```

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
browser-ai serve --backend gemini (--visible optional to have interactive browser open)
```

### Minimal Built-In Web UI

A lightweight local testing UI is available at `http://localhost:8000/`, reusing the existing OpenAI-compatible endpoints (`GET /v1/models` and `POST /v1/chat/completions`). Start the server as usual with:

```bash
browser-ai serve --backend gemini
```

### 2. Interactive CLI

You can also chat directly with the backend in your terminal:

```Bash
browser-ai chat --backend gemini "<your prompt here>"
```

### 3. Integration

Point your favorite LLM client (e.g. Continue) to:

- Base URL: `http://localhost:8000/v1`
- Model: `gemini` [Currently Implemented]

## 🛠 Project Structure

- `browser_ai/src/backends/`: Contains the browser automation logic for different LLMs (e.g. Gemini)
- `browser_ai/src/browser_ai/server.py`: FastAPI implementation of the OpenAI-compatible API.
- `browser_ai/src/browser_ai/cli.py`: Command-line interface logic.
- `browser_ai/src/pyproject.toml`: Project dependencies and entry points.
## 📜 License

### MIT