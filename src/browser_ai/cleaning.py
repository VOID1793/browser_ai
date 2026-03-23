"""
browser_ai.cleaning
~~~~~~~~~~~~~~~~~~~
Utilities for cleaning and extracting text from browser-scraped responses.

All functions in this module are pure (no browser I/O) except for
extract_clean_response_text(), which requires a live Playwright element.
"""

from __future__ import annotations

import html
import re
from typing import List

from browser_ai.config import JUNK_LINES


# ── Plain-text cleaning ───────────────────────────────────────────────────────


def strip_edit_response(text: str) -> str:
    """
    Extract the fence-INCLUSIVE replacement block from an edit-mode response.

    Continue's /v1/completions (edit mode) uses the response text to REPLACE
    the user's selection verbatim.  The user's selection includes the fence
    markers (e.g. ```mermaid ... ```).  Therefore, we must return
    the fence markers too — stripping them causes Continue to diff "delete the
    fence lines from the file."

    Strategy:
      Walk code-block openers in REVERSE (last → first).  For each opener,
      find its first subsequent closer.  Return the LAST complete code block
      INCLUDING its opener and closer fence lines.

      Last-block preference: LLMs frequently show the original before the
      replacement; the final code block is the intended answer.

      Unclosed-block fallback: if the last opener has no closer (LLM truncated
      its output), return everything from that opener to end-of-text.  This
      is better than returning the raw prose-wrapped response.

    Returns ``text`` unchanged when no fence is found (plain-text replacement).
    Never returns an empty string — if extraction yields nothing, returns raw.
    """
    text = text.strip()
    if not text:
        return text

    openers = list(re.finditer(r'^```[^\n]*\n', text, re.MULTILINE))
    if not openers:
        return text  # no fence — plain text replacement

    for opener in reversed(openers):
        remaining = text[opener.end():]
        closer_match = re.search(r'^```\s*$', remaining, re.MULTILINE)
        if not closer_match:
            continue
        inner = remaining[:closer_match.start()].strip()
        if not inner:
            continue  # empty block — skip

        # Reconstruct fence-inclusive block.
        # opener.group(0) is e.g. "```mermaid\n"  — strip trailing newline for clean join.
        fence_open = opener.group(0).rstrip('\n')
        fence_close = "```"
        result = f"{fence_open}\n{inner}\n{fence_close}"
        return result.replace('\r\n', '\n').replace('\r', '\n')

    # Unclosed fallback: return from last opener to end-of-text (LLM truncated).
    last = openers[-1]
    candidate = text[last.end():].strip()
    if candidate and not re.match(r'^```\s*$', candidate):
        fence_open = last.group(0).rstrip('\n')
        result = f"{fence_open}\n{candidate}"
        return result.replace('\r\n', '\n').replace('\r', '\n')

    return text  # absolute fallback — return raw so caller gets something


def clean_response_text(raw: str) -> str:
    """
    Clean a plain-text string scraped from a browser-based LLM response.

    Steps:
      1. Two-pass HTML entity unescaping.
         Some UIs render characters inside code blocks as literal entity text
         (e.g. '&gt;' displayed on screen rather than '>').  A single pass
         of html.unescape() fixes singly-escaped entities; a second pass
         catches doubly-escaped cases like '&amp;gt;' → '&gt;' → '>'.
      2. Remove known UI junk lines (case-insensitive exact match) — fence-aware
         so that valid code lines that happen to match a junk string are never
         stripped from inside a code block.
      3. Collapse runs of 3+ consecutive blank lines to 2.

    NOTE: This function deliberately does NOT unwrap ```markdown fences.
    Doing so unconditionally caused non-deterministic chat output: LLMs
    sporadically choose to wrap markdown-heavy responses in a ```markdown
    fence without being asked, and stripping that wrapper on every response
    made identical prompts produce different output depending on the LLM's
    mood.  edit mode (strip_edit_response) handles nested/wrapped fences via
    the last-complete-block strategy without needing a pre-pass here.
    """
    if not raw:
        return raw

    # 1. Two-pass entity unescape
    text = html.unescape(html.unescape(raw))

    # 1b. Convert literal \n (backslash-n) to actual newlines outside code fences.
    # Some DOM contexts return text with \n as two literal characters rather than
    # actual newline bytes. This converts them so the response renders correctly.
    # We track fence depth so we don't corrupt code block content.
    fence_depth = 0
    result_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('```'):
            if stripped == '```':
                # Bare ``` toggles between outside (depth=0) and inside (depth=1)
                fence_depth = 1 if fence_depth == 0 else 0
            else:
                fence_depth += 1  # language-tagged opener always increases depth
            result_lines.append(line)
            continue
        if fence_depth == 0:
            line = line.replace('\\n', '\n')
        result_lines.append(line)
    text = '\n'.join(result_lines)

    # 2. Strip UI junk lines — fence-aware so code lines are never stripped.
    # The JS extractor already suppresses button elements (the primary source
    # of junk), so this is a safety net for text nodes that slip through.
    # We track fence depth (depth > 0 = inside a fence) and skip stripping
    # while inside any fenced block to avoid corrupting code content.
    lines = text.splitlines()
    cleaned: List[str] = []
    fence_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            if stripped == '```':
                # Bare ``` toggles between outside (depth=0) and inside (depth=1)
                fence_depth = 1 if fence_depth == 0 else 0
            else:
                fence_depth += 1  # language-tagged opener always increases depth
            cleaned.append(line)
            continue
        if fence_depth == 0 and stripped.lower() in JUNK_LINES:
            continue  # UI junk outside a fence — drop it
        cleaned.append(line)
    text = "\n".join(cleaned)

    # 3. Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── DOM extraction (requires Playwright element) ─────────────────────────────

# JavaScript walker injected into the browser page.
# Reconstructs markdown-friendly output from the DOM structure, preserving
# code block fences and headings that inner_text() would discard.
_EXTRACT_JS = r"""
(element) => {
    const allPres = Array.from(element.querySelectorAll('pre'));
    const suppressedNodes = new Set();

    for (const pre of allPres) {
        let sib = pre.previousSibling;
        while (sib) {
            const sibText = sib.textContent ? sib.textContent.trim() : '';
            if (sibText) { suppressedNodes.add(sib); break; }
            sib = sib.previousSibling;
        }
        const parent = pre.parentElement;
        if (parent) {
            const labelEl = parent.querySelector(
                '.code-block-decoration, .language-label, [class*="lang"]'
            );
            if (labelEl) suppressedNodes.add(labelEl);
        }
    }

    function getLangForPre(pre) {
        // Prefer explicit data attributes set by the UI framework.
        if (pre.dataset && pre.dataset.lang) return pre.dataset.lang.toLowerCase();
        const parent = pre.parentElement;
        if (parent && parent.dataset && parent.dataset.lang)
            return parent.dataset.lang.toLowerCase();

        // CSS language class on the inner <code> element (e.g. language-python).
        const codeEl = pre.querySelector('code');
        if (codeEl) {
            for (const cls of codeEl.classList)
                if (cls.startsWith('language-')) return cls.slice(9).toLowerCase();
        }

        // Dedicated language-label element in the parent container.
        // Only accept it if the text looks like a real language tag:
        // a single token, no spaces, reasonable length.
        if (parent) {
            const labelEl = parent.querySelector(
                '.code-block-decoration, .language-label, [class*="lang"]'
            );
            if (labelEl) {
                const t = labelEl.textContent.trim();
                if (t && !t.includes(' ') && t.length <= 20)
                    return t.toLowerCase();
            }
        }

        // ChatGPT code block header: the language name appears as the text
        // content of the first child div inside the <pre> that contains the
        // block label (e.g. "Mermaid", "Python").  It is a single short word
        // with no spaces.  Only accept it when a .cm-content is also present
        // (confirms this is the ChatGPT CodeMirror UI, not generic content).
        if (pre.querySelector('.cm-content')) {
            // Walk the first-level divs looking for a short single-word label.
            const headerCandidates = pre.querySelectorAll(
                'div > div > svg ~ div, div[class*="font-medium"]'
            );
            for (const el of headerCandidates) {
                const t = el.textContent.trim();
                if (t && !t.includes('\n') && t.split(' ').length <= 2 && t.length <= 20) {
                    // Normalise: "Mermaid" → "mermaid", "Python" → "python", etc.
                    return t.toLowerCase().replace(/\s+/g, '');
                }
            }
        }

        // Do NOT fall back to previous-sibling text content — UI labels like
        // "Code snippet" or "Copy" will be misidentified as language tags.
        return '';
    }

    function walk(node) {
        if (suppressedNodes.has(node)) return '';
        if (node.nodeType === Node.TEXT_NODE) return node.textContent;
        if (node.nodeType !== Node.ELEMENT_NODE) return '';

        const tag = node.tagName.toLowerCase();
        const junkTags = ['button', 'mat-icon', 'svg', 'tool-use'];
        if (junkTags.includes(tag)) return '';

        if (tag === 'pre') {
            const codeEl = node.querySelector('code');
            // innerText respects CSS visibility; some UIs (e.g. ChatGPT mermaid
            // diagrams) hide the source <code> element after rendering it as an SVG
            // or canvas.  innerText returns '' for hidden elements.  Fall back to
            // textContent (visibility-agnostic) to recover the raw source.
            let codeText = '';
            if (codeEl) {
                codeText = codeEl.innerText;
                if (!codeText) codeText = codeEl.textContent || '';
            }
            if (!codeText) {
                // ChatGPT Code view renders source in a CodeMirror editor
                // (.cm-content) using <span>line</span><br> pairs instead of a
                // <code> element.  innerText on the outer <pre> only returns the
                // header label text ("Mermaid") — the CM editor content is not
                // exposed that way.  Walk childNodes explicitly: span -> text, br -> \n.
                const cmContent = node.querySelector('.cm-content');
                if (cmContent) {
                    let lines = '';
                    for (const child of cmContent.childNodes) {
                        if (child.nodeType === Node.TEXT_NODE) {
                            lines += child.textContent;
                        } else if (child.nodeType === Node.ELEMENT_NODE) {
                            const t = child.tagName.toLowerCase();
                            if (t === 'br') {
                                lines += '\n';
                            } else {
                                // span or other inline — recurse into textContent
                                lines += child.textContent;
                            }
                        }
                    }
                    codeText = lines;
                }
            }
            if (!codeText) {
                codeText = node.innerText || node.textContent || '';
            }
            const lang = getLangForPre(node);
            return '\\n```' + lang + '\\n' + codeText.trimEnd() + '\\n```\\n';
        }
        if (tag === 'code') return '`' + node.innerText + '`';
        if (/^h[1-6]$/.test(tag)) {
            const level = parseInt(tag[1]);
            return '\\n' + '#'.repeat(level) + ' ' + node.innerText.trim() + '\\n';
        }

        const blockTags = ['p', 'div', 'li', 'tr', 'blockquote', 'br'];
        const isBlock = blockTags.includes(tag);
        let result = '';
        for (const child of node.childNodes) result += walk(child);

        if (tag === 'br') return '\\n';
        if (tag === 'li') return '\\n- ' + result.trim();
        if (tag === 'tr') return result + '\\n';
        if (isBlock && result.trim()) return '\\n' + result.trim() + '\\n';
        return result;
    }

    return walk(element);
}
"""


async def extract_clean_response_text(element) -> str:
    """
    Extract clean markdown-friendly text from a Playwright DOM element.

    Strategy (in order of preference):
      1. Run _EXTRACT_JS in the browser to reconstruct code fences and
         structural elements from the DOM.
      2. Fall back to inner_text() on a tighter child container.
      3. Last resort: inner_text() on the element itself.

    All paths run through clean_response_text() for entity unescaping,
    junk-line removal, and markdown-fence unwrapping.
    """
    # 1. JS structural extraction
    try:
        raw = await element.evaluate(_EXTRACT_JS)
        if raw and raw.strip():
            return clean_response_text(raw)
    except Exception as exc:
        print(f"[browser] JS extraction failed: {exc}", flush=True)

    # 2. Tighter child container
    for child_sel in [
        ".response-content",
        ".markdown",
        "model-response-text",
        ".model-response-text__content",
    ]:
        try:
            child = element.locator(child_sel).first
            if await child.count() > 0:
                raw = (await child.inner_text()).strip()
                if raw:
                    return clean_response_text(raw)
        except Exception:
            pass

    # 3. Full element fallback
    try:
        raw = (await element.inner_text()).strip()
        return clean_response_text(raw)
    except Exception:
        return ""