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

def unwrap_markdown_fence(text: str) -> str:
    """
    Strip an outer ```markdown / ```md wrapper added by the LLM in response
    to an explicit _WRAP_INSTRUCTION request.

    Only unwraps ```markdown and ```md openers — never a bare ``` opener.
    Bare ``` at the start of a response is legitimate content (e.g. a code
    block is the first thing in the reply) and must not be stripped.

    Tracks fence nesting depth so that inner code blocks (``` inside the
    wrapped content) do not prematurely terminate the unwrap.  The outer
    closer is the ``` that brings the depth back to zero.

    Returns the text unchanged if no recognised opener is found, if the
    opener is unclosed, or if the extracted content is empty.
    """
    text = text.strip()
    for open_tag in ("```markdown", "```md"):
        if not text.startswith(open_tag):
            continue
        rest = text[len(open_tag):]
        first_newline = rest.find('\n')
        if first_newline == -1:
            return text  # malformed — no newline after opener
        content_start = first_newline + 1
        lines = rest[content_start:].splitlines()
        # depth=1 because we are inside the outer opener.
        # Any line starting with ``` that is NOT a closer (has chars after)
        # opens a new inner block (depth+1).  A bare ``` line closes one
        # level (depth-1).  When depth reaches 0 we have the outer closer.
        depth = 1
        collected: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == "```":
                depth -= 1
                if depth == 0:
                    inner = "\n".join(collected).strip()
                    return inner if inner else text
                collected.append(line)
            elif stripped.startswith("```") and len(stripped) > 3:
                depth += 1
                collected.append(line)
            else:
                collected.append(line)
        return text  # unclosed
    return text


def strip_edit_response(text: str) -> str:
    """
    Extract the replacement content from an edit-mode response.

    Continue's /v1/completions (edit mode) inserts the response verbatim into
    the user's file.  LLMs frequently wrap their edit in a fenced code block,
    and may also add a heading, preamble, or trailing explanation around it.
    None of that surrounding text belongs in the file.

    Strategy:
      1. Use ``re.MULTILINE`` with a ``^``-anchored pattern to locate the
         first fenced code block opener (`` ``` `` optionally followed by a
         language tag) that appears at the true start of a line.  This is
         immune to invisible-character and mixed-line-ending artefacts
         (``\\r\\n``, stray ``\\r``, Unicode spaces) that caused the previous
         line-by-line depth-tracker to silently fail at runtime.
      2. Once the opener is found, collect everything until the *last* bare
         `` ``` `` closer that appears at the start of a line.  Using the last
         closer (rather than the first) correctly preserves nested fences,
         e.g. a ``markdown`` block that wraps an inner ``python`` block.
      3. If the opener is found but no closer follows, return the raw text
         unchanged rather than returning an empty string.
      4. If no fenced block is present at all, return the raw text unchanged
         (the response may legitimately be plain-text replacement content).

    Handles all preamble forms without special-casing them:
      - Bare response:     `` ``` ``\\ncontent\\n`` ``` ``               → content
      - With heading:      ``## Title``\\n`` ```lang ``\\ncontent\\n`` ``` `` → content
      - With preamble:     ``Here is the update:``\\n`` ``` ``\\n…       → content
      - Leaked UI label:   ``Markdown``\\n`` ```mermaid ``\\ncontent\\n`` ``` `` → content
      - Nested fences:     `` ```markdown ``\\n…`` ```python ``\\n…`` ``` ``\\n…`` ``` `` → full inner content
    """
    text = text.strip()
    if not text:
        return text

    # Locate the first opener: ``` followed by an optional language tag, then
    # a newline.  re.MULTILINE makes ^ match at the start of every line.
    opener_match = re.search(r'^```[^\n]*\n', text, re.MULTILINE)
    if not opener_match:
        return text  # no fence — plain text replacement

    content_start = opener_match.end()
    remaining = text[content_start:]

    # Find ALL bare ``` closers in the remaining text (``` optionally followed
    # by whitespace, at the start of a line).  \s* handles trailing \r or
    # non-breaking spaces that would defeat a bare string equality check.
    closer_matches = list(re.finditer(r'^```\s*$', remaining, re.MULTILINE))
    if not closer_matches:
        return text  # unclosed fence — return as-is rather than empty string

    # Use the LAST closer so that inner fence blocks are fully captured.
    last_closer = closer_matches[-1]
    inner = remaining[:last_closer.start()].strip()
    if not inner:
        return text

    # Normalise CRLF / bare-CR line endings that the DOM extractor may have
    # introduced.  Continue inserts the returned text verbatim into the user's
    # file; stray \r characters would corrupt Unix files on WSL2.
    inner = inner.replace('\r\n', '\n').replace('\r', '\n')
    return inner


def clean_response_text(raw: str) -> str:
    """
    Clean a plain-text string scraped from a browser-based LLM response.

    Steps:
      1. Two-pass HTML entity unescaping.
         Some UIs render characters inside code blocks as literal entity text
         (e.g. '&gt;' displayed on screen rather than '>').  A single pass
         of html.unescape() fixes singly-escaped entities; a second pass
         catches doubly-escaped cases like '&amp;gt;' → '&gt;' → '>'.
      2. Remove known UI junk lines (case-insensitive exact match).
      3. Collapse runs of 3+ consecutive blank lines to 2.
      4. Unwrap an outer ```markdown fence if present.
    """
    if not raw:
        return raw

    # 1. Two-pass entity unescape
    text = html.unescape(html.unescape(raw))

    # 2. Strip UI junk lines
    lines = text.splitlines()
    cleaned: List[str] = [
        line for line in lines
        if line.strip().lower() not in JUNK_LINES
    ]
    text = "\n".join(cleaned)

    # 3. Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Unwrap markdown fence
    text = unwrap_markdown_fence(text)

    return text.strip()


# ── DOM extraction (requires Playwright element) ─────────────────────────────

# JavaScript walker injected into the browser page.
# Reconstructs markdown-friendly output from the DOM structure, preserving
# code block fences and headings that inner_text() would discard.
_EXTRACT_JS = """
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
            const codeText = codeEl ? codeEl.innerText : node.innerText;
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