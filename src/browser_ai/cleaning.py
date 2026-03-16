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
    Extract the replacement content from an edit-mode response.

    Continue's /v1/completions (edit mode) inserts the response verbatim into
    the user's file.  LLMs frequently wrap their edit in a fenced code block,
    and may also add a heading, preamble, or trailing explanation around it.
    None of that surrounding text belongs in the file.

    Strategy:
      1. Locate ALL fenced code block openers (`` ``` `` optionally followed by
         a language tag, anchored to the start of a line via ``re.MULTILINE``).
      2. Walk the openers in REVERSE order (last to first).  For each opener,
         find the FIRST bare `` ``` `` closer that follows it.
         Return the content of the LAST complete code block found this way.

    Rationale for last-block preference:
      LLMs often begin their response by quoting the *original* code or
      providing an intermediate attempt before giving the final result.  The
      last complete code block in the response is therefore the most likely
      to contain the correct replacement.  Using the first-opener → last-closer
      span (the previous strategy) incorrectly captured all intermediate code
      and prose in between.

    Edge-case handling:
      - If no opener exists → return raw text (may be plain-text replacement).
      - If the last opener has no closer (truncated response) → try the
        previous opener, continuing until a closed block is found.
      - If all openers are unclosed → return raw text unchanged rather than
        an empty string.
      - Nested fences (e.g. ``markdown`` wrapping ``python``): the outer
        ``markdown`` fence is already unwrapped by ``clean_response_text``
        before this function is called, so only the inner fence remains.

    Handles all preamble forms without special-casing them:
      - Bare response:      `` ``` ``\\ncontent\\n`` ``` ``                → content
      - With heading:       ``## Title``\\n`` ```lang ``\\ncontent\\n`` ``` `` → content
      - With preamble:      ``Here is the update:``\\n`` ``` ``\\n…         → content
      - Multi-block (old+new code shown): only the LAST block is returned.
    """
    text = text.strip()
    if not text:
        return text

    # All line-anchored fence openers.  re.MULTILINE makes ^ match at the
    # start of every line, not just the start of the string.
    openers = list(re.finditer(r'^```[^\n]*\n', text, re.MULTILINE))
    if not openers:
        return text  # no fence present — plain text replacement, return as-is

    # Walk backwards: prefer the last code block (the LLM's "final answer").
    for opener in reversed(openers):
        remaining = text[opener.end():]
        # First bare ``` closer after this opener.  \s* handles trailing \r
        # or non-breaking spaces that would defeat a bare equality check.
        closer_match = re.search(r'^```\s*$', remaining, re.MULTILINE)
        if not closer_match:
            continue  # opener has no closer — try the previous opener
        inner = remaining[:closer_match.start()].strip()
        if not inner:
            continue  # empty block — skip and try the previous opener
        # Normalise CRLF / bare-CR introduced by the DOM extractor.
        # Continue inserts returned text verbatim; stray \r corrupts files.
        inner = inner.replace('\r\n', '\n').replace('\r', '\n')
        return inner

    # Every opener was unclosed — return the raw text rather than empty string.
    return text


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
                # Bare ``` is either a closer (depth > 0) or a bare opener (depth == 0)
                fence_depth = max(0, fence_depth - 1) if fence_depth > 0 else fence_depth + 1
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