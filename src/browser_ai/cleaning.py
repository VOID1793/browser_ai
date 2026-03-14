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
    Strip an outer ```markdown ... ``` wrapper that the server asked the LLM
    to produce so raw markdown can be returned without browser-rendering artefacts.

    Handles ```markdown, ```md, and plain ``` openers.
    Returns the text unchanged if no matching outer fence is found.
    """
    text = text.strip()
    for open_tag in ("```markdown", "```md", "```"):
        if text.startswith(open_tag):
            inner = text[len(open_tag):]
            last = inner.rfind("```")
            if last != -1:
                return inner[:last].strip()
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
        if (pre.dataset && pre.dataset.lang) return pre.dataset.lang.toLowerCase();
        const parent = pre.parentElement;
        if (parent && parent.dataset && parent.dataset.lang)
            return parent.dataset.lang.toLowerCase();
        const codeEl = pre.querySelector('code');
        if (codeEl) {
            for (const cls of codeEl.classList)
                if (cls.startsWith('language-')) return cls.slice(9).toLowerCase();
        }
        let sib = pre.previousSibling;
        while (sib) {
            const t = sib.textContent ? sib.textContent.trim() : '';
            if (t) return t.toLowerCase();
            sib = sib.previousSibling;
        }
        if (parent) {
            const labelEl = parent.querySelector(
                '.code-block-decoration, .language-label, [class*="lang"]'
            );
            if (labelEl) return labelEl.textContent.trim().toLowerCase();
        }
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
