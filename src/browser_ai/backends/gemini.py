"""
browser_ai.backends.gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend for Google Gemini (https://gemini.google.com/app).

This is the reference implementation.  All shared machinery lives in
BaseBrowserBackend.  This class only declares the selectors and URL that
are specific to the Gemini web UI.

Selector maintenance
--------------------
If the Gemini UI changes and the backend stops working, this is the only
file that needs updating.  Inspect the live page in a visible browser
(browser-ai serve --backend gemini --visible) and update the relevant
selector list below.
"""

from __future__ import annotations

from browser_ai.backends.base import BaseBrowserBackend


class GeminiBackend(BaseBrowserBackend):
    """Playwright-backed browser session for Google Gemini."""

    label = "gemini"
    URL = "https://gemini.google.com/app"

    # Consent / cookie dialogs shown on first visit or region change.
    CONSENT_SELECTORS = [
        'button:has-text("Accept all")',
        'button:has-text("I agree")',
        'button:has-text("Agree")',
        '[aria-label="Accept all"]',
    ]

    # Ordered list of selectors tried to locate the chat input box.
    # The first visible match wins.
    INPUT_SELECTORS = [
        'rich-textarea div[contenteditable="true"]',
        'div[contenteditable="true"][aria-label*="message"]',
        'div[contenteditable="true"][aria-label*="prompt"]',
        'div[contenteditable="true"][aria-label*="Enter"]',
        'div[contenteditable="true"]',
        "textarea",
        # Broad fallbacks also used by _goto_and_prepare to detect page ready
        'rich-textarea',
        '[contenteditable="true"][placeholder]',
        'div[role="textbox"]',
    ]

    # Ordered list of selectors tried to locate the send button.
    SEND_SELECTORS = [
        'button[aria-label*="Send"]',
        'button[aria-label*="send"]',
        'button[data-test-id="send-button"]',
        'button.send-button',
        'button[jsname="V67aGc"]',
    ]

    # Selectors that match individual response bubbles.
    # The count before submission is compared to the count after to detect
    # when a new response has appeared.
    RESPONSE_SELECTORS = [
        "model-response",
        "[data-response-index]",
        ".model-response-text",
        "message-content.model-response-text",
        ".response-content",
        ".markdown",
    ]

    # Selectors visible while the model is generating.
    # When none match and text is stable, generation is considered complete.
    GENERATING_SELECTORS = [
        '[data-test-id="stop-generating-button"]',
        'button[aria-label*="Stop"]',
        'mat-spinner',
    ]