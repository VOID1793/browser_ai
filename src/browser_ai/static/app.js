(() => {
  const modelInput = document.querySelector('#model');
  const systemInput = document.querySelector('#systemPrompt');
  const promptInput = document.querySelector('#prompt');
  const sendBtn = document.querySelector('#sendBtn');
  const clearBtn = document.querySelector('#clearBtn');
  const chatEl = document.querySelector('#chat');
  const statusEl = document.querySelector('#status');
  const streamCheckbox = document.querySelector('#stream');

  if (!modelInput || !promptInput || !sendBtn || !clearBtn || !chatEl || !statusEl || !streamCheckbox) {
    console.warn('[browser-ai ui] One or more required elements are missing from the DOM.');
  }

  /** @type {{ role: 'user' | 'assistant', content: string }[]} */
  let messages = [];

  function setStatus(text, isError = false) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.classList.toggle('status-error', !!isError);
  }

  function scrollChatToBottom() {
    if (!chatEl) return;
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function createMessageElement(role, content, isError = false) {
    const item = document.createElement('div');
    item.className = 'chat-item ' + (role === 'assistant' ? 'assistant' : 'user');
    if (isError) {
      item.classList.add('error');
    }

    const roleEl = document.createElement('div');
    roleEl.className = 'chat-role';
    roleEl.textContent = isError ? 'Error' : role;

    const contentEl = document.createElement('div');
    contentEl.className = 'chat-content';
    contentEl.textContent = content;

    item.appendChild(roleEl);
    item.appendChild(contentEl);

    return { item, contentEl };
  }

  function appendMessageToTranscript(role, content, isError = false) {
    if (!chatEl) return null;

    const emptyState = chatEl.querySelector('#transcript-empty');
    if (emptyState) {
      emptyState.remove();
    }

    const { item, contentEl } = createMessageElement(role, content, isError);
    chatEl.appendChild(item);
    scrollChatToBottom();
    return { item, contentEl };
  }

  function buildPayloadMessages(currentUserContent) {
    const payloadMessages = [];

    const systemContent = systemInput ? systemInput.value.trim() : '';
    if (systemContent) {
      payloadMessages.push({
        role: 'system',
        content: systemContent,
      });
    }

    for (const m of messages) {
      payloadMessages.push({ role: m.role, content: m.content });
    }

    payloadMessages.push({
      role: 'user',
      content: currentUserContent,
    });

    return payloadMessages;
  }

  async function sendMessage() {
    if (!modelInput || !promptInput || !streamCheckbox) return;

    const model = modelInput.value.trim() || 'gemini';
    const userContent = promptInput.value.trim();
    const stream = !!streamCheckbox.checked;

    if (!userContent) {
      setStatus('Prompt is empty.', true);
      return;
    }

    setStatus('Sending…');
    if (sendBtn) sendBtn.disabled = true;
    if (clearBtn) clearBtn.disabled = true;

    const userView = appendMessageToTranscript('user', userContent);
    if (!userView) {
      if (sendBtn) sendBtn.disabled = false;
      if (clearBtn) clearBtn.disabled = false;
      setStatus('Unable to render message.', true);
      return;
    }

    const payloadMessages = buildPayloadMessages(userContent);

    const body = {
      model,
      messages: payloadMessages,
      stream,
    };

    let assistantContent = '';
    let assistantView = appendMessageToTranscript('assistant', '');

    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer local-ui',
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status} ${res.statusText || ''} ${text ? `- ${text}` : ''}`.trim());
      }

      if (stream) {
        if (!assistantView) {
          assistantView = appendMessageToTranscript('assistant', '');
        }

        const reader = res.body && res.body.getReader ? res.body.getReader() : null;
        if (!reader) {
          const data = await res.json();
          const choice = data && data.choices && data.choices[0];
          const content = choice && choice.message && choice.message.content;
          assistantContent = typeof content === 'string' ? content : '';
          if (assistantView && assistantView.contentEl) {
            assistantView.contentEl.textContent = assistantContent;
          }
        } else {
          const decoder = new TextDecoder('utf-8');
          let buffer = '';

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const parts = buffer.split('\n');
            buffer = parts.pop() || '';

            for (const line of parts) {
              const trimmed = line.trim();
              if (!trimmed || !trimmed.startsWith('data:')) continue;

              const dataStr = trimmed.slice(5).trim();
              if (!dataStr || dataStr === '[DONE]') {
                continue;
              }

              let parsed;
              try {
                parsed = JSON.parse(dataStr);
              } catch {
                continue;
              }

              try {
                const choice = parsed.choices && parsed.choices[0];
                const delta = choice && choice.delta;
                const piece = delta && typeof delta.content === 'string' ? delta.content : '';
                if (!piece) continue;

                assistantContent += piece;
                if (assistantView && assistantView.contentEl) {
                  assistantView.contentEl.textContent = assistantContent;
                  scrollChatToBottom();
                }
              } catch {
                // ignore malformed chunk
              }
            }
          }
        }
      } else {
        const data = await res.json();
        const choice = data && data.choices && data.choices[0];
        const content = choice && choice.message && choice.message.content;
        assistantContent = typeof content === 'string' ? content : '';

        if (assistantView && assistantView.contentEl) {
          assistantView.contentEl.textContent = assistantContent;
        } else {
          assistantView = appendMessageToTranscript('assistant', assistantContent);
        }
      }

      if (assistantContent) {
        messages.push({ role: 'user', content: userContent });
        messages.push({ role: 'assistant', content: assistantContent });
      }

      promptInput.value = '';
      setStatus('Done.');
    } catch (err) {
      console.error('[browser-ai ui] Error:', err);
      appendMessageToTranscript('assistant', String(err && err.message ? err.message : err), true);
      setStatus('Error.', true);
    } finally {
      if (sendBtn) sendBtn.disabled = false;
      if (clearBtn) clearBtn.disabled = false;
    }
  }

  function clearChat() {
    messages = [];
    if (chatEl) {
      chatEl.innerHTML = '';
      const empty = document.createElement('div');
      empty.id = 'transcript-empty';
      empty.className = 'transcript-empty';
      empty.textContent = 'Messages will appear here once you send a prompt.';
      chatEl.appendChild(empty);
    }
    setStatus('Cleared. Idle.');
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      sendMessage();
    }
  }

  if (sendBtn) {
    sendBtn.addEventListener('click', () => {
      sendMessage();
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      clearChat();
    });
  }

  if (promptInput) {
    promptInput.addEventListener('keydown', handleKeyDown);
  }

  setStatus('Idle · ready to call /v1/chat/completions');
})();

