<script lang="ts">
  import { openaiApiKey } from './core'

  let keyInput = ''
  let showModal = true
  let error = ''

  function handleSubmit() {
    if (!keyInput.trim()) {
      error = 'Please enter your API key'
      return
    }

    if (!keyInput.startsWith('sk-')) {
      error = 'API key should start with sk-'
      return
    }

    openaiApiKey.set(keyInput)
    showModal = false
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && keyInput.trim()) {
      handleSubmit()
    }
  }
</script>

{#if showModal}
  <div class="modal-overlay">
    <div class="modal">
      <h2>OpenAI API Key</h2>
      <p class="subtitle">Enter your API key to use texture generation and reconstruction features.</p>

      <input
        type="password"
        placeholder="sk-..."
        bind:value={keyInput}
        on:keydown={handleKeyDown}
        autofocus
      />

      {#if error}
        <p class="error">{error}</p>
      {/if}

      <p class="info">Your key is stored in memory for this session only and not saved to disk.</p>

      <button on:click={handleSubmit} disabled={!keyInput.trim()}>
        Continue
      </button>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
  }

  .modal {
    background: #2a3a4a;
    padding: 2.5rem;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    min-width: 350px;
    color: #e3ebf7;
  }

  h2 {
    margin: 0 0 0.5rem 0;
    font-size: 1.5rem;
    color: #ffffff;
  }

  .subtitle {
    margin: 0 0 1.5rem 0;
    color: #b0b8c0;
    font-size: 0.9rem;
  }

  input {
    width: 100%;
    padding: 0.75rem;
    margin-bottom: 1rem;
    background: #1a2a3a;
    border: 1px solid #3a4a5a;
    border-radius: 6px;
    color: #e3ebf7;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    transition: border-color 0.2s;
  }

  input:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.1);
  }

  .error {
    color: #ff6b6b;
    font-size: 0.85rem;
    margin: 0.5rem 0 1rem 0;
  }

  .info {
    color: #8a9aaa;
    font-size: 0.8rem;
    margin: 1rem 0;
    line-height: 1.4;
  }

  button {
    width: 100%;
    padding: 0.75rem;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 500;
    transition: background 0.2s;
  }

  button:hover:not(:disabled) {
    background: #0056b3;
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
