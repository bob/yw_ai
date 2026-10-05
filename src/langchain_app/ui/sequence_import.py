"""Browser-side handoff to the authenticated Asana Catalog import page."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from langchain_app.ui.sequence import SequenceView


def _save_component():
    """Ensure the component is registered in the active Streamlit runtime."""
    return st.components.v2.component(
        "asana_catalog_sequence_import",
        html='<button type="button">Save</button><div role="status" aria-live="polite"></div>',
        css="""
        button { font: inherit; color: var(--st-text-color); background: var(--st-background-color);
          border: 1px solid var(--st-border-color); border-radius: 0.5rem;
          padding: 0.45rem 0.75rem; cursor: pointer; }
        button:hover { border-color: var(--st-primary-color); }
        button:disabled { opacity: 0.6; cursor: default; }
        [role=status] { font: inherit; margin-top: 0.25rem; }
        """,
        js="""
        export default function ({ data, parentElement }) {
          const button = parentElement.querySelector('button');
          const status = parentElement.querySelector('[role=status]');
          let cleanup = () => {};
          button.onclick = () => {
            cleanup();
            const catalogOrigin = new URL(data.catalog_url).origin;
            const url = new URL('/sequences/import', catalogOrigin);
            url.searchParams.set('origin', window.location.origin);
            const receiver = window.open(url.href, '_blank');
            if (!receiver) {
              status.textContent = 'Allow popups to save the sequence.';
              return;
            }
            button.disabled = true;
            status.textContent = 'Waiting for Asana Catalog. Sign in in the new tab if needed.';
            const onMessage = (event) => {
              if (event.origin !== catalogOrigin || event.source !== receiver) return;
              switch (event.data?.type) {
                case 'asana-catalog:sequence-import:ready':
                  receiver.postMessage({
                    type: 'asana-catalog:sequence-import:save', sequence: data.sequence,
                  }, catalogOrigin);
                  status.textContent = 'Saving in Asana Catalog…';
                  break;
                case 'asana-catalog:sequence-import:saved':
                  status.textContent = 'Saved. The sequence is open in the new tab.';
                  cleanup();
                  break;
                case 'asana-catalog:sequence-import:error':
                  status.textContent = event.data.error || 'Could not save sequence.';
                  // Keep listening: the import page supports retrying the request.
                  break;
              }
            };
            window.addEventListener('message', onMessage);
            const closedTimer = window.setInterval(() => {
              if (!receiver.closed) return;
              status.textContent = 'Import tab was closed before saving was confirmed. Try Save again.';
              cleanup();
            }, 1000);
            cleanup = () => {
              window.removeEventListener('message', onMessage);
              window.clearInterval(closedTimer);
              button.disabled = false;
            };
          };
          return () => { cleanup(); button.onclick = null; };
        }
        """,
    )


def render_sequence_import(sequence: SequenceView, *, catalog_url: str, key: str) -> None:
    _save_component()(data={"sequence": sequence, "catalog_url": catalog_url}, key=key)
