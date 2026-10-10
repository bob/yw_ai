"""Focus the native chat input once a response has finished."""

from uuid import UUID

import streamlit as st


def focus_chat_input(request_id: UUID) -> None:
    # st.chat_input has no public focus option. Wait for its cleared value and
    # enabled state to reach the browser before focusing, without scrolling.
    st.html(
        """
        <script>
        // Completed response: __REQUEST_ID__
        (() => {
            let attempts = 0;
            const focus = () => {
                const input = document.querySelector('.st-key-sequence-prompt textarea');
                if (input && !input.disabled && input.value === '') {
                    input.focus({preventScroll: true});
                    return;
                }
                if (++attempts < 300) requestAnimationFrame(focus);
            };
            requestAnimationFrame(focus);
        })();
        </script>
        """.replace("__REQUEST_ID__", str(request_id)),
        unsafe_allow_javascript=True,
    )
