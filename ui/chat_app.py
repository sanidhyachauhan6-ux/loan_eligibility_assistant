import os
import json
import time
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv
from session_utils import get_session_id


st.set_page_config(
    page_title="LoanAssist",
    page_icon="💬",
    layout="centered",
)


load_dotenv()

API_URL = st.sidebar.text_input(
    "API URL",
    os.getenv("API_URL", "http://localhost:8001"),
)

API_KEY = os.getenv("API_KEY", "")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💬 LoanAssist")

st.caption(
    "Your loan eligibility assistant for preliminary pre-qualification."
)

with st.sidebar:
    st.header("LoanAssist")

    st.info(
        "This is a preliminary assessment based on the available "
        "eligibility rules. Final approval requires normal verification "
        "and approval."
    )

    if st.button("🆕 New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("API")
    st.caption(API_URL)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = get_session_id(st.session_state)

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def ask_api(prompt):
    try:
        idempotency_key = str(uuid.uuid4())
        session_id = get_session_id(st.session_state)

        response = requests.post(
            f"{API_URL}/ask",
            headers={
                "Accept": "application/json",
                "X-API-Key": API_KEY,
                "Idempotency-Key": idempotency_key,
                "X-Session-Id": session_id,
            },
            json={
                "question": prompt,
            },
            timeout=180,
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to LoanAssist at {API_URL}. "
            "Make sure the FastAPI server is running."
        )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "The request took too long to complete. "
            "Please try again."
        )
        return None

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response else "unknown"

        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text if exc.response else str(exc)

        st.error(
            f"LoanAssist returned an error ({status}): {detail}"
        )
        return None

    except requests.exceptions.RequestException as exc:
        st.error(
            f"Could not contact LoanAssist: {exc}"
        )
        return None

    except ValueError:
        st.error(
            "LoanAssist returned an invalid response."
        )
        return None


def stream_api(prompt):
    """Yield SSE events from the streaming API endpoint."""
    try:
        idempotency_key = str(uuid.uuid4())
        session_id = get_session_id(st.session_state)

        with requests.post(
            f"{API_URL}/ask/stream",
            headers={
                "Accept": "text/event-stream",
                "X-API-Key": API_KEY,
                "Idempotency-Key": idempotency_key,
                "X-Session-Id": session_id,
            },
            json={"question": prompt},
            timeout=180,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    yield json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not stream a response from LoanAssist: {exc}")
        yield {"type": "error"}


# ---------------------------------------------------------------------------
# Decision display
# ---------------------------------------------------------------------------

def display_decision(decision):
    """
    Display an eligibility decision.

    General conversational responses have decision=None and therefore
    don't display an eligibility badge.
    """

    if decision == "PRE_QUALIFIED":
        st.success("✅ Pre-qualified")

    elif decision == "NOT_PRE_QUALIFIED":
        st.error("❌ Not pre-qualified")

    elif decision == "NEEDS_INFORMATION":
        st.info("ℹ️ More information needed")

    elif decision == "MANUAL_REVIEW":
        st.warning("⚠️ Manual review required")


# ---------------------------------------------------------------------------
# Policy sources
# ---------------------------------------------------------------------------

def display_citations(citations):
    """Display policy citations without cluttering the main answer."""

    if not citations:
        return

    with st.expander("📚 View policy sources"):
        for i, citation in enumerate(citations, start=1):
            rule_id = citation.get("rule_id")

            text = citation.get(
                "text",
                "",
            )

            st.markdown(f"**{i}. {rule_id}**")

            if text:
                st.caption(text.replace("#", r"\#"))


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(message["content"])

        # Decision and citations are stored as metadata, not inside
        # the conversational text.
        if role == "assistant":

            decision = message.get("decision")

            if decision:
                display_decision(decision)

            citations = message.get("citations", [])

            if citations:
                display_citations(citations)


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

if not st.session_state.messages:

    st.markdown("### How can I help?")

    st.caption(
        "Ask a question about loan eligibility or provide your "
        "details for a preliminary assessment."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Try asking:**

            - What are the eligibility requirements?
            - What information do I need?
            """
        )

    with col2:
        st.markdown(
            """
            **Or tell me:**

            - My age is 45
            - My annual income is ₹800,000
            """
        )


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prompt = st.chat_input(
    "Ask about loan eligibility or enter your details..."
)


# ---------------------------------------------------------------------------
# Handle user message
# ---------------------------------------------------------------------------

if prompt:

    # ---------------------------------------------------------------
    # Store and display user message
    # ---------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------------------------------------------------------
    # Assistant response
    # ---------------------------------------------------------------

    with st.chat_message("assistant"):

        start_time = time.perf_counter()

        result = None
        answer = ""
        decision = ""
        citations = []
        answer_placeholder = st.empty()

        for event in stream_api(prompt):

            event_type = event.get("type")

            # -------------------------------------------------------
            # Streaming answer
            # -------------------------------------------------------
            if event_type == "delta":
                answer += event.get("content", "")

                # Update the SAME placeholder
                answer_placeholder.markdown(answer)

            # -------------------------------------------------------
            # Final structured response
            # -------------------------------------------------------
            elif event_type == "final":
                result = event

                # Expected structure:
                #
                # {
                #     "type": "final",
                #     "answer": "...",
                #     "decision": {...},
                #     "citations": [...]
                # }

                final_answer = event.get("answer", "")
                decision = event.get("decision")
                citations = event.get("citations", [])

                # Only update the placeholder if the final answer
                # is actually different from what was streamed.
                if final_answer and final_answer != answer:
                    answer = final_answer
                    answer_placeholder.markdown(answer)

            # -------------------------------------------------------
            # Error
            # -------------------------------------------------------
            elif event_type == "error":
                st.error(event.get("message", "An error occurred."))
                break

        # -----------------------------------------------------------
        # Everything below happens ONCE, after streaming is complete
        # -----------------------------------------------------------

        response_time = time.perf_counter() - start_time
        
        if not answer:

            answer = (
                "Sorry, I couldn't generate a response. "
                "Please try again."
            )

            answer_placeholder.warning(answer)
        
        # Eligibility decision
        if decision:
            display_decision(decision)

        # Citations
        if citations:
            display_citations(citations)

        st.caption(
            f"Response time: {response_time:.2f}s"
        )

            # -------------------------------------------------------
            # Store structured assistant message
            #
            # IMPORTANT:
            # Don't append citations as a Python list to the message
            # text. Store them separately as metadata.
            # -------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "decision": decision,
                "citations": citations,
            }
        )