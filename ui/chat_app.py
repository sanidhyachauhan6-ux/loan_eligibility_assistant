import os
import time
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv


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


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def ask_api(prompt):
    try:
        idempotency_key = str(uuid.uuid4())

        response = requests.post(
            f"{API_URL}/ask",
            headers={
                "Accept": "application/json",
                "X-API-Key": API_KEY,
                "Idempotency-Key": idempotency_key,
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
            doc = citation.get(
                "doc",
                "Policy document",
            )

            text = citation.get(
                "text",
                "",
            )

            st.markdown(f"**{i}. {doc}**")

            if text:
                st.caption(text)


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

        with st.spinner(
            "LoanAssist is preparing your response..."
        ):
            result = ask_api(prompt)

        elapsed = time.perf_counter() - start_time

        if result:

            answer = result.get(
                "answer",
                "",
            )

            decision = result.get(
                "decision",
                None,
            )

            citations = result.get(
                "citations",
                [],
            )

            refused = result.get(
                "refused",
                False,
            )

            reason = result.get(
                "reason",
                None,
            )

            # -------------------------------------------------------
            # Answer
            # -------------------------------------------------------

            if answer:
                st.markdown(answer)
            else:
                st.warning(
                    "LoanAssist returned no answer."
                )

            # -------------------------------------------------------
            # Eligibility decision
            #
            # General conversation should have decision=None.
            # Therefore no eligibility badge is shown for:
            #
            # "What are you?"
            # "Hello"
            # "What can you help me with?"
            # etc.
            # -------------------------------------------------------

            if decision:
                display_decision(decision)

            # -------------------------------------------------------
            # Refusal
            # -------------------------------------------------------

            if refused:
                st.warning(
                    "Request refused."
                )

            # Don't expose internal reason to the applicant.
            # Keep it for backend/audit logs.

            # -------------------------------------------------------
            # Citations
            # -------------------------------------------------------

            display_citations(citations)

            # -------------------------------------------------------
            # Developer latency information
            #
            # Keep this while developing/testing.
            # Remove or hide behind a debug flag in production.
            # -------------------------------------------------------

            st.caption(
                f"Response time: {elapsed:.2f}s"
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

        else:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Sorry, I couldn't generate a response. "
                        "Please try again."
                    ),
                    "decision": None,
                    "citations": [],
                }
            )