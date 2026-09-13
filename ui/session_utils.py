import uuid


def get_session_id(state):
    """Return the same per-browser session id for the life of the UI session.

    This stored id is passed to the API as the X-Session-Id header so the
    backend can reuse the same conversation history, audit file and trace group
    instead of generating a fresh anonymous session id on each request.
    """
    session_id = state.get("session_id")
    if not session_id:
        session_id = f"user-{uuid.uuid4().hex[:24]}"
        state["session_id"] = session_id
    return session_id
