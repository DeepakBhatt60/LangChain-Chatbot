import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI

load_dotenv()

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 AI Chat Assistant")
st.caption("Your AI companion powered by LangChain + Mistral AI")

# --------------------------------------------------
# Modes
# --------------------------------------------------

MODES = {
    "🤝 Best Friend":
        "Act as my best friend. Be casual, funny, honest, supportive, and friendly.",

    "🧠 Mentor":
        "Act as a wise mentor. Give thoughtful, practical, encouraging advice and ask useful guiding questions.",

    "😂 Comedian":
        "Act as a comedian. Respond with humor, witty jokes, and funny observations while still answering the question.",

    "👨‍🏫 Strict Teacher":
        "Act as a strict but helpful teacher. Correct mistakes clearly, explain concepts simply, and push me to improve.",

    "🔥 Motivator":
        "Act as an energetic motivational coach. Encourage me, challenge me, and push me toward taking action.",

    "💼 Career Coach":
        "Act as a professional career coach. Give practical advice about jobs, skills, interviews, resumes, and career growth.",

    "💡 Creative Thinker":
        "Act as a creative problem solver. Generate unique ideas, think outside the box, and provide multiple creative approaches.",

    "😎 Casual Assistant":
        "Act as a friendly AI assistant. Be conversational, helpful, concise, and easy to understand."
}

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🎭 Chat Mode")

selected_mode = st.sidebar.radio(
    "Choose a personality:",
    list(MODES.keys())
)

st.sidebar.divider()

st.sidebar.info(
    "💡 Select a personality to change how the AI responds."
)

st.sidebar.caption("Model: ministral-3b-2512")
st.sidebar.caption("Powered by Mistral AI + LangChain")

# --------------------------------------------------
# Initialize Model
# --------------------------------------------------

if "model" not in st.session_state:

    st.session_state.model = ChatMistralAI(
        model="ministral-3b-2512",
        temperature=0.9,
        api_key=st.secrets["MISTRAL_API_KEY"]
    )

# --------------------------------------------------
# Initialize / Reset Chat History
# --------------------------------------------------

if (
    "current_mode" not in st.session_state
    or st.session_state.current_mode != selected_mode
):

    st.session_state.current_mode = selected_mode

    st.session_state.messages = [
        SystemMessage(
            content=MODES[selected_mode]
        )
    ]

# --------------------------------------------------
# Reset Chat Button
# --------------------------------------------------

if st.sidebar.button("🔄 Reset Chat", use_container_width=True):

    st.session_state.messages = [
        SystemMessage(
            content=MODES[selected_mode]
        )
    ]

    st.rerun()

# --------------------------------------------------
# Welcome Message
# --------------------------------------------------

if len(st.session_state.messages) == 1:

    with st.chat_message("assistant"):

        st.write(
            f"👋 Hello! I'm your **{selected_mode}** AI assistant. "
            "How can I help you today?"
        )

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for msg in st.session_state.messages:

    if isinstance(msg, HumanMessage):

        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):

        with st.chat_message("assistant"):
            st.write(msg.content)

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

user_input = st.chat_input(
    "💬 Type your message..."
)

if user_input:

    # Add user message
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            response = st.session_state.model.invoke(
                st.session_state.messages
            )

        st.write(response.content)

    # Save AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )