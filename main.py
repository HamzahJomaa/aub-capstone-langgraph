from importlib.metadata import metadata

import gradio as gr
from langgraph_sdk import get_client
from gradio import ChatMessage

# Setup LangGraph client
client = get_client(
    url="https://aub-capstone-2-25bde6b4ddc35bc2adaf9804bd676a1b.us.langgraph.app",
    api_key="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"
)
assistant_id = "agent"

# --- LangGraph thread management ---
async def create_new_thread():
    thread = await client.threads.create()
    return thread["thread_id"], []

async def delete_thread(thread_id):
    if thread_id:
        await client.threads.delete(thread_id=thread_id)
    thread = await client.threads.create()
    return thread["thread_id"], []

# --- Streaming Response with yield ---
async def chat_stream(message, thread_id, history):
    _input = {
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ]
    }

    history.append(
        ChatMessage(role="user", content=str(message)))
    yield history, thread_id  # Show user's message immediately

    async for chunk in client.runs.stream(
            thread_id,
            assistant_id,
            input=_input,
            stream_mode=["values", "messages"]
    ):
        if isinstance(chunk.data, dict):
            if chunk.data.get("messages"):
                message = chunk.data["messages"][-1]
                assistant_reply = message["content"]
                message_type = message["type"]
                if message_type == "tool":
                    history.append(
                            ChatMessage(role="assistant", metadata={"title": assistant_reply, "tool_called": True}, content=assistant_reply))
                elif message_type == "ai":
                    history.append(
                        ChatMessage(role="assistant", content=assistant_reply))
                else:
                    history.append(
                        ChatMessage(role="assistant", content=""))
                yield history, thread_id  # Yield after handling message
        elif isinstance(chunk.data, list):
            content_chunk = chunk.data[0]["content"]
            if (chunk.data[0]["response_metadata"].get("finish_reason") and chunk.data[0]["response_metadata"].get("finish_reason") == "stop"):
                history[-1] = ChatMessage(role="assistant", metadata=history[-1].metadata, content=content_chunk)
            yield history, thread_id


# --- Gradio UI ---
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML('<iframe src="https://46.101.200.219" width="100%" style="border:none;height:80vh; background: white; margin:0; padding:0; display:block;"></iframe>')
        
        with gr.Column():
            gr.Markdown("## 🤖 LangGraph Assistant with Streaming & Tools")

            chatbot = gr.Chatbot(type="messages", label="chatbot")
            message = gr.Textbox(placeholder="Ask me anything...")
            state_thread = gr.State()
            state_history = gr.State([])

            with gr.Row():
                send_btn = gr.Button("Send")
                new_btn = gr.Button("🧵 New Thread", variant="secondary")
                delete_btn = gr.Button("🗑️ Delete Thread", variant="stop")

    send_btn.click(
        chat_stream,
        inputs=[message, state_thread, state_history],
        outputs=[chatbot, state_thread],
        show_progress=True  # Optional: shows loading spinner
    )
    message.submit(
        chat_stream,
        inputs=[message, state_thread, state_history],
        outputs=[chatbot, state_thread],
        show_progress=True
    )

    new_btn.click(
        create_new_thread,
        outputs=[state_thread, chatbot]
    )

    delete_btn.click(
        delete_thread,
        inputs=[state_thread],
        outputs=[state_thread, chatbot]
    )

    demo.load(create_new_thread, outputs=[state_thread, chatbot])
demo.launch()
