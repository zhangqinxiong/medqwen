"""
server.py
Serves both models on port 7860 via gr.mount_gradio_app.
  /base     -> Base Qwen2-1.5B
  /medical  -> Fine-tuned (Medical)

Models load lazily on first request to each path.
Only one model resides in GPU memory at a time.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, gc
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from unsloth import FastLanguageModel
import uvicorn

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
MERGED_PATH = "./medical_qwen_merged"
MAX_SEQ_LEN = 1024
SYSTEM_PROMPT = "你是一个专业的临床医生，请根据患者的症状提供准确、严谨且温暖的医学建议。"

model = None
tokenizer = None
current_model = None


def load_model(mt):
    global model, tokenizer, current_model
    if current_model == mt:
        return model, tokenizer
    # Unload
    if model is not None:
        print(f"Unloading {current_model}")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        model = None
        tokenizer = None
    # Load
    name = MERGED_PATH if mt == "ft" else MODEL_NAME
    label = "Fine-tuned" if mt == "ft" else "Base"
    print(f"Loading {label}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        dtype=None,
    )
    model = FastLanguageModel.for_inference(model)
    current_model = mt
    print(f"{label} loaded. GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    return model, tokenizer


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return " ".join(texts)
    return str(content) if content else ""


def make_predict_fn(mt):
    def predict(message, history):
        m, t = load_model(mt)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            if isinstance(turn, dict):
                msgs.append({"role": turn.get("role", "user"), "content": extract_text(turn.get("content", ""))})
            elif isinstance(turn, (tuple, list)):
                msgs.append({"role": "user", "content": extract_text(turn[0])})
                if len(turn) > 1 and turn[1]:
                    msgs.append({"role": "assistant", "content": extract_text(turn[1])})
        msgs.append({"role": "user", "content": extract_text(message)})
        text = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = t([text], return_tensors="pt").to("cuda")
        outputs = m.generate(
            **inputs, max_new_tokens=512, temperature=0.7,
            top_p=0.9, repetition_penalty=1.05,
        )
        return t.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return predict


def build_block(title, mt):
    with gr.Blocks(title=title, fill_height=True) as block:
        gr.Markdown(f"# {title}")
        gr.ChatInterface(
            fn=make_predict_fn(mt),
            examples=[
                ["头痛发烧喉咙痛三天了，可能是什么病？"],
                ["膝盖上长红疙瘩，不疼不痒，是什么问题？"],
                ["晚上失眠躺一两个小时睡不着，怎么办？"],
                ["血压偏高140/90需要注意什么？"],
            ],
        )
    return block


# Load FT model at startup (default)
print("Starting up - loading Fine-tuned model...")
load_model("ft")

app = FastAPI()
app = gr.mount_gradio_app(app, build_block("Fine-tuned (Medical)", "ft"), path="/medical")
app = gr.mount_gradio_app(app, build_block("Base Qwen2-1.5B", "base"), path="/base")


@app.get("/")
async def root():
    html = """
    <!DOCTYPE html>
    <html><body style="font-family:sans-serif;padding:2em">
    <h1>Medical Qwen Chat</h1>
    <p>Select a model to chat:</p>
    <ul>
      <li><a href="/base">/base</a> - Base Qwen2-1.5B</li>
      <li><a href="/medical">/medical</a> - Fine-tuned (Medical)</li>
    </ul>
    </body></html>
    """
    return HTMLResponse(html)


if __name__ == "__main__":
    print("Server running on http://localhost:7860")
    print("  /base     -> Base Qwen2-1.5B")
    print("  /medical  -> Fine-tuned (Medical)")
    uvicorn.run(app, host="0.0.0.0", port=7860)
