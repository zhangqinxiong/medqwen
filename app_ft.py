import os, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import gradio as gr
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./medical_qwen_merged", max_seq_length=1024,
    load_in_4bit=False, dtype=None)
model = FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = "你是一个专业的临床医生，请根据患者的症状提供准确、严谨且温暖的医学建议。"


def extract_text(content):
    """Extract plain text from Gradio's multimodal content format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
        return " ".join(texts)
    return str(content) if content else ""


def predict(msg, history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in history:
        role = "user"
        content = ""
        if isinstance(t, dict):
            role = t.get("role", "user") or "user"
            content = extract_text(t.get("content", ""))
        elif isinstance(t, (tuple, list)):
            content = extract_text(t[0])
            if len(t) > 1 and t[1]:
                msgs.append({"role": "assistant", "content": extract_text(t[1])})
        msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": extract_text(msg)})

    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=512, temperature=0.7,
        top_p=0.9, repetition_penalty=1.05,
    )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


with gr.Blocks(title="Fine-tuned (Medical)", fill_height=True) as demo:
    gr.Markdown("# Fine-tuned (Medical)")
    gr.ChatInterface(fn=predict, examples=[
        ["头痛发烧喉咙痛三天了，可能是什么病？"],
        ["膝盖上长红疙瘩，不疼不痒，是什么问题？"],
        ["晚上失眠躺一两个小时睡不着，怎么办？"],
    ])
demo.launch(server_name="0.0.0.0", server_port=7862)
