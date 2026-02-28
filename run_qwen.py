import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-Next")
_MODEL = None
_TOKENIZER = None


def get_model_and_tokenizer():
  from transformers import AutoModelForCausalLM, AutoTokenizer

  global _MODEL, _TOKENIZER
  if _MODEL is None or _TOKENIZER is None:
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      torch_dtype="auto",
      device_map="auto",
    )
  return _MODEL, _TOKENIZER


def generate_text(prompt, max_new_tokens=2048):
  model, tokenizer = get_model_and_tokenizer()
  messages = [{"role": "user", "content": prompt}]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
  )
  model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
  generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=max_new_tokens,
  )
  output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
  return tokenizer.decode(output_ids, skip_special_tokens=True)


class PromptHandler(BaseHTTPRequestHandler):
  def _send_json(self, status_code, payload):
    encoded = json.dumps(payload).encode("utf-8")
    self.send_response(status_code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(encoded)))
    self.end_headers()
    self.wfile.write(encoded)

  def do_GET(self):
    if self.path == "/health":
      self._send_json(200, {"status": "ok"})
      return
    self._send_json(404, {"error": "not_found"})

  def do_POST(self):
    if self.path != "/prompt":
      self._send_json(404, {"error": "not_found"})
      return
    try:
      length = int(self.headers.get("Content-Length", "0"))
      body = self.rfile.read(length) if length > 0 else b"{}"
      data = json.loads(body.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
      self._send_json(400, {"error": "invalid_json", "details": str(exc)})
      return

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
      self._send_json(400, {"error": "invalid_prompt"})
      return

    max_new_tokens = data.get("max_new_tokens", 2048)
    try:
      content = generate_text(prompt, int(max_new_tokens))
      self._send_json(200, {"content": content})
    except Exception as exc:
      self._send_json(500, {"error": "generation_failed", "details": str(exc)})


if __name__ == "__main__":
  host = os.environ.get("HOST", "0.0.0.0")
  port = int(os.environ.get("PORT", "8000"))
  server = ThreadingHTTPServer((host, port), PromptHandler)
  print(f"Server listening on http://{host}:{port}")
  server.serve_forever()
