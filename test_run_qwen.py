import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import run_qwen


class ChatHandlerTests(unittest.TestCase):
    def start_server(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(1)
            self.assertFalse(thread.is_alive(), "Server thread did not stop cleanly")

        self.addCleanup(cleanup)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def post_json(self, url, payload):
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_post_returns_generated_text(self):
        handler = run_qwen.make_handler(lambda prompt: f"Echo: {prompt}")
        base_url = self.start_server(handler)

        status, payload = self.post_json(base_url + "/", {"prompt": "tell me a joke"})

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"data": "Echo: tell me a joke"})

    def test_post_rejects_missing_prompt(self):
        handler = run_qwen.make_handler(lambda prompt: prompt)
        base_url = self.start_server(handler)

        request = Request(
            base_url + "/",
            data=json.dumps({"wrong": "field"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises(HTTPError) as context:
            urlopen(request)

        self.assertEqual(context.exception.code, 400)
        body = json.loads(context.exception.read().decode("utf-8"))
        self.assertEqual(body, {"error": "Field 'prompt' must be a non-empty string"})

    def test_post_returns_server_error_when_generation_fails(self):
        def broken_generate(prompt):
            raise RuntimeError("generation failed")

        handler = run_qwen.make_handler(broken_generate)
        base_url = self.start_server(handler)

        request = Request(
            base_url + "/",
            data=json.dumps({"prompt": "tell me a joke"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises(HTTPError) as context:
            urlopen(request)

        self.assertEqual(context.exception.code, 500)
        body = json.loads(context.exception.read().decode("utf-8"))
        self.assertEqual(body, {"error": "generation failed"})


if __name__ == "__main__":
    unittest.main()
