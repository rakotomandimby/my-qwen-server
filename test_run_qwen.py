import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "run_qwen.py"


def load_run_qwen_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=0),
    )
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = object
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.BitsAndBytesConfig = object

    original_modules = {
        "torch": sys.modules.get("torch"),
        "transformers": sys.modules.get("transformers"),
    }
    sys.modules["torch"] = fake_torch
    sys.modules["transformers"] = fake_transformers

    try:
        spec = importlib.util.spec_from_file_location("run_qwen_under_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


class RunQwenConfigCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_qwen = load_run_qwen_module()

    def test_text_config_fields_are_copied_to_top_level(self):
        config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                num_hidden_layers=8,
                layer_types=["linear_attention"] * 8,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                linear_num_key_heads=16,
            )
        )

        self.run_qwen.ensure_text_config_fields(config)

        self.assertEqual(config.num_hidden_layers, 8)
        self.assertEqual(config.layer_types, ["linear_attention"] * 8)
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(config.linear_num_key_heads, 16)

    def test_layer_types_default_matches_qwen35_pattern(self):
        config = types.SimpleNamespace(
            num_hidden_layers=6,
            text_config={"full_attention_interval": 3},
        )

        self.run_qwen.ensure_layer_types(config)

        self.assertEqual(
            config.layer_types,
            [
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )

    def test_layer_types_uses_root_full_attention_interval_when_needed(self):
        config = types.SimpleNamespace(
            num_hidden_layers=4,
            full_attention_interval=2,
            text_config={},
        )

        self.run_qwen.ensure_layer_types(config)

        self.assertEqual(
            config.layer_types,
            [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
        )

    def test_layer_types_rejects_non_positive_interval(self):
        config = types.SimpleNamespace(
            num_hidden_layers=4,
            full_attention_interval=0,
            text_config={},
        )

        with self.assertRaisesRegex(ValueError, "full_attention_interval"):
            self.run_qwen.ensure_layer_types(config)

    def test_vocab_and_pad_token_id_support_dict_text_config(self):
        config = types.SimpleNamespace(
            text_config={
                "vocab_size": 1234,
                "pad_token_id": 42,
            },
            eos_token_id=7,
        )
        tokenizer = types.SimpleNamespace(vocab_size=None, pad_token_id=None)

        self.run_qwen.ensure_vocab_size(config, tokenizer)
        self.run_qwen.ensure_pad_token_id(config, tokenizer)

        self.assertEqual(config.vocab_size, 1234)
        self.assertEqual(config.pad_token_id, 42)

    def test_prefers_conditional_generation_loader_for_composite_qwen35_config(self):
        causal_loader = type(
            "AutoModelForCausalLM",
            (),
            {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
        )
        conditional_loader = type(
            "Qwen3_5ForConditionalGeneration",
            (),
            {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
        )
        self.run_qwen.AutoModelForCausalLM = causal_loader
        self.run_qwen.transformers.Qwen3_5ForConditionalGeneration = conditional_loader

        config = types.SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            text_config={},
            vision_config={},
        )

        loader = self.run_qwen.get_model_loader(config)

        self.assertIs(loader, conditional_loader)

    def test_falls_back_to_causal_lm_loader_when_conditional_loader_is_unavailable(self):
        causal_loader = type(
            "AutoModelForCausalLM",
            (),
            {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
        )
        self.run_qwen.AutoModelForCausalLM = causal_loader
        self.run_qwen.transformers.Qwen3_5ForConditionalGeneration = None

        config = types.SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            text_config={},
            vision_config={},
        )

        loader = self.run_qwen.get_model_loader(config)

        self.assertIs(loader, causal_loader)


if __name__ == "__main__":
    unittest.main()
