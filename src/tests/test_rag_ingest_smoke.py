import os
import importlib.util
import unittest
from unittest.mock import patch

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(dep) is not None
    for dep in (
        "faiss",
        "sentence_transformers",
        "pypdf",
        "numpy",
        "dotenv",
    )
)

if HAS_RUNTIME_DEPS:
    import ingest
    import rag


@unittest.skipUnless(HAS_RUNTIME_DEPS, "Missing runtime deps required by ingest/rag")
class RagIngestSmokeTests(unittest.TestCase):
    def test_build_prompt_includes_question_and_context(self):
        prompt = rag.build_prompt("What is stress?", ["Stress is a response."])
        self.assertIn("What is stress?", prompt)
        self.assertIn("Stress is a response.", prompt)

    def test_generate_falls_back_when_provider_fails(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False):
            with patch.object(rag, "retrieve", return_value=["ctx"]):
                with patch.object(rag, "_gen_gemini", side_effect=RuntimeError("boom")):
                    with patch.object(rag, "_gen_flan", return_value="offline answer"):
                        out = rag.generate("help me")
        self.assertIn("offline answer", out)
        self.assertIn("general information, not medical advice", out)

    def test_chunk_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            ingest.chunk("a b c d e", size=5, overlap=5)


if __name__ == "__main__":
    unittest.main()
