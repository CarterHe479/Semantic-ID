import unittest

import importlib
import numpy as np
import torch

from src.models.RQ_VAE import ResidualQuantizer
from src.models.VQ_VAE import VectorQuantizerEMA


def require_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest(f"{name} is not installed") from exc


class QuantizerEvalModeTest(unittest.TestCase):
    def test_vq_eval_does_not_update_codebook(self):
        torch.manual_seed(0)
        quant = VectorQuantizerEMA(K=8, d=4)
        quant.eval()
        before = quant.codebook.detach().clone()

        with torch.no_grad():
            quant(torch.randn(5, 4))

        self.assertTrue(torch.equal(before, quant.codebook.detach()))

    def test_rq_eval_does_not_update_codebooks(self):
        torch.manual_seed(0)
        quant = ResidualQuantizer(L=2, K=8, d=4)
        quant.eval()
        before = [cb.detach().clone() for cb in quant.codebooks]

        with torch.no_grad():
            quant(torch.randn(5, 4))

        for old, new in zip(before, quant.codebooks):
            self.assertTrue(torch.equal(old, new.detach()))

    def test_init_from_codebook_syncs_ema_buffers(self):
        codebook = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        quant = VectorQuantizerEMA(K=3, d=4)
        quant.init_codebook(codebook)

        self.assertTrue(torch.equal(quant.codebook.detach(), codebook))
        self.assertTrue(torch.equal(quant.ema_weight, codebook))
        self.assertTrue(torch.equal(quant.ema_cluster_size, torch.ones(3)))


class EvalFallbackTest(unittest.TestCase):
    def test_empty_candidates_are_misses_by_default(self):
        pd = require_module("pandas")
        from src.retrieval.eval_offline import evaluate_method

        X = np.eye(2, dtype=np.float32)
        ids = np.array(["a", "b"])
        id2idx = {"a": 0, "b": 1}
        profiles = {"u1": X[0]}
        test_df = pd.DataFrame([{"user_id": "u1", "item_id": "a"}])
        codebooks = np.array([[0.0, 1.0]], dtype=np.float32)

        res = evaluate_method(
            name="vq",
            codebooks=codebooks,
            inverted={},
            test_df=test_df,
            profiles=profiles,
            X=X,
            ids=ids,
            id2idx=id2idx,
            topk_list=(1,),
        )

        self.assertEqual(res["Recall@1"], 0.0)
        self.assertEqual(res["empty_candidate_queries"], 1)
        self.assertEqual(res["full_search_fallback_queries"], 0)

    def test_empty_candidates_can_use_explicit_full_search_fallback(self):
        pd = require_module("pandas")
        from src.retrieval.eval_offline import evaluate_method

        X = np.eye(2, dtype=np.float32)
        ids = np.array(["a", "b"])
        id2idx = {"a": 0, "b": 1}
        profiles = {"u1": X[0]}
        test_df = pd.DataFrame([{"user_id": "u1", "item_id": "a"}])
        codebooks = np.array([[0.0, 1.0]], dtype=np.float32)

        res = evaluate_method(
            name="vq",
            codebooks=codebooks,
            inverted={},
            test_df=test_df,
            profiles=profiles,
            X=X,
            ids=ids,
            id2idx=id2idx,
            topk_list=(1,),
            allow_full_search_fallback=True,
        )

        self.assertEqual(res["Recall@1"], 1.0)
        self.assertEqual(res["full_search_fallback_queries"], 1)


if __name__ == "__main__":
    unittest.main()
