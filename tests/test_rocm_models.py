import subprocess
import sys

import pytest


def run_rocm_smoke(model_name: str) -> None:
    # Initialize Torch's HIP runtime before Thinc imports CuPy. The reverse
    # order loads two HIP runtime copies and leaves Torch unable to see the GPU.
    import torch

    torch_rocm_available = torch.cuda.is_available()

    import cupy
    import spacy
    from thinc.api import get_current_ops

    from tests.model_smoke import assert_representative_parse, representative_parse

    spacy.require_cpu()
    cpu_parse = representative_parse(spacy.load(model_name))
    assert_representative_parse(cpu_parse)

    assert cupy.cuda.runtime.is_hip
    assert torch.version.hip is not None
    assert torch_rocm_available
    assert spacy.require_gpu(0)
    assert get_current_ops().xp is cupy

    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.reset_peak_memory_stats(0)
    gpu_parse = representative_parse(spacy.load(model_name))
    cupy.cuda.runtime.deviceSynchronize()

    assert_representative_parse(gpu_parse)
    assert gpu_parse == cpu_parse
    if model_name == "ja_ginza":
        assert cupy.get_default_memory_pool().total_bytes() > 0
    else:
        assert torch.cuda.max_memory_allocated(0) > 0


@pytest.mark.nlp_model
@pytest.mark.rocm
@pytest.mark.parametrize("model_name", ["ja_ginza", "ja_ginza_electra"])
def test_model_matches_cpu_parse_on_rocm(model_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_rocm_models", model_name],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    run_rocm_smoke(sys.argv[1])
