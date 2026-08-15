import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def project_configuration() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_uv_selects_only_python_314() -> None:
    configuration = project_configuration()

    assert configuration["project"]["requires-python"] == ">=3.14,<3.15"
    assert (ROOT / ".python-version").read_text(encoding="utf-8") == "3.14\n"


def test_backend_extra_owns_every_direct_server_dependency() -> None:
    configuration = project_configuration()

    assert configuration["project"]["dependencies"] == ["duckdb>=1.5.5,<2"]
    assert configuration["project"]["optional-dependencies"]["backend"] == [
        "anyio>=4.14.2,<5",
        "fastapi>=0.141.1,<1",
        "pydantic>=2.13.4,<3",
        "uvicorn>=0.52.1,<1",
    ]


def test_model_and_accelerator_extras_are_orthogonal() -> None:
    configuration = project_configuration()
    extras = configuration["project"]["optional-dependencies"]

    assert extras["builder"] == [
        "click>=8.4.2,<9",
        "polars[pyarrow,excel]>=1.43.2,<2",
        "ginza==5.2.0",
        "ja-ginza==5.2.0",
        "confection>=0.1.5,<1",
        "numpy>=2,<3",
        "spacy>=3.8.11,<3.8.12",
        "wtpsplit>=2.2.1,<3",
    ]
    assert extras["electra"] == [
        "ja-ginza-electra==5.2.0",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
    ]
    assert extras["cpu"] == ["torch==2.13.0+cpu"]
    assert extras["cuda"] == [
        "torch==2.13.0+cu126",
        "cupy-cuda12x>=14.1.1,<15",
    ]
    platform_marker = "sys_platform == 'linux' and platform_machine == 'x86_64'"
    assert extras["rocm"] == [
        f"torch==2.13.0+rocm7.2; {platform_marker}",
        f"cupy-rocm-7-0==14.1.1; {platform_marker}",
        f"triton-rocm==3.7.1; {platform_marker}",
    ]


def test_accelerator_extras_select_one_matching_torch_index() -> None:
    configuration = project_configuration()
    uv = configuration["tool"]["uv"]

    conflicts = {
        frozenset(member["extra"] for member in conflict)
        for conflict in uv["conflicts"]
    }
    assert conflicts == {
        frozenset(("cpu", "cuda")),
        frozenset(("cpu", "rocm")),
        frozenset(("cuda", "rocm")),
    }
    assert uv["sources"]["torch"] == [
        {"index": "pytorch-cpu", "extra": "cpu"},
        {"index": "pytorch-cuda", "extra": "cuda"},
        {"index": "pytorch-rocm", "extra": "rocm"},
    ]
    indexes = {entry["name"]: entry for entry in uv["index"]}
    assert indexes == {
        "pytorch-cpu": {
            "name": "pytorch-cpu",
            "url": "https://download.pytorch.org/whl/cpu",
            "explicit": True,
        },
        "pytorch-cuda": {
            "name": "pytorch-cuda",
            "url": "https://download.pytorch.org/whl/cu126",
            "explicit": True,
        },
        "pytorch-rocm": {
            "name": "pytorch-rocm",
            "url": "https://download.pytorch.org/whl/rocm7.2",
            "explicit": True,
        },
    }


def test_python_314_compatibility_exceptions_are_narrow() -> None:
    uv = project_configuration()["tool"]["uv"]

    assert uv["override-dependencies"] == ["transformers==4.57.6"]
    assert uv["extra-build-variables"] == {
        "spacy-alignments": {"PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1"},
        "sudachipy": {"PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1"},
    }


def test_rocm_transitive_packages_use_the_rocm_index() -> None:
    sources = project_configuration()["tool"]["uv"]["sources"]

    assert sources["triton-rocm"] == [{"index": "pytorch-rocm", "extra": "rocm"}]


def test_nix_consumers_use_python_314() -> None:
    flake = (ROOT / "flake.nix").read_text(encoding="utf-8")

    assert "python = pkgs.python314;" in flake
    assert "smokeFixturePython = pkgs.python314.withPackages" in flake
    assert "pkgs.python312" not in flake


def test_nix_builds_sudachipy_with_its_legacy_python_314_requirements() -> None:
    flake = (ROOT / "flake.nix").read_text(encoding="utf-8")

    assert '"sudachipy"' in flake
    assert "setuptools-rust = [ ];" in flake
    assert "pkgs.cargo" in flake
    assert "pkgs.rustc" in flake
    assert "pkgs.rustPlatform.cargoSetupHook" in flake
    assert "pkgs.rustPlatform.importCargoLock" in flake
    assert 'PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1";' in flake
