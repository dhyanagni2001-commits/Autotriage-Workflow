"""
Reference script for packaging + deploying the fine-tuned retriever and
classifier as real-time SageMaker endpoints, per the "SageMaker
endpoint" option in cloud/README.md's inference-location decision.

*** Not run as part of this project — a real-time endpoint bills per **
*** hour it's up, and this is a portfolio project, not a live service. **
Read it as the deployment story the architecture diagram implies; run it
yourself only if you want an actual endpoint (ml.t2.medium is the
cheapest CPU instance type as of writing — check current pricing first).

Usage (after training on Kaggle and copying artifacts back locally):
    python cloud/sagemaker/deploy_endpoints.py --config config.yaml --which classifier
    python cloud/sagemaker/deploy_endpoints.py --config config.yaml --which retriever
"""
import argparse
import json
import tarfile
from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker.pytorch import PyTorchModel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_classifier_tarball(cfg, out_path: Path):
    model_dir = _REPO_ROOT / cfg["classifier"]["output_dir"]
    with tarfile.open(out_path, "w:gz") as tar:
        for f in model_dir.iterdir():
            if f.is_file():
                tar.add(f, arcname=f.name)
        tar.add(Path(__file__).parent / "inference_classifier.py", arcname="code/inference.py")


def _make_retriever_tarball(cfg, out_path: Path):
    biencoder_dir = _REPO_ROOT / cfg["retriever"]["output_dir"]
    index_path = _REPO_ROOT / cfg["retriever"]["faiss_index_path"]
    kb_path = _REPO_ROOT / cfg["data"]["processed_dir"] / "kb_articles.jsonl"

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(biencoder_dir, arcname="biencoder")
        tar.add(index_path, arcname="kb.index")
        tar.add(index_path.with_suffix(".ids.json"), arcname="kb.index.ids.json")
        tar.add(kb_path, arcname="kb_articles.jsonl")
        tar.add(Path(__file__).parent / "inference_retriever.py", arcname="code/inference.py")


def deploy(which: str, cfg: dict, role_arn: str, instance_type: str = "ml.t2.medium"):
    session = sagemaker.Session()
    bucket = session.default_bucket()

    tarball_path = _REPO_ROOT / f"cloud/sagemaker/{which}-model.tar.gz"
    if which == "classifier":
        _make_classifier_tarball(cfg, tarball_path)
    elif which == "retriever":
        _make_retriever_tarball(cfg, tarball_path)
    else:
        raise ValueError(f"unknown --which={which!r}, expected 'classifier' or 'retriever'")

    s3_uri = session.upload_data(str(tarball_path), bucket=bucket, key_prefix=f"autotriage/{which}")

    model = PyTorchModel(
        model_data=s3_uri,
        role=role_arn,
        framework_version="2.2",
        py_version="py310",
        entry_point="inference.py",
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=f"autotriage-{which}",
    )
    print(f"Deployed endpoint: {predictor.endpoint_name}")
    print(f"Set {'RETRIEVER_ENDPOINT' if which == 'retriever' else 'CLASSIFIER_ENDPOINT'}"
          f"={predictor.endpoint_name} on the corresponding Lambda.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--which", required=True, choices=["classifier", "retriever"])
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    deploy(args.which, cfg, args.role_arn, args.instance_type)


if __name__ == "__main__":
    main()
