# Notice and third-party acknowledgements

Quec-fr-CA-llm-training is a governance-first pipeline for Québec French language-model data validation, curation, evaluation, and training-pack preparation.

## No endorsement

References to external datasets, corpora, benchmarks, models, organizations, institutions, or software projects do not imply affiliation, endorsement, permission, or approval.

## Software dependencies

The pipeline currently depends on open-source Python tooling including Pydantic, PyYAML, Typer, and Rich. Development and validation use tools such as pytest and Ruff.

Optional local training workflows may use Unsloth, Hugging Face Transformers, Hugging Face Datasets, TRL, Accelerate, PEFT, bitsandbytes, safetensors, tqdm, SentencePiece, and protobuf.

## Models referenced

Model references may include Qwen, Dolphin3, GGUF runtime references, and adapter/LoRA artifacts. A model reference does not imply that weights are redistributed by this repository or that downstream fine-tuned models are cleared for release.

## Corpus and benchmark policy

Corpus, dataset, benchmark, and institutional references are tracked as training-approved, evaluation-only, holdout-only, catalog-only, permission-required, or excluded. No external resource should be treated as production/commercial training material unless its license and permission status explicitly allow that use.

## Generated artifacts

Generated training packs, reports, diagnostics, micro-packs, and model artifacts may contain derived, synthetic, or source-dependent material. They require separate review before release.

## Contact-before-release rule

For any public dataset/model release, publication, or commercial training run, maintainers should review `THIRD_PARTY_SOURCES.yaml`, source manifests, dataset cards, model cards, and permission manifests first.