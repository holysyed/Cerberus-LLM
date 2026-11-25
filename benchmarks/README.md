# Cerberus evaluation v1

This benchmark measures case-level detection and single-request latency at the packaged strict policy's operating point. The input ablation evaluates all three configurations on the same 128 cases in the same order. The output evaluation uses a separate set of 68 labeled replies.

## Dataset

| Source | Task | Positive | Benign | Sampling |
|---|---|---:|---:|---|
| deepset/prompt-injections | Input injection | 24 | 24 | Stratified sample of upstream test split |
| xTRam1/safe-guard-prompt-injection | Input injection | 24 | 24 | Stratified sample of upstream test split |
| Cerberus | Input injection | 16 | 16 | Fixed authored messages, document snippets and tool outputs |
| Cerberus | PII or secret present | 10 | 10 | Fixed prose and structured replies |
| Cerberus | Unsafe reply, refusal bypass, or prompt disclosure | 8 | 8 | Fixed output and trusted prompt where relevant |
| Cerberus | Unsupported claim | 8 | 8 | Fixed claim paired with its evidence |
| Cerberus | Schema mismatch | 8 | 8 | Fixed JSON contract, with valid and invalid candidate replies |

The public subsets retain their publishers' binary labels. For each pinned test file, the loader hashes case-folded, whitespace-normalized text, sorts by hash, and takes the first 24 distinct rows of each class. It processes deepset first and deduplicates selected texts across both corpora. The manifest records each upstream row index, label and SHA-256 of its original text. Downloaded Parquet files stay in the local cache. [Public sample manifest](public.v1.json) · [Split revisions and checksums](data.py)

The first upstream dataset is published by deepset. At the pinned revision its card lists Apache-2.0 at the top level and CC-BY-4.0 in dataset metadata. The second dataset is published by xTRam1. Refer to the upstream cards for provenance and terms. The classifier's model card identifies deepset as training data. The deepset slice therefore carries a risk of training overlap. Source-level confusion matrices in the result artifact let readers inspect the other slices independently. [deepset dataset card](https://huggingface.co/datasets/deepset/prompt-injections/blob/4f61ecb038e9c3fb77e21034b22511b523772cdd/README.md) · [Safe-Guard dataset card](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection/blob/a3a877d608f37b7d20d9945671902df895ecdb46/README.md) · [Classifier model card](https://huggingface.co/fmops/distilbert-prompt-injection)

Cerberus's authored labels include benign discussion of security terms, multilingual inputs, indirect instructions, unmarked person names, paraphrases, and contradictions. Output positives indicate that the corresponding guard should find an issue. For example, a claim contradicting its source is positive even if it reuses almost all the source's words. These are small, balanced diagnostic sets. Production prevalence, languages, and domain vocabulary can change precision and false-positive rates substantially. [Authored examples and labels](synthetic.v1.jsonl)

## Reproduce a run

Run these commands from the repository root in the active virtual environment. The baseline evaluates the heuristics and output detectors.

```sh
python -m pip install -e ".[benchmark]"
python -m benchmarks.data --download
python -m benchmarks.run
```

For the full comparison, install the optional local model dependencies and make the judge model available in a running Ollama server on loopback.

```sh
python -m pip install -e ".[benchmark,ml,ner]"
python -m spacy download en_core_web_sm
ollama pull qwen2.5:0.5b
python -m benchmarks.run --full --ner --threads 2 --output .cache/phase6/full.json
```

The classifier loads the pinned FMOps checkpoint. To use an existing copy of that checkpoint, add `--classifier PATH_TO_SNAPSHOT`. The result records local file fingerprints when a directory is supplied. The judge uses Ollama's structured JSON output, temperature zero, seed 42 and a 160-token output budget. The result records the installed Ollama version and the local model digest. The package's standard `LLMJudge` validates each verdict. All model calls during the published run used local inference.

## Reading the numbers

- **Decision unit.** One request is one observation regardless of the number of documents or tool outputs. The ablation uses the actual input pipeline with configured weights, gates, threshold and fail-closed behavior. Optional layers score every source. Standalone classifier and judge rows apply the raw 0.5 confidence gate to the largest score across sources.
- **Confusion matrix.** TP and FN count labeled positives. FP and TN count labeled benign cases. Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, F1 is `2TP/(2TP+FP+FN)`, and FPR is `FP/(FP+TN)`. A zero denominator is unmeasured. Each result includes the number of unavailable observations. Those cases contribute to coverage and are excluded from accuracy and successful-call timing, even if the fail-closed policy blocked them.
- **Output findings.** A PII finding counts when Cerberus sanitizes a reply and releases it. The PII score measures detection per reply. Character-span coverage and redaction completeness need separate evaluation. The strict output row uses all 52 PII, safety and schema cases. Grounding has its own 16-case row with evidence checking enabled. The `PII + spaCy` row uses the same 20 cases as regex PII.
- **Timing.** Each profile runs sequentially at concurrency one after warm-up. Percentiles use linear interpolation over one observation per scored case. Input stack timings include pipeline execution. Standalone backend times are measured around actual calls within the stack and summed across sources. Output detector rows time their individual checks. The JSON also reports combined input and output guard latency on admitted requests using a fixed benign reply. Application generation, HTTP gateway handling, SQLite persistence, schema retries, and model initialization are outside that timing.
- **Input length.** A separate size check runs the heuristic pipeline 40 times after one warm-up on a 32,768-character source and again on a 65,536-character request split between the message and one document. Both use repeated benign timetable prose. The latency budget comparison applies to those specific workloads and this machine.
- **Comparison limits.** The order is fixed with seed 615. Runtime caches, hardware, prompt length, and model versions affect latency. The corpus has few examples per language and per output category. Judge confidence is self-reported. A difference of one case can move an eight-positive output task's recall by 12.5 percentage points.

The reference run includes aggregate and per-source metrics, the policy, runtime details, and dataset fingerprints. [Machine-readable results](results.v1.json)
