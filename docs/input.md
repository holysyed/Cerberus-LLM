# Input detection

Pass the user message as the first argument to the input pipeline. Put the exact
retrieved passages and tool results that the model will see in the context.
Scan again when a tool returns new data. Trusted instructions belong in the
application's system message.

```python
from cerberus.core import GuardContext, Pipeline, load_policy
from cerberus.guards.injection import InputGuard

policy = load_policy("strict")
inputs = Pipeline([InputGuard(policy)], policy)
decision = inputs.run(
    "Summarize the office hours.",
    GuardContext(documents=("The office opens at 09:00.",)),
)
if decision.blocked:
    raise ValueError(f"Input blocked ({decision.request_id})")
```

Each enabled detector scores each source independently. Within a source,
Cerberus combines the detectors using `1 − ∏(1 − weight × score)`. The highest
source score determines the input decision. These scores drive policy actions
and have no calibrated probabilistic interpretation.

The default rule weight is 1.0. The optional classifier and judge weights are
0.9 and 0.8. Each optional detector has its own confidence threshold before
aggregation. Set these fields in the YAML `input` section and set the decision
threshold with `guards.input.threshold`. The strict preset uses the heuristic
layer out of the box. Enable model-backed layers explicitly when their runtime
and data handling fit the application.

The DistilBERT adapter loads its tokenizer and weights on first use. Its
training data includes deepset/prompt-injections. Scores on that public corpus
can overstate performance on independent application traffic. Evaluate
multilingual input on local examples before relying on its scores.
[Classifier model card](https://huggingface.co/fmops/distilbert-prompt-injection)
