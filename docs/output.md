# Output handling

`Guardrails.build(policy)` constructs the output chain in this order.

1. **Safety.** Check the original reply for toxicity, unsafe-content signals,
   refusal-bypass language, and overlap with the trusted system prompt.
2. **Grounding.** If enabled, compare claims with the supplied documents and
   tool outputs. Lexical overlap can miss contradictions and penalize valid
   paraphrases. Treat the threshold as a retrieval-domain tuning parameter.
3. **PII.** Detect sensitive spans and apply the configured replacement. The
   default is redaction. Masking retains some information. Hash mode uses keyed
   HMAC so repeated values can be matched within one key's lifetime. Use an
   application-owned random key of at least 32 bytes.
4. **Schema.** Validate the final candidate against the application's JSON
   Schema or Pydantic model. Applications supply schemas through trusted
   configuration. Keep schema definitions under the same control as code.

```python
from pydantic import BaseModel

from cerberus.core import load_policy
from cerberus.guards.factory import Guardrails

class Reply(BaseModel):
    summary: str

rails = Guardrails.build(load_policy(), output_model=Reply)
decision = rails.outputs.run('{"summary":"Write to help@example.invalid"}')
if decision.blocked:
    raise ValueError(f"Output blocked ({decision.request_id})")
reply = Reply.model_validate_json(decision.text)
```

The gateway can regenerate a schema-invalid reply a bounded number of times.
Every replacement goes through the entire output chain. The provider receives
the original guarded request and a fixed correction instruction. The rejected
reply stays local.

| Policy setting | Default | Effect |
|---|---|---|
| `mode` | `sanitize` in the packaged strict profile | Release a guard's replacement when one exists. Block other violations. |
| `critical_threshold` | `0.95` | Block immediately at or above this score in every mode. |
| `fail_mode` | `closed` | Block when a required check cannot run. With `open`, remaining checks continue. |
| `output.pii_mode` | `redact` | Use `mask` for limited display or `hash` for deterministic keyed pseudonyms. |
| `output.max_retries` | `1` | Permit up to three new generations for a schema-only failure. |
| `output.grounding_enabled` | `false` | Include lexical grounding when an application supplies evidence. |

For name recognition in free prose, install the `ner` extra and the spaCy
model selected by `output.ner_model`, then set `output.ner_enabled: true`.
The lightweight default recognizes explicitly labeled names. A valid card
checksum or a matching Indian identifier pattern indicates a candidate value.
It does not establish the owner's identity.
