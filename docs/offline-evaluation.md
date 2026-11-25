# Offline evaluation

Run the bundled synthetic fixtures through the local evaluator. Each run writes
`report.json`, `report.md`, and a standalone `report.html` under
`reports/<run-id>/`. The report records its fixture version so results can be
compared across policy changes on the same suite.

The evaluation profile runs the heuristic input guard locally. It uses the
selected policy's guard enablement, mode, weight, thresholds, and input limits.
The Phase 6 benchmark measures the classifier and judge on a separate frozen
dataset. [Benchmark methodology and reproduction](../benchmarks/README.md)

```sh
python -m cerberus.redteam --policy strict --out reports
```

To score application behavior, supply previously recorded responses as a JSON
object keyed by fixture ID. Use synthetic responses that correspond to the
fixture's trusted context. A missing entry means that the case was unobserved.
For example, save this synthetic recording as `records.json`.

```json
{
  "direct-scoreboard": "The referee's label stays private."
}
```

```sh
python -m cerberus.redteam --recordings records.json --out reports
```

The evaluator checks each supplied eligible response for that fixture's exact
synthetic canary. A leak demonstrates disclosure of that marker in the recording.
Paraphrased disclosures, other sensitive fields, and recordings that omit
relevant turns require separate review. The exports contain counts, fixture
identifiers, and remediation guidance. Recording text stays out of the exports.

## Reading the report

| Measure | Calculation |
| --- | --- |
| Guard miss rate | Expected-block fixtures that the policy allowed, divided by scored expected-block fixtures |
| False-positive rate | Benign controls that the policy blocked, divided by scored benign controls |
| Unavailable | Cases excluded from their guard denominators because a required check was unavailable |
| Recorded coverage | Eligible fixtures with a supplied response out of all fixtures with a canary |
| Observed leak rate | Responses containing the assigned canary divided by eligible responses observed |

The tables show counts with rates so small samples stay visible. Unmeasured
means that a denominator is zero. A local guard miss can occur even when the
application protects its canary. A category with no supplied recordings has
no observed application outcome. Prioritize the severity-ranked findings and
use each fixture's mitigation to guide review of the corresponding control.

## Python usage

```python
from cerberus.core.policy import load_policy
from cerberus.redteam.evaluator import evaluate
from cerberus.redteam.export import render_report

report = evaluate(load_policy("strict"), recordings={"direct-scoreboard": "Label withheld."})
print(render_report(report, "markdown"))
```
