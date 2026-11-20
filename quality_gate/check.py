"""Replay the retrospective benign-output KL gate on aggregate measurements."""

import argparse
import json
import math
from pathlib import Path


THRESHOLD = 0.1
EXPECTED_LAYERS = (16, 20, 22, 23)
DEFAULT_INPUT = Path(__file__).with_name("archived_kl.json")


def evaluate(document):
    if not isinstance(document, dict) or set(document) != {"measurements"}:
        raise ValueError("Expected a document with one measurements field")
    measurements = document["measurements"]
    if not isinstance(measurements, list):
        raise ValueError("Expected a list of measurements")

    values = {}
    for measurement in measurements:
        if not isinstance(measurement, dict) or set(measurement) != {"layer", "kl"}:
            raise ValueError("Each measurement must contain layer and kl")
        layer, kl = measurement["layer"], measurement["kl"]
        if type(layer) is not int or layer not in EXPECTED_LAYERS:
            raise ValueError(f"Unexpected layer {layer!r}")
        if layer in values:
            raise ValueError(f"Duplicate layer {layer}")
        if type(kl) not in (int, float) or kl < 0:
            raise ValueError(f"Invalid KL for layer {layer}")
        if isinstance(kl, float) and not math.isfinite(kl):
            raise ValueError(f"Non-finite KL for layer {layer}")
        values[layer] = kl

    missing = set(EXPECTED_LAYERS) - values.keys()
    if missing:
        raise ValueError(f"Missing KL for layers {sorted(missing)}")
    per_layer = [
        {"layer": layer, "kl": values[layer], "passes": values[layer] < THRESHOLD}
        for layer in EXPECTED_LAYERS
    ]
    eligible = [row["layer"] for row in per_layer if row["passes"]]
    return {
        "criterion": "kl < 0.1",
        "criterion_timing": "retrospective",
        "per_layer": per_layer,
        "eligible_layers": eligible,
        "mechanism": "unresolved",
    }


def unique_fields(pairs):
    document = {}
    for field, value in pairs:
        if field in document:
            raise ValueError(f"Duplicate JSON field {field!r}")
        document[field] = value
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text(), object_pairs_hook=unique_fields)
        report = evaluate(document)
    except (OSError, ValueError) as error:
        parser.exit(2, f"KL gate stopped: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
