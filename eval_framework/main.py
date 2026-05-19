# main.py
import argparse
from importlib import import_module
from typing import Any, Callable
import model

# Mapping: task ID -> evaluate_dataset function in the corresponding eval module
_EVAL_ENTRYPOINTS = {
    "1": ("eval_1", "evaluate_dataset"),
    "2": ("eval_2", "evaluate_dataset"),
    "3": ("eval_3", "evaluate_dataset"),
    "4": ("eval_4", "evaluate_dataset"),
    "5": ("eval_5", "evaluate_dataset"),
}

# Question variant alias -> CSV column name
_QUESTION_FIELD = {
    "std":  "Question_std",
    "cot":  "Question_cot",
    "chem": "Question_chem",
}

def _load_model(spec: str, kwargs_json: str | None) -> Any:
    """
    Support loading a model in the form 'pkg.module:ClassName' and pass init
    arguments via a JSON string. The model must provide .call(query: str) -> str.
    If --model is not provided, use a simple echo model (returns the query).
    """
    if not spec:
        class EchoModel:
            def call(self, q: str) -> str:
                # Placeholder: no <SMILES> tags; for pipeline sanity checks
                return q
        return EchoModel()

    if ":" not in spec:
        raise ValueError("--model must be in the form 'module:Class'.")
    mod_name, cls_name = spec.split(":", 1)
    mod = import_module(mod_name)
    cls = getattr(mod, cls_name)
    import json as _json
    kwargs = {} if not kwargs_json else _json.loads(kwargs_json)
    return cls(**kwargs)

def main():
    parser = argparse.ArgumentParser(description="Evaluation framework for chemical reaction tasks")
    parser.add_argument("--task", choices=["1", "2", "3", "4", "5"], required=True,
                        help="Select evaluation task: 1/2/3/4/5 correspond to eval_1/2/3/4/5")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--variant", choices=["std", "cot", "chem"], default="std",
                        help="Question variant (maps to Question_std / Question_cot / Question_chem)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of repeated evaluations per sample")
    parser.add_argument("--model", default="",
                        help="Model class in the form 'package.module:ClassName'; must implement .call(query)->str")
    parser.add_argument("--model-kwargs", default="",
                        help="Model initialization arguments (JSON string)")
    args = parser.parse_args()

    module_name, func_name = _EVAL_ENTRYPOINTS[args.task]
    eval_mod = import_module(module_name)
    eval_fn: Callable[..., None] = getattr(eval_mod, func_name)

    model = _load_model(args.model, args.model_kwargs)
    qfield = _QUESTION_FIELD[args.variant]

    # Run the selected task
    eval_fn(
        input_path=args.input,
        output_path=args.output,
        model=model,
        num_runs=args.num_runs,
        question_field=qfield,
        variant=args.variant
    )

if __name__ == "__main__":
    main()
