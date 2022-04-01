import argparse
from dooly import DoolyConverter
from dooly.converters import is_available_pororo


if is_available_pororo():
    from pororo.pororo import SUPPORTED_TASKS
else:
    raise ModuleNotFoundError("Please install pororo with: `pip install pororo`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Convert Pororo module to Dooly.")
    parser.add_argument("--task", type=str, help="task name")
    parser.add_argument("--save_path", type=str, help="path to save.")
    args = parser.parse_args()

    available_tasks = DoolyConverter.subclasses.keys()
    assert args.task not in available_tasks

    pororo_factory = SUPPORTED_TASKS[args.task]
    for lang, n_model in pororo_factory.get_available_models():
        converter = DoolyConverter.load(
            task=args.task, lang=lang, n_model=n_model, save_path=args.save_path
        )
        converter.convert()
