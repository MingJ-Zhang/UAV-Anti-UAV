# Contributing

Issues and pull requests are welcome.

For a bug report, include the command, configuration, Python/PyTorch versions, traceback, and a small reproducible example. For benchmark changes, include the dataset split, checkpoint, language input, sequence count, and evaluation command.

Before submitting a change:

```bash
python tools/sla_v3_unit_test.py
python -m unittest discover -s tests -v
python -m compileall -q lib tracking tools tests
```

Keep changes focused and explain their effect on model behavior. Update the corresponding guide when changing a command or configuration. CPU checks run automatically through GitHub Actions.
