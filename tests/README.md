- `modules` for unit test (UT): test the main modules including dataset, transform, loader, model, loss, optimizer, and scheduler.

To test all modules:
```shell
pytest tests/modules/*.py
```

- `tasks` for system test (ST): test the training and validation pipeline.

To test the training process (in graph mode and pynative+mindspore.jit mode) and the validation process, run
```shell
pytest tests/tasks/test_train_val_imagenet_subset.py
```
