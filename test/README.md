- `modules` for unit test (UT): test the main modules including dataset, transform, loader, model, loss, optimizer, and scheduler.  

To test all modules: 
```
# run the command line in the root dir of the git repo
pytest test/modules/*.py
```

- `tasks` for system test (ST): test the training and validation pipeline. 

To test the training process (in graph mode and pynative+ms_function mode) and the validation process, run
```
pytest test/task/test_train_val.py
```


