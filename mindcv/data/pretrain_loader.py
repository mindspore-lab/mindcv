"""
Create dataloader for pre-training
"""
import inspect

__all__ = ["create_loader_pretrain"]


def create_loader_pretrain(
    dataset, batch_size, drop_remainder=False, transform=None, num_parallel_workers=None, python_multiprocessing=False
):
    if transform is None:
        raise ValueError("tranform should not be None for pre-training.")

    # notes: mindspore-2.0 delete parameter 'column_order'
    sig = inspect.signature(dataset.map)
    pass_column_order = False if "kwargs" in sig.parameters else True

    dataset = dataset.map(
        operations=transform,
        input_columns="image",
        output_columns=transform.output_columns,
        column_order=transform.output_columns if pass_column_order else None,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )
    if not pass_column_order:
        dataset = dataset.project(transform.output_columns)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    return dataset
