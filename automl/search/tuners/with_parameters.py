import inspect
import ray
from ray.tune.utils.trainable import detect_checkpoint_function
from ray.tune.registry import _ParameterRegistry


def with_parameters(trainable, **kwargs):
    """Wrapper for trainables to pass arbitrary large data objects.

    This wrapper function will store all passed parameters in the Ray
    object store and retrieve them when calling the function. It can thus
    be used to pass arbitrary data, even datasets, to Tune trainables.

    This can also be used as an alternative to ``functools.partial`` to pass
    default arguments to trainables.

    When used with the function API, the trainable function is called with
    the passed parameters as keyword arguments. When used with the class API,
    the ``Trainable.setup()`` method is called with the respective kwargs.

    Args:
        trainable: Trainable to wrap.
        **kwargs: parameters to store in object store.

    Function API example:

    .. code-block:: python

        from ray import tune

        def train(config, data=None):
            for sample in data:
                loss = update_model(sample)
                tune.report(loss=loss)

        data = HugeDataset(download=True)

        tune.run(
            tune.with_parameters(train, data=data),
            # ...
        )

    Class API example:

    .. code-block:: python

        from ray import tune

        class MyTrainable(tune.Trainable):
            def setup(self, config, data=None):
                self.data = data
                self.iter = iter(self.data)
                self.next_sample = next(self.iter)

            def step(self):
                loss = update_model(self.next_sample)
                try:
                    self.next_sample = next(self.iter)
                except StopIteration:
                    return {"loss": loss, done: True}
                return {"loss": loss}

        data = HugeDataset(download=True)

        tune.run(
            tune.with_parameters(MyTrainable, data=data),
            # ...
        )

    """
    from ray.tune.trainable import Trainable

    if not callable(trainable) or (
        inspect.isclass(trainable) and not issubclass(trainable, Trainable)
    ):
        raise ValueError(
            f"`tune.with_parameters() only works with function trainables "
            f"or classes that inherit from `tune.Trainable()`. Got type: "
            f"{type(trainable)}."
        )

    parameter_registry = _ParameterRegistry()
    ray.worker._post_init_hooks.append(parameter_registry.flush)

    # Objects are moved into the object store
    prefix = f"{str(trainable)}_"
    for k, v in kwargs.items():
        parameter_registry.put(prefix + k, v)

    trainable_name = getattr(trainable, "__name__", "tune_with_parameters")

    if inspect.isclass(trainable):
        # Class trainable
        keys = list(kwargs.keys())

        class _Inner(trainable):
            def setup(self, config):
                setup_kwargs = {}
                for k in keys:
                    setup_kwargs[k] = parameter_registry.get(prefix + k)
                # Only change in AutoPipeline, different setup call with refs
                super(_Inner, self).setup(
                    config, refs=(parameter_registry.references,), **setup_kwargs
                )

        _Inner.__name__ = trainable_name
        return _Inner
    else:
        # Function trainable
        use_checkpoint = detect_checkpoint_function(trainable, partial=True)
        keys = list(kwargs.keys())

        def inner(config, checkpoint_dir=None):
            fn_kwargs = {}
            if use_checkpoint:
                default = checkpoint_dir
                sig = inspect.signature(trainable)
                if "checkpoint_dir" in sig.parameters:
                    default = sig.parameters["checkpoint_dir"].default or default
                fn_kwargs["checkpoint_dir"] = default

            for k in keys:
                fn_kwargs[k] = parameter_registry.get(prefix + k)
            trainable(config, **fn_kwargs)

        inner.__name__ = trainable_name

        # Use correct function signature if no `checkpoint_dir` parameter
        # is set
        if not use_checkpoint:

            def _inner(config):
                inner(config, checkpoint_dir=None)

            _inner.__name__ = trainable_name

            if hasattr(trainable, "__mixins__"):
                _inner.__mixins__ = trainable.__mixins__
            return _inner

        if hasattr(trainable, "__mixins__"):
            inner.__mixins__ = trainable.__mixins__

        return inner
