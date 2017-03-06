Chainer direct forward
====

This is experimental code to call `forward` method of `chainer.function` directly.

# Requirements

* [Chainer](http://chainer.org/) 1.21.0

# Usage

```
$ python mnist.py [-g gpu_device_id] [-b batch_size] [-i iteration]
```

Options:
* `gpu_device_id`: GPU device ID, negative value indicates CPU (default: -1)
* `batch_size`: Mini batch size (default: 100)
* `iteration`: Number of iteration (default: 100)
