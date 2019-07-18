from chainer.training import extension
import numpy as np

class PolynomialShift(extension.Extension):
    def __init__(self, attr, batchsize, len_dataset, power=0.9, stop_trigger=[90000, "iteration"]):
        self._attr = attr
        self._power = power
        self._init = None
        self._t = 0
        self._last_value = 0

        if stop_trigger[1] == 'iteration':
            self._maxiter = stop_trigger[0]
        elif stop_trigger[1] == 'epoch':
            n_iter_per_epoch = len_dataset / float(batchsize)
            self._maxiter = float(stop_trigger[0] * n_iter_per_epoch)

    def initialize(self, trainer):
        optimizer = trainer.updater.get_optimizer('main')
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

    def __call__(self, trainer):
        self._t += 1

        optimizer = trainer.updater.get_optimizer('main')
        value = self._init * ((1 - (self._t / self._maxiter)) ** self._power)
        setattr(optimizer, self._attr, value)
        self._last_value = value

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)

class CosineDecayWithWarmup(extension.Extension):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from _init rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Args:
      self._t: int64 (scalar) tensor representing global step.
      learning_rate_base: base learning rate.(the maximum one)
      hold_base_rate_steps: Optional number of steps to hold base learning rate
        before decaying.
    Returns:
      a (scalar) float tensor representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than _maxiter.
    """
    def __init__(self, batchsize, len_dataset, stop_trigger=[90000, "iteration"],
                             learning_rate_base=0.2, warmup_steps=0, hold_base_rate_steps=0, attr='lr'):
        self._attr = attr
        self._init = None
        self._t = 0
        self._last_value = 0
        self._learning_rate_base = learning_rate_base
        self._warmup_steps = warmup_steps
        self._hold_base_rate_steps = hold_base_rate_steps

        if stop_trigger[1] == 'iteration':
            self._maxiter = stop_trigger[0]
        elif stop_trigger[1] == 'epoch':
            n_iter_per_epoch = len_dataset / float(batchsize)
            self._maxiter = float(stop_trigger[0] * n_iter_per_epoch)

    def initialize(self, trainer):
        optimizer = trainer.updater.get_optimizer('main')
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

    def __call__(self, trainer):
        self._t += 1

        optimizer = trainer.updater.get_optimizer('main')

        if self._maxiter < self._warmup_steps:
            raise ValueError('self._maxiter must be larger or equal to warmup_steps.')
        learning_rate = 0.5 * self._learning_rate_base * (1 + np.cos( np.pi *
            (self._t - self._warmup_steps - self._hold_base_rate_steps)
            / float(self._maxiter - self._warmup_steps - self._hold_base_rate_steps)
            ))
        if self._hold_base_rate_steps > 0:
            if self._t > self._warmup_steps + self._hold_base_rate_steps:
                learning_rate = learning_rate
            else:
                learning_rate = self._learning_rate_base
        if self._warmup_steps > 0:
            if self._learning_rate_base < self._init:
                raise ValueError('_learning_rate_base must be larger or equal to warmup_learning_rate.')
            slope = (self._learning_rate_base - self._init) / self._warmup_steps
            warmup_rate = slope * self._t + self._init
            if self._t < self._warmup_steps:
                learning_rate = warmup_rate
            else:
                learning_rate =  learning_rate
        value=0.0 if self._t > self._maxiter else learning_rate

        setattr(optimizer, self._attr, value)
        self._last_value = value

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)

