from chainer import serializers, optimizers, cuda, training
from chainer import reporter as reporter_module
from chainer.training import extensions
import six

def VOC_mAP():
    @training.make_extension(trigger=(1, 'epoch'))
    def _VOC_mAP(trainer):
        print("hoge")
    return _VOC_mAP

class MyEvaluator(extensions.Evaluator):
    default_name="myval"
    def evaluate(self):
        #target = self._targets['main']

        summary = reporter_module.DictSummary()
        for name, target in six.iteritems(self._targets):
            iterator = self._iterators['main']
            #target = self._targets['main']
            eval_func = self.eval_func or target

            if self.eval_hook:
                self.eval_hook(self)

            if hasattr(iterator, 'reset'):
                iterator.reset()
                it = iterator
            else:
                it = copy.copy(iterator)

            #summary = reporter_module.DictSummary()
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    in_arrays = self.converter(batch, self.device)
                    with function.no_backprop_mode():
                        if isinstance(in_arrays, tuple):
                            eval_func(*in_arrays)
                        elif isinstance(in_arrays, dict):
                            eval_func(**in_arrays)
                        else:
                            eval_func(in_arrays)

                summary.add(observation)
        return summary.compute_mean()

