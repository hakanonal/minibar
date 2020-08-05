from keras.metrics import Metric
from keras.utils import metrics_utils
from keras import backend as K
import numpy as np

class OverallPerformance(Metric):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        super(OverallPerformance, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        if top_k is not None and K.backend() != 'tensorflow':
            raise RuntimeError(
                '`top_k` argument for `Overall Performance` metric is currently supported '
                'only with TensorFlow backend.')

        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        denom = (self.true_positives + self.false_positives + self.false_negatives)
        result = K.switch(
            K.greater(denom, 0),
            self.true_positives / denom,
            K.zeros_like(self.true_positives))

        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(metrics_utils.to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.weights])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(OverallPerformance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
