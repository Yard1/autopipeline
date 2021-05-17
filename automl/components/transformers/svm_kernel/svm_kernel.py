from ..transformer import Transformer
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.linear_model.svm.svm import SVM


class SVMKernel(Transformer):
    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None or isinstance(config.estimator, SVM)
        )
