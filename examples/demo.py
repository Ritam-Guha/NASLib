import logging
import sys

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
)

from naslib.search_spaces import (
    NasBench301SearchSpace,
    SimpleCellSearchSpace,
    NasBench201SearchSpace,
    HierarchicalSearchSpace,
    NasBench101SearchSpace
)

# from naslib.search_spaces.nasbench101 import graph
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api

search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "nasbench301": NasBench301SearchSpace(),
    "simplecell": SimpleCellSearchSpace(),
    "hierarchical": HierarchicalSearchSpace(),
}

def main():
    search_space_name = "nasbench101"
    dataset_name = "cifar10"
    optimizer_name = "re"

    # Read args and config, setup logger
    config = utils.get_config_from_args()
    utils.set_seed(config.seed)

    logger = setup_logger(config.save + "/log.log")
    # logger.setLevel(logging.INFO)   # default DEBUG is very verbose

    utils.log_args(config)

    supported_optimizers = {
        "darts": DARTSOptimizer(),
        "gdas": GDASOptimizer(),
        "drnas": DrNASOptimizer(),
        "rs": RandomSearch(config),
        "re": RegularizedEvolution(config),
        "ls": LocalSearch(config),
        "bananas": Bananas(config),
        "bp": BasePredictor(config),
    }

    # Changing the search space is one line of code
    search_space = search_spaces[search_space_name]
    dataset_api = get_dataset_api(search_space_name, dataset_name)

    # Changing the optimizer is one line of code
    # optimizer = supported_optimizers[config.optimizer]
    optimizer = supported_optimizers[optimizer_name]
    optimizer.adapt_search_space(search_space, search_space_name, dataset_api=dataset_api)

    # Start the search and evaluation
    trainer = Trainer(optimizer, config)

    if not config.eval_only:
        checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
        trainer.search(resume_from=checkpoint)

    checkpoint = utils.get_last_checkpoint(config, search=False) if config.resume else ""
    trainer.evaluate(resume_from=checkpoint, dataset_api=dataset_api)


if __name__ == "__main__":
    main()
