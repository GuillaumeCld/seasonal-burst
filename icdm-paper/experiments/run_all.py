#!/usr/bin/env python3
"""Script to run all ICDM paper experiments."""

import argparse
import logging
from pathlib import Path
import time

from utils.config import load_config, setup_output_dirs

# Import experiment modules
import exhaustive_algo_scaling
import numerical
import skipping_tradeoff
import grid_skipping_tradeoff
import sum_computation
import online_scaling

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_experiment(name: str, config_path: Path) -> None:
    """
    Run a single experiment with its configuration.
    
    Parameters:
        name: Name of the experiment
        config_path: Path to the configuration file
    """
    logger.info(f"Starting experiment: {name}")
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(config_path)
        setup_output_dirs(config)
        
        # Run appropriate experiment
        if name == "exhaustive_algo_scaling":
            exhaustive_algo_scaling.run(config)
        elif name == "numerical":
            numerical.run(config)
        elif name == "skipping_tradeoff":
            skipping_tradeoff.run(config)
        elif name == "grid_skipping_tradeoff":
            grid_skipping_tradeoff.run(config)
        elif name == "sum_computation":
            sum_computation.run(config)
        elif name == "online_scaling":
            online_scaling.run(config)
        else:
            logger.error(f"Unknown experiment: {name}")
            return
        
        duration = time.time() - start_time
        logger.info(f"Finished experiment {name} in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in experiment {name}: {str(e)}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Run ICDM paper experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["all", "exhaustive_algo_scaling", "numerical", "skipping_tradeoff", 
                "grid_skipping_tradeoff", "sum_computation", "online_scaling"],
        default=["all"],
        help="Experiments to run"
    )
    args = parser.parse_args()
    
    # Define all experiments and their config files
    experiments = {
        "exhaustive_algo_scaling": "configs/exhaustive_algo_scaling.ini",
        "numerical": "configs/numerical.ini",
        "skipping_tradeoff": "configs/skipping_tradeoff.ini",
        "grid_skipping_tradeoff": "configs/grid_skipping_tradeoff.ini",
        "sum_computation": "configs/sum_computation.ini",
        "online_scaling": "configs/online_scaling.ini"
    }
    
    # Determine which experiments to run
    if "all" in args.experiments:
        selected_experiments = experiments
    else:
        selected_experiments = {name: path for name, path in experiments.items() 
                              if name in args.experiments}
    
    # Run selected experiments
    for name, config_path in selected_experiments.items():
        config_path = Path(__file__).parent / config_path
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            continue
        
        run_experiment(name, config_path)

if __name__ == "__main__":
    main() 