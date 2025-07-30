# Traveling Salesman Problem (TSP) Optimization Using Genetic Algorithms

## Overview

This project implements genetic algorithms (GA) and related advanced methods to solve various Traveling Salesman Problem (TSP) scenarios, including classical TSP, large-scale optimization, multi-objective optimization, multi-tasking optimization, and bi-level optimization with depot selection.

## Project Structure

```
Project/
├── data/
│   ├── TSP.csv             # Coordinates of 100 original customers
│   └── Depot.csv           # Depot locations for bi-level optimization
│
├── src/
│   ├── __init__.py         # Package initialization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tsp_problem.py          # Classical TSP model
│   │   ├── large_scale.py          # Large-scale TSP model with clustering
│   │   ├── multiobjective.py       # Multi-objective TSP model
│   │   ├── multitasking.py         # Multi-tasking TSP model
│   │   └── bilevel.py              # Bi-level TSP model with depot selection
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Functions to load data
│   │   ├── visualization.py        # Visualization tools
│   │   └── metrics.py              # Performance metrics calculations
│   │
│   └── tasks/
│       ├── __init__.py
│       ├── task1_tsp.py            # Classical TSP solution
│       ├── task2_large_scale.py    # Large-scale optimization solution
│       ├── task3_multiobjective.py # Multi-objective optimization solution
│       ├── task4_multitasking.py   # Multi-tasking optimization solution (MFEA)
│       └── task5_bilevel.py        # Bi-level optimization with depot selection
│
├── experiments/
│   ├── task1_results.ipynb         # Notebook for classical TSP experiments
│   ├── task2_results.ipynb         # Notebook for large-scale optimization experiments
│   ├── task3_results.ipynb         # Notebook for multi-objective experiments
│   ├── task4_results.ipynb         # Notebook for multitasking experiments
│   └── task5_results.ipynb         # Notebook for bi-level optimization experiments
│
├── docs/
│   ├── Result Image/               # Experiments result image
│   └── Report.pdf                  # Project report
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Installation and Setup

### Environment

This project is tested and recommended to run with:

- **Python**: 3.9
- **NumPy**: 1.23
- **Geatpy**: 2.7.0
- **matplotlib**==3.10.1
- **pandas**==2.2.3
- **scikit_learn**==1.6.1

### Installation

Use the following commands to set up the project environment:

```bash
# Clone the project repository
git clone <>
cd Project

# Create and activate a virtual environment (recommended)
conda create -n tsp_env python=3.9
conda activate tsp_env

# Install dependencies
pip install -r requirements.txt
```

## How to Run the Experiments

### Using Jupyter Notebooks (Recommended)

Navigate to the `experiments` directory and run:

```bash
jupyter notebook
```

Open the relevant notebook (e.g., `task1_results.ipynb`) to run experiments interactively and visualize results.

### Direct Python Scripts

Each task has a dedicated script located in `src/tasks/`. You can run them individually, e.g.:

```bash
python src/tasks/task1_tsp.py
```

Adjust parameters directly in the scripts as needed.

## Explanation of Code Structure

- **Models** (`src/models/`): Defines different TSP problem formulations (classical, large-scale, multi-objective, etc.).
- **Utilities** (`src/utils/`): Provides common functions for data loading, visualization, and performance metrics.
- **Tasks** (`src/tasks/`): Implements the genetic algorithms and specialized methods for each TSP variant, leveraging models and utilities.
- **Experiments** (`experiments/*.ipynb`): Jupyter Notebooks for interactive analysis, visualization, sensitivity studies, and documenting results.

## Project Tasks Overview

1. **Classical TSP**: Solve a standard 100-customer TSP using genetic algorithms.
2. **Large-scale TSP**: Cluster customers to manage large-scale TSP efficiently, then optimize.
3. **Multi-objective TSP**: Optimize simultaneously for distance minimization and profit maximization, comparing Pareto methods and weighted objectives.
4. **Multi-tasking TSP**: Apply Multifactorial Evolutionary Algorithm (MFEA) to optimize multiple TSP problems concurrently with knowledge transfer.
5. **Bi-level TSP with Depot Selection**: Optimize depot selection at an upper decision level and TSP route at a lower decision level, using hierarchical evolutionary methods.

## Results and Analysis

Each notebook under `experiments/`:

- Runs multiple experiments with different parameters.
- Conducts parameter sensitivity analysis (population size, mutation probability, crossover probability).
- Visualizes routes, Pareto fronts, clustering results, and other analytical results clearly.

## Project Documentation

The `docs/` directory includes a detailed report (`Report.pdf`) covering:

- Problem definitions
- Algorithm designs and parameters
- Comprehensive results, analysis, and discussions
- Conclusions and future work recommendations

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to Prof. TAN Kay Chen and tutors Haokai Hong and Lulu Cao for their guidance.

## Contact

For questions or suggestions, please contact Robbie Fenggui Rao (robbie.rao@connect.polyu.hk, 24037513R).