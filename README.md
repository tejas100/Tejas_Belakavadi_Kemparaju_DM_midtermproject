# Tejas_Belakavadi_Kemparaju_DM_midtermproject


## Requirements

- Python 3.x
- pandas
- mlxtend
- itertools
- collections

## Installation

1. Clone the repository: git clone
2. Install required packages: pip install pandas mlxtend
   
## Usage
1. Run the main script: Belakavadi_Kemparaju_Tejas.ipynb in jupyteer notebook or Belakavadi_Kemparaju_Tejas.py
2. Follow the prompts to:
- Select a store for analysis
- Enter minimum support and confidence thresholds
3. View the results, including:
- Execution time for each algorithm
- Comparison of frequent itemsets and rules generated
- Detailed association rules for each method
- Item count and support analysis

## Data

The project includes transaction data for five stores:
- Amazon
- Best Buy
- Nike
- Walmart
- Target

Data is stored in CSV format, with each file containing at least 20 transactions.

## Algorithm Details

1. **Brute Force**: Custom implementation that exhaustively checks all possible itemsets.
2. **Apriori**: Efficient algorithm for frequent itemset generation, implemented using mlxtend.
3. **FP-Growth**: Tree-based approach for mining frequent patterns, also implemented using mlxtend.

## Results

The program outputs:
- Execution time for each algorithm
- Number of frequent itemsets and rules generated by each method
- Detailed association rules with confidence and support metrics
- Item-wise count and support analysis

## Contact

Tejas Belakavadi Kemparaju: tb389@njit.edu


   
