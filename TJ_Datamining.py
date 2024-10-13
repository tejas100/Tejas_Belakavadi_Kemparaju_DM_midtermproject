#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import itertools
from collections import defaultdict
import time
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Function to load transactions from CSV
def load_transactions_from_csv(filename):
    df = pd.read_csv(filename, header=None)
    return [set(str(item) for item in transaction if pd.notna(item)) for transaction in df.values.tolist()]

# Function to validate input for support and confidence values
def validate_input_float(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value / 100  # Convert percentage to decimal
            else:
                print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Function to validate input for store selection
def validate_input_int(prompt, min_val, max_val):
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# Store items for transaction generation
stores = {
    "Amazon": "amazon_transactions.csv",
    "Best Buy": "best buy_transactions.csv",
    "Nike": "nike_transactions.csv",
    "Walmart": "walmart_transactions.csv",
    "Target": "target_transactions.csv"
}

# Brute force algorithm for finding frequent itemsets and generating rules
def brute_force(transactions, min_support, min_confidence):
    def get_item_counts(itemsets):
        item_counts = defaultdict(int)
        for transaction in transactions:
            for itemset in itemsets:
                if set(itemset).issubset(transaction):
                    item_counts[itemset] += 1
        return item_counts

    items = set(item for transaction in transactions for item in transaction)
    n = len(transactions)
    frequent_itemsets = {}
    k = 1

    while True:
        itemsets = list(itertools.combinations(items, k))
        item_counts = get_item_counts(itemsets)
        frequent_items = {frozenset(item): count/n for item, count in item_counts.items() if count/n >= min_support}
        if not frequent_items:
            break
        frequent_itemsets[k] = frequent_items
        k += 1

    rules = []
    for k in range(2, len(frequent_itemsets) + 1):
        for itemset in frequent_itemsets[k]:
            for i in range(1, k):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(itemset) - antecedent
                    if antecedent in frequent_itemsets[len(antecedent)]:
                        support = frequent_itemsets[k][itemset]
                        confidence = support / frequent_itemsets[len(antecedent)][antecedent]
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, confidence, support))

    return frequent_itemsets, rules


# Apriori algorithm using the mlxtend library
def library_apriori(transactions, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = mlxtend_apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

# FP-Growth algorithm using the mlxtend library
def fp_growth_method(transactions, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

# Compare the results of brute force, Apriori, and FP-Growth algorithms
def compare_results(brute_force_results, apriori_results, fpgrowth_results):
    bf_itemsets, bf_rules = brute_force_results
    ap_itemsets, ap_rules = apriori_results
    fp_itemsets, fp_rules = fpgrowth_results

    print("\nComparison of results:")
    print(f"Brute Force: {sum(len(itemsets) for itemsets in bf_itemsets.values())} frequent itemsets, {len(bf_rules)} rules")
    print(f"Apriori: {len(ap_itemsets)} frequent itemsets, {len(ap_rules)} rules")
    print(f"FP-Growth: {len(fp_itemsets)} frequent itemsets, {len(fp_rules)} rules")

    # Check if all algorithms produce the same results
    same_itemsets = (sum(len(itemsets) for itemsets in bf_itemsets.values()) == len(ap_itemsets) == len(fp_itemsets))
    same_rules = len(bf_rules) == len(ap_rules) == len(fp_rules)
    print(f"Same number of frequent itemsets: {same_itemsets}")
    print(f"Same number of rules: {same_rules}")

# Display the association rules for each algorithm
def display_rules(rules, algorithm_name):
    print(f"\n{algorithm_name} Association Rules:")
    if isinstance(rules, list):  # For brute force results
        for i, (antecedent, consequent, confidence, support) in enumerate(rules, 1):
            print(f"Rule {i}: {set(antecedent)} -> {set(consequent)} "
                  f"(Confidence: {confidence:.2f}, Support: {support:.2f})")
    elif isinstance(rules, pd.DataFrame):  # For library results (Apriori and FP-Growth)
        for i, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            print(f"Rule {i+1}: {set(antecedents)} -> {set(consequents)} "
                  f"(Confidence: {rule['confidence']:.2f}, Support: {rule['support']:.2f})")
    else:
        print("No rules found or unsupported rule format.")

# Main loop for user interaction
while True:
    print("\nWelcome User! Here are the available stores you can select from:")
    for i, store in enumerate(stores.keys(), 1):
        print(f"{i}. {store}")
    
    # Store selection using integer input validation
    store_choice = validate_input_int("Enter the number of the store you want to analyze (1-5): ", 1, 5)
    store_name = list(stores.keys())[store_choice - 1]
    
    try:
        # Load transactions from CSV file for the selected store
        transactions = load_transactions_from_csv(stores[store_name])
        
        print(f"\nAnalyzing {store_name} transactions:")
        min_support = validate_input_float("Enter minimum support (1-100): ", 1, 100)
        min_confidence = validate_input_float("Enter minimum confidence (1-100): ", 1, 100)
        
        print("\nRunning algorithms...")
        
        # Run brute force algorithm
        start_time = time.time()
        bf_itemsets, bf_rules = brute_force(transactions, min_support, min_confidence)
        bf_time = time.time() - start_time
        
        # Run Apriori algorithm
        start_time = time.time()
        ap_itemsets, ap_rules = library_apriori(transactions, min_support, min_confidence)
        ap_time = time.time() - start_time
        
        # Run FP-Growth algorithm
        start_time = time.time()
        fp_itemsets, fp_rules = fp_growth_method(transactions, min_support, min_confidence)
        fp_time = time.time() - start_time
        
        print(f"\nBrute Force Time: {bf_time:.4f} seconds")
        print(f"Apriori Time: {ap_time:.4f} seconds")
        print(f"FP-Growth Time: {fp_time:.4f} seconds")
        
        # Identify the fastest algorithm
        fastest_algorithm = min(("Brute Force", bf_time), ("Apriori", ap_time), ("FP-Growth", fp_time), key=lambda x: x[1])
        print(f"\nThe fastest algorithm was: {fastest_algorithm[0]} with a time of {fastest_algorithm[1]:.4f} seconds")
        
        # Compare results and display rules
        compare_results((bf_itemsets, bf_rules), (ap_itemsets, ap_rules), (fp_itemsets, fp_rules))
        display_rules(bf_rules, "Brute Force")
        display_rules(ap_rules, "Apriori")
        display_rules(fp_rules, "FP-Growth")
        
        # Display item counts and support
        print("\nItem Counts:")
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        for item, count in item_counts.items():
            support = count / len(transactions)
            meets_threshold = "Meets" if support >= min_support else "Does not meet"
            print(f"{item}: Count = {count}, Support = {support:.2f} ({meets_threshold} support threshold)")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your input data and try again.")
    
    # Ask user if they want to analyze another store
    print("\nDo you want to analyze another store? (y/n)")
    if input().lower() != 'y':
        break

print("Thank you!")


# In[ ]:




