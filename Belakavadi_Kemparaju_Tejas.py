#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import itertools
from collections import defaultdict
import time
from mlxtend.frequent_patterns import apriori as apriori_func
from mlxtend.frequent_patterns import fpgrowth as fpgrowth_func
from mlxtend.frequent_patterns import association_rules as association_rules_func
from mlxtend.preprocessing import TransactionEncoder

# Load transactions from a CSV
def read_transactions(csv_file):
    raw_data = pd.read_csv(csv_file, header=None)
    return [set(str(item) for item in transaction if pd.notna(item)) for transaction in raw_data.values.tolist()]

# Input from user
def request_float(prompt, min_val, max_val):
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val / 100
            else:
                print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input, please enter a valid number.")


def request_int(prompt, min_val, max_val):
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input, please enter a valid number.")

# Brute force algorithm
def brute_force_algorithm(transactions, support_thresh, confidence_thresh):
    def count_itemsets(itemsets):
        item_count = defaultdict(int)
        for transaction in transactions:
            for itemset in itemsets:
                if set(itemset).issubset(transaction):
                    item_count[itemset] += 1
        return item_count

    items = set(item for transaction in transactions for item in transaction)
    transaction_count = len(transactions)
    frequent_sets = {}
    k = 1

    while True:
        item_combinations = list(itertools.combinations(items, k))
        item_counts = count_itemsets(item_combinations)
        frequent_items = {frozenset(itemset): count / transaction_count for itemset, count in item_counts.items() if count / transaction_count >= support_thresh}
        if not frequent_items:
            break
        frequent_sets[k] = frequent_items
        k += 1

    rules = []
    for k in range(2, len(frequent_sets) + 1):
        for itemset in frequent_sets[k]:
            for i in range(1, k):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(itemset) - antecedent
                    if antecedent in frequent_sets[len(antecedent)]:
                        support = frequent_sets[k][itemset]
                        confidence = support / frequent_sets[len(antecedent)][antecedent]
                        if confidence >= confidence_thresh:
                            rules.append((antecedent, consequent, confidence, support))

    return frequent_sets, rules

# Apriori algorithm using mlxtend
def run_apriori(transaction_data, support_thresh, confidence_thresh):
    encoder = TransactionEncoder()
    transaction_matrix = encoder.fit(transaction_data).transform(transaction_data)
    transaction_df = pd.DataFrame(transaction_matrix, columns=encoder.columns_)
    frequent_sets = apriori_func(transaction_df, min_support=support_thresh, use_colnames=True)
    rules = association_rules_func(frequent_sets, metric="confidence", min_threshold=confidence_thresh)
    return frequent_sets, rules

# FP-Growth algorithm 
def run_fp_growth(transaction_data, support_thresh, confidence_thresh):
    encoder = TransactionEncoder()
    transaction_matrix = encoder.fit(transaction_data).transform(transaction_data)
    transaction_df = pd.DataFrame(transaction_matrix, columns=encoder.columns_)
    frequent_sets = fpgrowth_func(transaction_df, min_support=support_thresh, use_colnames=True)
    rules = association_rules_func(frequent_sets, metric="confidence", min_threshold=confidence_thresh)
    return frequent_sets, rules

# Compare output between different algorithms
def compare_algorithms_output(brute_force_results, apriori_results, fp_growth_results):
    brute_itemsets, brute_rules = brute_force_results
    apriori_itemsets, apriori_rules = apriori_results
    fp_itemsets, fp_rules = fp_growth_results

    print("\nComparative Analysis of Results:")
    print(f"Brute Force - Itemsets: {sum(len(itemset) for itemset in brute_itemsets.values())}, Rules: {len(brute_rules)}")
    print(f"Apriori - Itemsets: {len(apriori_itemsets)}, Rules: {len(apriori_rules)}")
    print(f"FP-Growth - Itemsets: {len(fp_itemsets)}, Rules: {len(fp_rules)}")

    identical_itemsets = sum(len(itemset) for itemset in brute_itemsets.values()) == len(apriori_itemsets) == len(fp_itemsets)
    identical_rules = len(brute_rules) == len(apriori_rules) == len(fp_rules)
    print(f"Are itemsets identical? {identical_itemsets}")
    print(f"Are rules identical? {identical_rules}")


def show_generated_rules(rule_set, algorithm_name):
    print(f"\n{algorithm_name} Generated Rules:")
    if isinstance(rule_set, list):  # If rules come from brute force
        for idx, (antecedent, consequent, confidence, support) in enumerate(rule_set, 1):
            print(f"Rule {idx}: {set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f}, Support: {support:.2f})")
    elif isinstance(rule_set, pd.DataFrame):  # For Apriori and FP-Growth
        for idx, rule in rule_set.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            print(f"Rule {idx + 1}: {set(antecedents)} -> {set(consequents)} (Confidence: {rule['confidence']:.2f}, Support: {rule['support']:.2f})")
    else:
        print("No rules found or format not supported.")

# Main driver loop
if __name__ == "__main__":
    store_files = {
        "Amazon": "amazon_transactions.csv",
        "Best Buy": "best_buy_transactions.csv",
        "Nike": "nike_transactions.csv",
        "Walmart": "walmart_transactions.csv",
        "Target": "target_transactions.csv"
    }

    while True:
        print("\nSelect a store to process:")
        for idx, store in enumerate(store_files.keys(), 1):
            print(f"{idx}. {store}")
        store_choice = request_int("Choose store number (1-5): ", 1, 5)
        chosen_store = list(store_files.keys())[store_choice - 1]

        try:
            transactions = read_transactions(store_files[chosen_store])
            print(f"\nProcessing {chosen_store} transactions...")

            support_level = request_float("Set minimum support percentage (1-100): ", 1, 100)
            confidence_level = request_float("Set minimum confidence percentage (1-100): ", 1, 100)

            print("\nRunning algorithms...")

            # Brute force algorithm timing
            start = time.time()
            brute_force_itemsets, brute_force_rules = brute_force_algorithm(transactions, support_level, confidence_level)
            brute_force_time = time.time() - start

            # Apriori algorithm timing
            start = time.time()
            apriori_itemsets, apriori_rules = run_apriori(transactions, support_level, confidence_level)
            apriori_time = time.time() - start

            # FP-Growth algorithm timing
            start = time.time()
            fp_growth_itemsets, fp_growth_rules = run_fp_growth(transactions, support_level, confidence_level)
            fp_growth_time = time.time() - start

            # Display timing results
            print(f"\nExecution Time (seconds):\nBrute Force: {brute_force_time:.4f}\nApriori: {apriori_time:.4f}\nFP-Growth: {fp_growth_time:.4f}")

            # Determine fastest algorithm
            fastest_method = min([("Brute Force", brute_force_time), ("Apriori", apriori_time), ("FP-Growth", fp_growth_time)], key=lambda x: x[1])
            print(f"\nFastest algorithm: {fastest_method[0]} with {fastest_method[1]:.4f} seconds.")

            # Compare outputs
            compare_algorithms_output((brute_force_itemsets, brute_force_rules), (apriori_itemsets, apriori_rules), (fp_growth_itemsets, fp_growth_rules))

            # Display rules for each algorithm
            show_generated_rules(brute_force_rules, "Brute Force")
            show_generated_rules(apriori_rules, "Apriori")
            show_generated_rules(fp_growth_rules, "FP-Growth")

            # Display item frequency
            print("\nItem Frequencies in Transactions:")
            item_counts = defaultdict(int)
            for txn in transactions:
                for item in txn:
                    item_counts[item] += 1

            for item, count in item_counts.items():
                support_val = count / len(transactions)
                status = "Above" if support_val >= support_level else "Below"
                print(f"{item}: Count = {count}, Support = {support_val:.2f} ({status} threshold)")

        except Exception as ex:
            print(f"An error occurred: {str(ex)}\nPlease review the data or input and try again.")

        # Continue or exit loop
        user_choice = input("\nAnalyze another store? (y/n): ").lower()
        if user_choice != 'y':
            break

    print("Thank you for using the Market Basket Analysis Tool!")


# In[ ]:
