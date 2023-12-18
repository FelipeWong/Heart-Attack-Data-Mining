from mlxtend.frequent_patterns import apriori, association_rules


def find_association_rules(data, min_support, min_confidence):
    # Apply Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)

    # Sort the rules by support and confidence in descending order
    rules = rules.sort_values(['support', 'confidence'], ascending=False)

    return rules
