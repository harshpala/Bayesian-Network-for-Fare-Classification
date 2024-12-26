#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model
# import time

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    
    trainData = pd.read_csv("./data/train_data.csv")
    validationData = pd.read_csv("./data/validation_data.csv")
    return trainData, validationData    

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    
    edges = []
    nodeOrder = ['Start_Stop_ID', 'End_Stop_ID', 'Distance', 'Zones_Crossed', 'Route_Type', 'Fare_Category']

    for i, startNode in enumerate(nodeOrder):
        for endNode in nodeOrder[i+1:]:
            edges.append((startNode, endNode))

    DAG = bn.make_DAG(edges)
    gviz = bn.plot_graphviz(DAG)
    gviz.render('Initial Bayesian Network', format='png', cleanup=True)
    
    # # Measure the time to fit the initial model
    # start_time = time.time()
    fittedModel = bn.parameter_learning.fit(DAG, df)
    # initial_fit_time = time.time() - start_time
    # print("\nEfficiency Comparison:")
    # print(f"Initial Model Fit Time: {initial_fit_time:.4f} seconds")
    return fittedModel

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    
    with open("base_model.pkl", 'rb') as f:
        initialModel = pickle.load(f)
    
    ''' not setting the alpha value makes it use the default which is 0.05 (5% significance level for pruning) and 
        explicitly specifying alpha we can control the strictness of the independence test,
        lower value means aggressive pruning - retaining only edges with very strong statistical significance and 
        higher value means pruning less aggressively - retaining edges with weaker significance.
    '''
    prunedModel = bn.independence_test(initialModel, df, prune=True)
    gviz = bn.plot_graphviz(prunedModel)
    gviz.render('Pruned Bayesian Network', format = 'png', cleanup=True)

    #  # Measure the time to fit the pruned model
    # start_time = time.time()
    fittedModel = bn.parameter_learning.fit(prunedModel, df)
    # pruned_fit_time = time.time() - start_time
    # print("\nEfficiency Comparison:")
    # print(f"Pruned Model Fit Time: {pruned_fit_time:.4f} seconds")

    return fittedModel

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    
    
    hcModel = bn.structure_learning.fit(
        df,
        methodtype='hc',
        scoretype='bic'
    )

    gviz = bn.plot_graphviz(hcModel)
    gviz.render('Optimized Bayesian Network', format = 'png', cleanup=True)

    # # Measure the time to fit the initial model
    # start_time = time.time()
    fittedModel = bn.parameter_learning.fit(hcModel, df)
    # optimized_fit_time = time.time() - start_time
    # print("\nEfficiency Comparison:")
    # print(f"Optimized Model Fit Time: {optimized_fit_time:.4f} seconds")
    return fittedModel

def save_model(fname, model):
    """Save the model to a file using pickle."""
    
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

        
# def compare_network_performance(initial_model, pruned_model):
#     """Compare the performance and complexity of initial and pruned Bayesian network models."""
#     # Calculate edge reduction
#     initial_edges = set(initial_model['model_edges'])
#     pruned_edges = set(pruned_model['model_edges'])
#     removed_edges = initial_edges - pruned_edges
#     edge_reduction = (len(initial_edges) - len(pruned_edges)) / len(initial_edges) * 100

#     # Print edge information
#     print("Initial Model Edges:", len(initial_edges))
#     print("new Model Edges:", len(pruned_edges))
#     print("Reduced Edges:", len(removed_edges), f"({edge_reduction:.2f}% reduction)")


############
## Driver ##
############

def main():

    # Load data
    train_df, val_df = load_data()
    print("data loaded..........")

    # Create and save base model
    
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    # print("\nModel Edge Comparison:")
    # compare_network_performance(base_model, pruned_network)
    # compare_network_performance(base_model, optimized_network)

    print("[+] Done")

if __name__ == "__main__":
    main()

