import networkx as nx
import powerlaw
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_graph(filepath):
    print(f"\nAnalyzing {filepath}...")
    try:
        G = nx.read_graphml(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Get degree sequence
    degrees = [d for n, d in G.degree()]
    degrees = [d for d in degrees if d > 0] # Powerlaw requires data > 0
    
    if not degrees:
        print("Graph has no edges or nodes with degree > 0.")
        return

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    
    # Fit power law
    # Discrete=True because degrees are integers
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    
    print(f"Power-law fit results:")
    print(f"  alpha: {fit.power_law.alpha:.4f}")
    print(f"  xmin: {fit.power_law.xmin}")
    print(f"  sigma: {fit.power_law.sigma:.4f}")
    
    # Compare distributions
    print("Comparison with other distributions:")
    
    # Power law vs Exponential
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"  vs Exponential: R={R:.4f}, p={p:.4f}")
    if p < 0.05:
        if R > 0:
            print("    -> Power law is favored over Exponential.")
        else:
            print("    -> Exponential is favored over Power law.")
    else:
        print("    -> Inconclusive (p >= 0.05).")

    # Power law vs Lognormal
    R, p = fit.distribution_compare('power_law', 'lognormal')
    print(f"  vs Lognormal: R={R:.4f}, p={p:.4f}")
    if p < 0.05:
        if R > 0:
            print("    -> Power law is favored over Lognormal.")
        else:
            print("    -> Lognormal is favored over Power law.")
    else:
        print("    -> Inconclusive (p >= 0.05).")

    # Plot
    fig = fit.plot_pdf(color='b', linewidth=2, label='Empirical PDF')
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig, label='Power Law Fit')
    fit.plot_ccdf(color='r', linewidth=2, ax=fig, label='Empirical CCDF')
    fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig, label='Power Law CCDF')
    
    plt.title(f"Degree Distribution Analysis: {os.path.basename(filepath)}")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k) / P(K>=k)")
    plt.legend()
    
    output_plot = filepath.replace('.graphml', '_degree_dist.png')
    plt.savefig(output_plot)
    plt.close()
    print(f"Plot saved to {output_plot}")

def main():
    files = sorted(glob.glob("/app/output/thesis_outputs/checkpoints/iteration_*_graph.graphml"))
    if not files:
        print("No graph files found.")
        return
        
    for f in files:
        analyze_graph(f)

if __name__ == "__main__":
    main()
