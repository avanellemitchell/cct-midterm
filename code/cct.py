import pandas as pd
import numpy as np

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def load_plant_knowledge_data(filepath):
    """
    Load plant knowledge CSV and return data as a NumPy array.
    Drops the 'Informant' column.
    """
    df = pd.read_csv(filepath)
    data = df.drop(columns='Informant').values
    return data

def build_cct_model(X):
    """
    Build and return the PyMC model for Cultural Consensus Theory.
    
    Parameters:
    -----------
    X : np.ndarray
        Binary response matrix of shape (N_informants, M_questions)
    
    Returns:
    --------
    model : pm.Model
        The PyMC model
    """
    N, M = X.shape  # number of informants and questions

    with pm.Model() as model:
        # Prior: Competence Di ∈ [0.5, 1.0]
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)

        # Prior: Consensus answers Zj ∈ {0,1}, weak prior (coin flip)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Reshape for broadcasting
        D_matrix = D[:, None]  # shape (N, 1), to match (N, M)

        # Core formula: pij = Z * D + (1 - Z) * (1 - D)
        p = Z * D_matrix + (1 - Z) * (1 - D_matrix)

        # Likelihood: Each response is a Bernoulli draw
        pm.Bernoulli("X", p=p, observed=X)

    return model


if __name__ == "__main__":
    data = load_plant_knowledge_data("../data/plant_knowledge.csv")
    print("Data shape:", data.shape)
    print(data[:2])  # Show first 2 rows

    # Step 1: Build the model
    model = build_cct_model(data)

    # Step 2: Sample from the posterior
    with model:
        idata = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True)

    # Step 3: Print summary
    print("\nPosterior Summary:")
    summary = az.summary(idata, var_names=["D", "Z"], round_to=2)
    print(summary)

    # Plot all D values (Informant competence) in one grid
    az.plot_posterior(
        idata,
        var_names=["D"],
        hdi_prob=0.94,
        grid=(2, 5),  # 2 rows, 5 columns for 10 informants
        figsize=(14, 6)
    )
    plt.suptitle("Posterior Distributions of Informant Competence (D)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot all Z values (Consensus answers) in one grid
    az.plot_posterior(
        idata,
        var_names=["Z"],
        hdi_prob=0.94,
        grid=(4, 5),  # 4 rows, 5 columns for 20 questions
        figsize=(14, 10)
    )
    plt.suptitle("Posterior Distributions of Consensus Answers (Z)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Step 4: Compare with Majority Vote
    import numpy as np

    # Reload data to make sure it's in the right format
    X = load_plant_knowledge_data("../data/plant_knowledge.csv")

    # Compute majority vote for each question (axis=0)
    majority_vote = np.round(X.mean(axis=0)).astype(int)

    # Compute posterior mean for Z
    z_means = idata.posterior["Z"].mean(dim=("chain", "draw")).values

    # Round Z posterior means to get model's consensus answer key
    model_consensus = np.round(z_means).astype(int)

    print("\nMajority Vote:")
    print(majority_vote)

    print("\nModel Consensus (rounded):")
    print(model_consensus)

    # Optional: Print where they differ
    print("\nQuestions where the model and majority vote differ:")
    diff_indices = np.where(majority_vote != model_consensus)[0]
    print(diff_indices)






