"""Boltz-2 scoring placeholder.

Users should implement `get_scores` according to Boltz-2 installation and configuration.

Reference:
Passaro et al. Boltz-2: Towards Accurate and Efficient Binding Affinity
Prediction. bioRxiv, 2025. doi:10.1101/2025.06.14.659707
"""


def get_scores(smiles):
    """TODO: compute Boltz-2 scores for a list of SMILES.

    Parameters
    ----------
    smiles : list[str]
        Molecules to score.

    Returns
    -------
    dict[str, float]
        Mapping from SMILES to normalized Boltz-2 score. The training code
        expects higher scores to be better.
    """
    raise NotImplementedError(
        "Please implement Boltz-2 scoring in `get_scores(smiles)` according "
        "to your local Boltz-2 setup."
    )
