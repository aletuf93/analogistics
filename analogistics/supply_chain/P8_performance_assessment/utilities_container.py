import pandas as pd


def convertContainerISOcodeToTEUFEU(D_hu: pd.DataFrame, codeField: str = '_id'):
    """
    Consider container ISO codes to define their size (TEU or FEU)

    Args:
        D_hu (pd.DataFrame): Input dataframe.
        codeField (str, optional): Column name containing ISO code of the container. Defaults to '_id'.

    Returns:
        D_hu (pd.Dataframe): Output dataframe with added size column.

    """

    ctSize = D_hu[codeField].str[:1]
    TEU = ctSize == '2'
    FEU = ctSize == '4'
    L5GO = ctSize == 'L'
    ctSize[TEU] = 'TEU'
    ctSize[FEU] = 'FEU'
    ctSize[L5GO] = 'L5GO'
    ctSize[~ (TEU | FEU | L5GO)] = 'OTHER'
    D_hu['ContainerSize'] = ctSize

    return D_hu
