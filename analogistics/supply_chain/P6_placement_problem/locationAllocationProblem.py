import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def totalCostASIS(year: int,
                  productivity: float,
                  logistic_cost_asis_dist: tuple,
                  operational_cost_asis_dist: tuple,
                  direct_labor_cost_asis_dist: tuple,
                  depreciation_cost_as_is_dist: tuple,
                  fixed_cost_asis_dist: tuple):
    """
    Performs a numerical simulation given a cost model to evaluate the cash flow at the following years

    Args:
        year (int): year to simulate.
        productivity (TYPE): number of parts produced.
        logistic_cost_asis_dist (tuple): tuple with mean and variace of the logistic cost per part.
        operational_cost_asis_dist (tuple): tuple with mean and variance of the operational cost per part.
        direct_labor_cost_asis_dist (tuple): tuple with mean and variance of the direct labor cost per part.
        depreciation_cost_as_is_dist (tuple): tuple with mean and variance of the depreciation cost per part.
        fixed_cost_asis_dist (tuple): tuple with mean and variance of the fixed costs.

    Returns:
        results (dict): dictionary with results.

    """

    results = {}

    # logistic costs
    if isinstance(logistic_cost_asis_dist, list):
        total_cost = 0
        for plant in range(0, len(logistic_cost_asis_dist)):
            cost_tuple = logistic_cost_asis_dist[plant]
            plant_cost = productivity[plant][year] * np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['logistic_cost_asis'] = total_cost

    else:
        results['logistic_cost_asis'] = productivity[year] * np.random.normal(logistic_cost_asis_dist[0], logistic_cost_asis_dist[1])

    # operational costs
    if isinstance(operational_cost_asis_dist, list):
        total_cost = 0
        for plant in range(0, len(operational_cost_asis_dist)):
            cost_tuple = operational_cost_asis_dist[plant]
            plant_cost = productivity[plant][year] * np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['operational_cost_asis'] = total_cost

    else:
        results['operational_cost_asis'] = productivity[year] * np.random.normal(operational_cost_asis_dist[0], operational_cost_asis_dist[1])

    # direct labour costs
    if isinstance(direct_labor_cost_asis_dist, list):
        total_cost = 0
        for plant in range(0, len(direct_labor_cost_asis_dist)):
            cost_tuple = direct_labor_cost_asis_dist[plant]
            plant_cost = productivity[plant][year] * np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['direct_labor_cost_asis'] = total_cost

    else:
        results['direct_labor_cost_asis'] = productivity[year] * np.random.normal(direct_labor_cost_asis_dist[0], direct_labor_cost_asis_dist[1])

    # fixed costs
    if isinstance(fixed_cost_asis_dist, list):
        total_cost = 0
        for plant in range(0, len(fixed_cost_asis_dist)):
            cost_tuple = fixed_cost_asis_dist[plant]
            plant_cost = np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['fixed_cost_asis'] = total_cost

    else:
        results['fixed_cost_asis'] = np.random.normal(fixed_cost_asis_dist[0], fixed_cost_asis_dist[1])

    # depreciation costs
    if isinstance(depreciation_cost_as_is_dist, list):
        total_cost = 0
        for plant in range(0, len(depreciation_cost_as_is_dist)):
            cost_tuple = depreciation_cost_as_is_dist[plant]
            plant_cost = np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['depreciation_cost_asis'] = total_cost

    else:
        results['depreciation_cost_asis'] = np.random.normal(depreciation_cost_as_is_dist[0], depreciation_cost_as_is_dist[1])

    # total cost (cash flow) withoud depreciation
    results['total_cost_asis'] = results['logistic_cost_asis'] + results['operational_cost_asis'] + results['direct_labor_cost_asis'] + results['fixed_cost_asis']
    return results


def totalCostTOBE(year: int,
                  productivity: float,
                  logistic_cost_tobe_dist: tuple,
                  operational_cost_tobe_dist: tuple,
                  direct_labor_cost_tobe_dist: tuple,
                  depreciation_cost_tobe_dist: tuple,
                  investment_cost_tobe_dist: tuple,
                  fixed_cost_tobe_dist: tuple):
    """

    Args:
        year (int): DESCRIPTION.
        productivity (float): number of parts produced.
        logistic_cost_asis_dist (tuple): tuple with mean and variace of the logistic cost per part.
        operational_cost_asis_dist (tuple): tuple with mean and variance of the operational cost per part.
        direct_labor_cost_asis_dist (tuple): tuple with mean and variance of the direct labor cost per part.
        depreciation_cost_as_is_dist (tuple): tuple with mean and variance of the depreciation cost per part.
        investment_cost_tobe_dist (tuple): tuple with mean and variance of the investment cost.
        fixed_cost_asis_dist (tuple): tuple with mean and variance of the fixed costs.

    Returns:
        results (TYPE): DESCRIPTION.

    """

    results = {}

    # logistic costs
    if isinstance(logistic_cost_tobe_dist, list):
        total_cost = 0
        for plant in range(0, len(logistic_cost_tobe_dist)):
            cost_tuple = logistic_cost_tobe_dist[plant]
            plant_cost = productivity[plant][year] * np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['logistic_cost_tobe'] = total_cost

    else:
        results['logistic_cost_tobe'] = productivity[year] * np.random.normal(logistic_cost_tobe_dist[0], logistic_cost_tobe_dist[1])

    # operational costs
    if isinstance(operational_cost_tobe_dist, list):
        total_cost = 0
        for plant in range(0, len(operational_cost_tobe_dist)):
            cost_tuple = operational_cost_tobe_dist[plant]
            plant_cost = productivity[plant][year] * np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['operational_cost_tobe'] = total_cost

    else:
        results['operational_cost_tobe'] = productivity[year] * np.random.normal(operational_cost_tobe_dist[0], operational_cost_tobe_dist[1])

    # direct labour costs
    if isinstance(direct_labor_cost_tobe_dist, list):
        total_cost = 0
        for plant in range(0, len(direct_labor_cost_tobe_dist)):
            cost_tuple = direct_labor_cost_tobe_dist[plant]
            plant_cost = productivity[plant][year] * np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['direct_labor_cost_tobe'] = total_cost

    else:
        results['direct_labor_cost_tobe'] = productivity[year] * np.random.normal(direct_labor_cost_tobe_dist[0], direct_labor_cost_tobe_dist[1])

    # fixed costs
    if isinstance(fixed_cost_tobe_dist, list):
        total_cost = 0
        for plant in range(0, len(fixed_cost_tobe_dist)):
            cost_tuple = fixed_cost_tobe_dist[plant]
            plant_cost = np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['fixed_cost_tobe'] = total_cost

    else:
        results['fixed_cost_tobe'] = np.random.normal(fixed_cost_tobe_dist[0], fixed_cost_tobe_dist[1])

    # depreciation costs
    if isinstance(depreciation_cost_tobe_dist, list):
        total_cost = 0
        for plant in range(0, len(depreciation_cost_tobe_dist)):
            cost_tuple = depreciation_cost_tobe_dist[plant]
            plant_cost = np.random.normal(cost_tuple[0], cost_tuple[1])
            total_cost = total_cost + plant_cost

        # update the global result
        results['depreciation_cost_tobe'] = total_cost

    else:
        results['depreciation_cost_tobe'] = np.random.normal(depreciation_cost_tobe_dist[0], depreciation_cost_tobe_dist[1])

    # investment costs
    if isinstance(investment_cost_tobe_dist, list):
        investment_cost = 0
        for plant in range(0, len(investment_cost_tobe_dist)):
            cost_tuple = investment_cost_tobe_dist[plant]
            plant_cost = np.random.normal(cost_tuple[0], cost_tuple[1])
            investment_cost = investment_cost + plant_cost

    else:
        investment_cost = np.random.normal(investment_cost_tobe_dist[0], investment_cost_tobe_dist[1])

    # return total cost per year plus investment cost at year zero (cash flow with no depreciation)
    if year == 0:  # if year=0 consider the investment cost
        results['total_cost_tobe'] = investment_cost + results['logistic_cost_tobe'] + results['operational_cost_tobe'] + results['direct_labor_cost_tobe'] + results['fixed_cost_tobe']
    else:
        results['total_cost_tobe'] = results['logistic_cost_tobe'] + results['operational_cost_tobe'] + results['direct_labor_cost_tobe'] + results['fixed_cost_tobe']

    return results


def runSimulation(num_iter: int,
                  years: int,
                  productivity_asis: int,
                  logistic_cost_asis_dist: tuple,
                  operational_cost_asis_dist: tuple,
                  direct_labor_cost_asis_dist: tuple,
                  depreciation_cost_as_is_dist: tuple,
                  fixed_cost_asis_dist: tuple,
                  productivity_tobe: int,
                  logistic_cost_tobe_dist: tuple,
                  operational_cost_tobe_dist: tuple,
                  direct_labor_cost_tobe_dist: tuple,
                  depreciation_cost_tobe_dist: tuple,
                  investment_cost_tobe_dist: tuple,
                  fixed_cost_tobe_dist: tuple) -> plt.figure:
    """
    Run montecarlo simulation for LAP.
    Args:
        num_iter (int): Number of iterations of the simulation.
        years (int): Number of years to simulate.
        productivity_asis (int): Number of parts to produce as-is.
        logistic_cost_asis_dist (tuple): tuple with mean and variance of the logistic cost as-is.
        operational_cost_asis_dist (tuple): tuple with mean and variance of the operational cost as-is.
        direct_labor_cost_asis_dist (tuple): tuple with mean and variance of the direct labor cost as-is.
        depreciation_cost_as_is_dist (tuple): tuple with mean and variance of the depreciation cost as-is.
        fixed_cost_asis_dist (tuple): tuple with mean and variance of the fixed cost as-is.
        productivity_tobe (int): Number of parts to produce to-be.
        logistic_cost_tobe_dist (tuple): tuple with mean and variance of the logistic cost to-be.
        operational_cost_tobe_dist (tuple): tuple with mean and variance of the operational cost to-be.
        direct_labor_cost_tobe_dist (tuple): tuple with mean and variance of the direct laor cost to-be.
        depreciation_cost_tobe_dist (tuple): tuple with mean and variance of the depreciation cost to-be.
        investment_cost_tobe_dist (tuple): tuple with mean and variance of the investment cost to-be.
        fixed_cost_tobe_dist (tuple): tuple with mean and variance of the fixed cost to-be.

    Returns:
        fig1 (plt.figure): Output figure of the simulation.

    """
    fig1 = plt.figure(figsize=[10, 8])  # define resulting figure

    for i in range(0, num_iter):
        cost_asis = []
        cost_tobe = []
        for j in range(0, years):

            # calculate and append costs
            results_asis = totalCostASIS(year=j,
                                         productivity=productivity_asis,
                                         logistic_cost_asis_dist=logistic_cost_asis_dist,
                                         operational_cost_asis_dist=operational_cost_asis_dist,
                                         direct_labor_cost_asis_dist=direct_labor_cost_asis_dist,
                                         depreciation_cost_as_is_dist=depreciation_cost_as_is_dist,
                                         fixed_cost_asis_dist=fixed_cost_asis_dist)
            cost_asis.append(results_asis['total_cost_asis'])

            results_tobe = totalCostTOBE(year=j,
                                         productivity=productivity_tobe,
                                         logistic_cost_tobe_dist=logistic_cost_tobe_dist,
                                         operational_cost_tobe_dist=operational_cost_tobe_dist,
                                         direct_labor_cost_tobe_dist=direct_labor_cost_tobe_dist,
                                         depreciation_cost_tobe_dist=depreciation_cost_tobe_dist,
                                         investment_cost_tobe_dist=investment_cost_tobe_dist,
                                         fixed_cost_tobe_dist=fixed_cost_tobe_dist
                                         )

            cost_tobe.append(results_tobe['total_cost_tobe'])

        # generates cumulative functions
        cost_asis = np.array(cost_asis)
        cost_tobe = np.array(cost_tobe)
        cost_asis_cum = np.cumsum(cost_asis)
        cost_tobe_cum = np.cumsum(cost_tobe)

        cashflow = cost_asis_cum - cost_tobe_cum
        plt.plot(range(0, years), cashflow)

    plt.hlines(0, 0, years)
    plt.title('Montecarlo LAP')
    plt.xlabel('years')
    plt.ylabel('return - €')
    return fig1


def runSimulationNoVariance(num_iter: int,
                            years: int,
                            productivity_asis: int,
                            logistic_cost_asis_dist: tuple,
                            operational_cost_asis_dist: tuple,
                            direct_labor_cost_asis_dist: tuple,
                            depreciation_cost_as_is_dist: tuple,
                            fixed_cost_asis_dist: tuple,
                            productivity_tobe: int,
                            logistic_cost_tobe_dist: tuple,
                            operational_cost_tobe_dist: tuple,
                            direct_labor_cost_tobe_dist: tuple,
                            depreciation_cost_tobe_dist: tuple,
                            investment_cost_tobe_dist: tuple,
                            fixed_cost_tobe_dist: tuple):
    """

    Args:
        num_iter (int): Number of iterations of the simulation.
        years (int): Number of years to simulate.
        productivity_asis (int): Number of parts to produce as-is.
        logistic_cost_asis_dist (tuple): tuple with mean and variance of the logistic cost as-is.
        operational_cost_asis_dist (tuple): tuple with mean and variance of the operational cost as-is.
        direct_labor_cost_asis_dist (tuple): tuple with mean and variance of the direct labor cost as-is.
        depreciation_cost_as_is_dist (tuple): tuple with mean and variance of the depreciation cost as-is.
        fixed_cost_asis_dist (tuple): tuple with mean and variance of the fixed cost as-is.
        productivity_tobe (int): Number of parts to produce to-be.
        logistic_cost_tobe_dist (tuple): tuple with mean and variance of the logistic cost to-be.
        operational_cost_tobe_dist (tuple): tuple with mean and variance of the operational cost to-be.
        direct_labor_cost_tobe_dist (tuple): tuple with mean and variance of the direct laor cost to-be.
        depreciation_cost_tobe_dist (tuple): tuple with mean and variance of the depreciation cost to-be.
        investment_cost_tobe_dist (tuple): tuple with mean and variance of the investment cost to-be.
        fixed_cost_tobe_dist (tuple): tuple with mean and variance of the fixed cost to-be.

    Returns:
        fig1 (plt.figure): Output figure of the simulation.
        df_results (pd.DataFrame): Output dataframe of the simulation.

    """

    fig1 = plt.figure(figsize=[10, 8])  # define resulting figure

    df_results = {}
    df_columns = ['year', 'personnell cost', 'logistics cost', 'operational cost',
                  'depreciation cost', 'fixed cost']

    df_conto_economico_asis = pd.DataFrame(columns=df_columns)
    df_conto_economico_tobe = pd.DataFrame(columns=df_columns)

    # cancel parameters variances

    # define AS-IS parameters
    logistic_cost_asis_dist_medione = (logistic_cost_asis_dist[0], 0)
    operational_cost_asis_dist_medione = (operational_cost_asis_dist[0], 0)
    direct_labor_cost_asis_dist_medione = (direct_labor_cost_asis_dist[0], 0)
    depreciation_cost_as_is_dist_medione = (depreciation_cost_as_is_dist[0], 0)
    fixed_cost_asis_dist_medione = (fixed_cost_asis_dist[0], 0)

    # logistic costs
    if isinstance(logistic_cost_tobe_dist, list):
        logistic_cost_tobe_dist_medione = []
        for tupla in logistic_cost_tobe_dist:
            logistic_cost_tobe_dist_medione.append((tupla[0], 0))
    else:
        logistic_cost_tobe_dist_medione = (logistic_cost_tobe_dist[0], 0)

    # operational costs
    if isinstance(operational_cost_tobe_dist, list):
        operational_cost_tobe_dist_medione = []
        for tupla in operational_cost_tobe_dist:
            operational_cost_tobe_dist_medione.append((tupla[0], 0))
    else:
        operational_cost_tobe_dist_medione = (operational_cost_tobe_dist[0], 0)

    # direct labour costs
    if isinstance(direct_labor_cost_tobe_dist, list):
        direct_labor_cost_tobe_dist_medione = []
        for tupla in direct_labor_cost_tobe_dist:
            direct_labor_cost_tobe_dist_medione.append((tupla[0], 0))
    else:
        direct_labor_cost_tobe_dist_medione = (direct_labor_cost_tobe_dist[0], 0)

    # depreciation costs
    if isinstance(depreciation_cost_tobe_dist, list):
        depreciation_cost_tobe_dist_medione = []
        for tupla in depreciation_cost_tobe_dist:
            depreciation_cost_tobe_dist_medione.append((tupla[0], 0))
    else:
        depreciation_cost_tobe_dist_medione = (depreciation_cost_tobe_dist[0], 0)

    # fixed costs
    if isinstance(fixed_cost_tobe_dist, list):
        fixed_cost_tobe_dist_medione = []
        for tupla in fixed_cost_tobe_dist:
            fixed_cost_tobe_dist_medione.append((tupla[0], 0))
    else:
        fixed_cost_tobe_dist_medione = (fixed_cost_tobe_dist[0], 0)

    # investment costs
    if isinstance(investment_cost_tobe_dist, list):
        investment_cost_tobe_dist_medione = []
        for tupla in investment_cost_tobe_dist:
            investment_cost_tobe_dist_medione.append((tupla[0], 0))
    else:
        investment_cost_tobe_dist_medione = (investment_cost_tobe_dist[0], 0)

    for i in range(0, num_iter):
        cost_asis = []
        cost_tobe = []
        for j in range(0, years):

            # generate as is costs
            results_asis = totalCostASIS(year=j,
                                         productivity=productivity_asis,
                                         logistic_cost_asis_dist=logistic_cost_asis_dist_medione,
                                         operational_cost_asis_dist=operational_cost_asis_dist_medione,
                                         direct_labor_cost_asis_dist=direct_labor_cost_asis_dist_medione,
                                         depreciation_cost_as_is_dist=depreciation_cost_as_is_dist_medione,
                                         fixed_cost_asis_dist=fixed_cost_asis_dist_medione)
            cost_asis.append(results_asis['total_cost_asis'])

            # update the income statement
            df_conto_economico_asis = df_conto_economico_asis.append(pd.DataFrame([[j,
                                                                     results_asis['direct_labor_cost_asis'],
                                                                     results_asis['logistic_cost_asis'],
                                                                     results_asis['operational_cost_asis'],
                                                                     results_asis['depreciation_cost_asis'],
                                                                     results_asis['fixed_cost_asis'],
                                                                     ]], columns=df_columns))

            # generate to-be costs
            results_tobe = totalCostTOBE(year=j,
                                         productivity=productivity_tobe,
                                         logistic_cost_tobe_dist=logistic_cost_tobe_dist_medione,
                                         operational_cost_tobe_dist=operational_cost_tobe_dist_medione,
                                         direct_labor_cost_tobe_dist=direct_labor_cost_tobe_dist_medione,
                                         depreciation_cost_tobe_dist=depreciation_cost_tobe_dist_medione,
                                         investment_cost_tobe_dist=investment_cost_tobe_dist_medione,
                                         fixed_cost_tobe_dist=fixed_cost_tobe_dist_medione
                                         )
            cost_tobe.append(results_tobe['total_cost_tobe'])

            # update the income statement
            df_conto_economico_tobe = df_conto_economico_tobe.append(pd.DataFrame([[j,
                                                                                    results_tobe['direct_labor_cost_tobe'],
                                                                                    results_tobe['logistic_cost_tobe'],
                                                                                    results_tobe['operational_cost_tobe'],
                                                                                    results_tobe['depreciation_cost_tobe'],
                                                                                    results_tobe['fixed_cost_tobe'],
                                                                                    ]], columns=df_columns))
        # calculate cumulative cash flows
        cost_asis = np.array(cost_asis)
        cost_tobe = np.array(cost_tobe)
        cost_asis_cum = np.cumsum(cost_asis)
        cost_tobe_cum = np.cumsum(cost_tobe)

        cashflow = cost_asis_cum - cost_tobe_cum
        plt.plot(range(0, years), cashflow)

    plt.hlines(0, 0, years)
    plt.title('Montecarlo LAP')
    plt.xlabel('years')
    plt.ylabel('return - €')

    # generate income statement to be
    df_conto_economico_tobe = df_conto_economico_tobe.groupby(['year']).agg({'personnell cost': ['mean'],
                                                                             'logistics cost': ['mean'],
                                                                             'operational cost': ['mean'],
                                                                             'depreciation cost': ['mean'],
                                                                             'fixed cost': ['mean']
                                                                             }).reset_index()
    df_conto_economico_tobe.columns = ['year', 'personnell cost', 'logistics cost', 'operational cost',
                                       'depreciation cost', 'fixed cost']
    df_results['df_conto_economico_tobe'] = df_conto_economico_tobe

    # generate income statement tobe per part
    df_conto_economico_tobe_pasto = df_conto_economico_tobe.copy()
    for column in ['year', 'personnell cost', 'logistics cost', 'operational cost',
                   'depreciation cost', 'fixed cost']:

        df_conto_economico_tobe_pasto[column] = df_conto_economico_tobe_pasto[column] / productivity_asis[0]
    df_results['df_conto_economico_tobe_part'] = df_conto_economico_tobe_pasto

    # generate income statement asis
    df_conto_economico_asis = df_conto_economico_asis.groupby(['year']).agg({'personnell cost': ['mean'],
                                                                             'logistics cost': ['mean'],
                                                                             'operational cost': ['mean'],
                                                                             'depreciation cost': ['mean'],
                                                                             'fixed cost': ['mean']
                                                                             }).reset_index()
    df_conto_economico_asis.columns = ['year', 'personnell cost', 'logistics cost', 'operational cost',
                                       'depreciation cost', 'fixed cost']

    df_results['df_conto_economico_asis'] = df_conto_economico_asis

    # generate income statement asis per part
    df_conto_economico_asis_pasto = df_conto_economico_asis.copy()
    for column in ['year', 'personnell cost', 'logistics cost', 'operational cost',
                   'depreciation cost', 'fixed cost']:

        df_conto_economico_asis_pasto[column] = df_conto_economico_asis_pasto[column] / productivity_asis[0]
    df_results['df_conto_economico_asis_part'] = df_conto_economico_asis_pasto

    return fig1, df_results
