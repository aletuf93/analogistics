# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import random
import datetime


class _node():

    def __init__(self, nodecode: int, min_latitude: float, max_latitude: float,
                 min_longitude: float, max_longitude: float):
        self.NODECODE = nodecode
        self.DESCRIPTION = f"NODE_{nodecode}"
        self.LATITUDE = np.random.uniform(min_latitude, max_latitude)  # latitude
        self.LONGITUDE = np.random.uniform(min_longitude, max_longitude)  # longitude
        self.CLIENT_TYPE = random.choice(['CLIENT_TYPE_1', 'CLIENT_TYPE_2'])  # type of client
        self.SIZE = np.random.uniform(1, 30)  # client size flow


class _part():
    def __init__(self, itemcode: str):
        self.ITEMCODE = itemcode
        self.PRODUCT_FAMILY = random.choice(['PRODUCT_FAMILY 1', 'PRODUCT_FAMILY 2'])  # only two product families


class _plant():
    def __init__(self, nodecode: str, listClient: list, min_latitude: float, max_latitude: float,
                 min_longitude: float, max_longitude: float):
        self.NODECODE = nodecode
        self.DESCRIPTION = f"NODE_{nodecode}"
        self.LATITUDE = np.random.uniform(min_latitude, max_latitude)  # latitude
        self.LONGITUDE = np.random.uniform(min_longitude, max_longitude)  # longitude
        self.listClient = listClient


class _movement():
    def __init__(self, nodeFrom: str, nodeTo: str, quantity: float, vehicle: str,
                 voyage: str, client: str, part: str, booking_timestamp: datetime.datetime,
                 execution_timestamp: datetime.datetime, travelTime: float,
                 average_time_window_days: float, num_users: int):

        # nodecode FROM information
        self.LOADING_NODE = nodeFrom.NODECODE
        self.LOADING_NODE_DESCRIPTION = nodeFrom.DESCRIPTION
        self.LOADING_NODE_LATITUDE = nodeFrom.LATITUDE
        self.LOADING_NODE_LONGITUDE = nodeFrom.LONGITUDE

        # planned
        self.PTA_FROM = execution_timestamp
        self.PTD_FROM = self.PTA_FROM + datetime.timedelta(average_time_window_days)  # static time windows

        # actual
        self.ATA_FROM = self.PTA_FROM + datetime.timedelta(np.random.normal(0, average_time_window_days / 4))
        self.ATD_FROM = self.PTD_FROM + datetime.timedelta(np.random.normal(0, average_time_window_days / 4))

        # nodecode TO information
        self.DISCHARGING_NODE = nodeTo.NODECODE
        self.DISCHARGING_NODE_DESCRIPTION = nodeTo.DESCRIPTION
        self.DISCHARGING_LATITUDE = nodeTo.LATITUDE
        self.DISCHARGING_LONGITUDE = nodeTo.LONGITUDE

        # planned
        self.PTA_TO = self.PTD_FROM + datetime.timedelta(np.random.normal(travelTime, travelTime / 4))
        self.PTD_TO = self.PTA_TO + datetime.timedelta(average_time_window_days)  # static time windows

        # actual
        self.ATA_TO = self.PTA_TO + datetime.timedelta(np.random.normal(0, average_time_window_days / 4))
        self.ATD_TO = self.PTD_TO + datetime.timedelta(np.random.normal(0, average_time_window_days / 4))

        # other information
        self.ITEMCODE = part.ITEMCODE
        self.PRODUCT_FAMILY = part.PRODUCT_FAMILY
        self.CLIENT = client
        self.VEHICLE_CODE = vehicle
        self.VOYAGE_CODE = voyage
        self.QUANTITY = quantity
        self.TIMESTAMP_IN = booking_timestamp
        self.PACKAGE_DESCRIPTION = random.choice(['TEU CONTAINER', 'FEU CONTAINER'])  # only two type of packages
        self.USER = random.choice([f"USER_{i}" for i in list(np.arange(0, num_users))])


def generateDistributionData(num_nodes: int = 25,
                             min_latitude: float = 41.413896,
                             max_latitude: float = 41.945192,
                             min_longitude: float = 13.972079,
                             max_longitude: float = 15.056329,
                             num_plants: int = 2,
                             num_parts: int = 2,
                             num_users: int = 8,
                             num_movements: int = 100,
                             movements_per_voyage: int = 25,
                             average_advance_planning: int = 7,  # days
                             average_time_window_days: float = 1 / 24,  # days (1 hour)
                             average_time_between_movements: float = 1 / 24 * 2,  # average two hours between movements
                             first_day: datetime.datetime = datetime.datetime(year=2020, month=1, day=2)
                             ):

    dict_nodes = {}
    nodecodes = np.arange(0, num_nodes)
    for nodecode in nodecodes:
        dict_nodes[nodecode] = _node(nodecode, min_latitude, max_latitude, min_longitude, max_longitude)

    dict_parts = {}
    itemcodes = np.arange(0, num_parts)
    for itemcode in itemcodes:
        dict_parts[itemcode] = _part(itemcode)

    dict_plant = {}
    plants = np.arange(0, num_plants)
    all_nodes = list(dict_nodes.keys())
    assigned_plant = [random.choice(plants) for i in all_nodes]

    for plant_code in plants:
        plant_name = f"PLANT_{plant_code}"
        idx = assigned_plant == plant_code
        idx_num = [i for i, x in enumerate(idx) if x]
        nodes_code = [str(all_nodes[node]) for node in idx_num]
        dict_plant[plant_name] = _plant(plant_name, nodes_code,
                                        min_latitude, max_latitude,
                                        min_longitude, max_longitude)  # all nodes of the network served by a single plant

    dict_movements = {}
    num_creati = 0

    while num_creati < num_movements:
        num_creati = num_creati + 1

        # random select nodeFrom
        nodeFrom = random.choice(dict_nodes)

        # random select nodeTo
        nodeTo = random.choice(dict_nodes)

        # random generate other values
        quantity = np.random.uniform(1, 10)
        vehicle = 'TRUCK 1'  # single vehicle
        client = random.choice(['CLIENT 1', 'CLIENT 2'])

        # random select sku
        part = random.choice(dict_parts)

        if num_creati == 1:
            execution_timestamp = first_day
        else:
            # generate waiting time
            wait = np.random.exponential(average_time_between_movements)
            execution_timestamp = dict_movements[num_creati - 1].PTD_TO + datetime.timedelta(wait)

        # generate booking timestamp
        advance = np.random.exponential(average_advance_planning)
        booking_timestamp = execution_timestamp - datetime.timedelta(advance)

        # generate travel time
        travelTime = np.abs(nodeFrom.LATITUDE - nodeTo.LATITUDE) + np.abs(nodeFrom.LONGITUDE - nodeTo.LONGITUDE)

        # generate voyage code
        voyage = math.floor(num_creati / movements_per_voyage)

        # create movement
        dict_movements[num_creati] = _movement(nodeFrom, nodeTo, quantity, vehicle,
                                               voyage, client, part, booking_timestamp,
                                               execution_timestamp, travelTime,
                                               average_time_window_days, num_users)

    # SAVE MOVEMENTS AND EXPORT
    D_nodes = pd.DataFrame()
    for node in dict_nodes:
        D_nodes = D_nodes.append(pd.DataFrame([vars(dict_nodes[node])]))
    # D_nodes.to_excel('nodes.xlsx')

    # SAVE SKUS
    D_parts = pd.DataFrame()
    for part in dict_parts:
        D_parts = D_parts.append(pd.DataFrame([vars(dict_parts[part])]))
    # D_parts.to_excel('parts.xlsx')

    # SAVE PLANTS
    D_plants = pd.DataFrame()
    for plant in dict_plant:
        D_plants = D_plants.append(pd.DataFrame([vars(dict_plant[plant])]))
    # D_plants.to_excel('plants.xlsx')

    # SAVE MOVEMENTS
    D_movements = pd.DataFrame()
    for mov in dict_movements:
        D_movements = D_movements.append(pd.DataFrame([vars(dict_movements[mov])]))
    # D_movements.to_excel('movements_dist.xlsx')

    return D_nodes, D_parts, D_plants, D_movements
