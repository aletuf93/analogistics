# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:30:38 2020

@author: aletu
"""
import numpy as np
import pandas as pd
import random
import datetime


def generateWarehouseData(num_SKUs: int = 100,
                          nodecode: int = 1,
                          idwh: list = ['LOGICAL_WH1', 'LOGICAL_WH2', 'FAKE'],
                          whsubarea: list = ['AREA 1'],
                          num_aisles: int = 5,
                          num_bays: int = 66,
                          num_levels: int = 5,
                          level_height: int = 1200,
                          bay_width: int = 800,
                          aisle_width: int = 4000,
                          num_movements: int = 1000,
                          num_ordercode: int = 800,
                          average_time_between_movements: float = 1 / 24,  # days
                          first_day: datetime.datetime = datetime.datetime(year=2020, month=1, day=2),
                          ):
    """
    Generate sample warehouse picking data

    Args:
        num_SKUs (int, optional): Number of SKUs of the Warehouse. Defaults to 100.
        nodecode (int, optional): Nodecode of the Warehouse. Defaults to 1.
        idwh (list, optional): List of logical clusters of the warehouse. Defaults to ['LOGICAL_WH1', 'LOGICAL_WH2', 'FAKE'].
        whsubarea (list, optional): List of physical areas of the warehouse. Defaults to ['AREA 1'].
        num_aisles (int, optional): Number of aisles of the warehouse. Defaults to 5.
        num_bays (int, optional): Number of bays pof the warheouse. Defaults to 66.
        num_levels (int, optional): Number of levels of the warehouse. Defaults to 5.
        level_height (int, optional): Height of a level. Defaults to 1200.
        bay_width (int, optional): Width of a bay. Defaults to 800.
        aisle_width (int, optional): Width of an Aisle. Defaults to 4000.
        num_movements (int, optional): Number of movements to generate. Defaults to 1000.
        num_ordercode (int, optional): Number of picking lists to generate. Defaults to 800.
        average_time_between_movements (float, optional): Average waiting time between movements. Defaults to 1 / 24.
        first_day (datetime.datetime, optional): First day of the picking list. Defaults to datetime.datetime(year=2020, month=1, day=2).

    Returns:
        D_locations (pd.dataFrame): Output DataFrame with storage locations.
        D_SKUs (pd.dataFrame): Output DataFrame with SKUs.
        D_movements (pd.dataFrame): Output DataFrame with movements.
        D_inventory (pd.dataFrame): Output DataFrame with inventory values.

    """

    class SKU():
        def __init__(self, itemcode: str):
            self.ITEMCODE = itemcode
            self.DESCRIPTION = f"PRODOTTO_{itemcode}"
            self.VOLUME = np.random.uniform(0.1, 100)  # volume in dm3
            self.WEIGHT = np.random.uniform(0.1, 10)  # weigth in Kg

    class STORAGE_LOCATION():
        def __init__(self, nodecode, idwh, whsubarea, idlocation,
                     loccodex, loccodey, loccodez, rack, bay, level):

            self.NODECODE = nodecode
            self.IDWH = idwh
            self.WHSUBAREA = whsubarea
            self.IDLOCATION = idlocation
            self.LOCCODEX = loccodex
            self.LOCCODEY = loccodey
            self.LOCCODEZ = loccodez
            self.RACK = rack
            self.BAY = bay
            self.LEVEL = level

    class MOVEMENTS():
        def __init__(self, itemcode, volume, weight, nodecode, idwh, whsubarea, idlocation,
                     rack, bay, level, loccodex, loccodey, loccodez,
                     ordercode, quantity, timestamp, inout, ordertype):
            self.ITEMCODE = itemcode
            self.NODECODE = nodecode
            self.IDWH = idwh
            self.WHSUBAREA = whsubarea
            self.IDLOCATION = idlocation
            self.RACK = rack
            self.BAY = bay
            self.LEVEL = level
            self.LOCCODEX = loccodex
            self.LOCCODEY = loccodey
            self.LOCCODEZ = loccodez
            self.ORDERCODE = ordercode
            self.PICKINGLIST = ordercode
            self.QUANTITY = quantity
            self.VOLUME = volume * quantity
            self.WEIGHT = weight * quantity
            self.TIMESTAMP_IN = timestamp
            self.INOUT = inout
            self.ORDERTYPE = ordertype

    class INVENTORY():
        def __init__(self, itemcode, nodecode, idwh, idlocation, quantity, timestamp):
            self.NODECODE = nodecode
            self.IDWH = idwh
            self.ITEMCODE = itemcode
            self.IDLOCATION = idlocation
            self.QUANTITY = quantity
            self.TIMESTAMP = timestamp

    dict_SKUs = {}
    itemcodes = np.arange(0, num_SKUs)
    for itemcode in itemcodes:
        dict_SKUs[itemcode] = SKU(itemcode)

    # % CREATE WH LAYOUT
    dict_locations = {}
    idlocation = 0

    for corsia in range(0, num_aisles):
        for campata in range(0, num_bays):
            for livello in range(0, num_levels):
                idlocation = idlocation + 1  # create a new location index

                # save parameters
                NODECODE = nodecode
                IDWH = random.choice(idwh)
                WHSUBAREA = random.choice(whsubarea)
                IDLOCATION = idlocation
                LOCCODEX = corsia * aisle_width
                LOCCODEY = campata * bay_width
                LOCCODEZ = livello * level_height

                # create storage location
                dict_locations[idlocation] = STORAGE_LOCATION(NODECODE,
                                                              IDWH,
                                                              WHSUBAREA,
                                                              IDLOCATION,
                                                              LOCCODEX,
                                                              LOCCODEY,
                                                              LOCCODEZ,
                                                              corsia,
                                                              campata,
                                                              livello)
    # create movements
    dict_movements = {}
    num_creati = 0
    ordercodes = np.arange(0, num_ordercode)

    while num_creati < num_movements:
        num_creati = num_creati + 1

        # random select sku
        sku = random.choice(dict_SKUs)
        itemcode = sku.ITEMCODE
        volume = sku.VOLUME
        weight = sku.WEIGHT

        # random select storage location
        loc_key = random.choice(list(dict_locations.keys()))
        loc = dict_locations[loc_key]
        nodecode = loc.NODECODE
        idwh = loc.IDWH
        whsubarea = loc.WHSUBAREA
        idlocation = loc.IDLOCATION
        loccodex = loc.LOCCODEX
        loccodey = loc.LOCCODEY
        loccodez = loc.LOCCODEZ
        rack = loc.RACK
        bay = loc.BAY
        level = loc.LEVEL

        # generates movements data
        ordercode = random.choice(ordercodes)
        quantity = np.random.lognormal(mean=2, sigma=1)
        wait = np.random.exponential(average_time_between_movements)
        if num_creati == 1:
            timestamp = first_day + datetime.timedelta(wait)
        else:
            timestamp = dict_movements[num_creati - 1].TIMESTAMP_IN + datetime.timedelta(wait)

        inout = random.choice(['+', '-', ' '])
        ordertype = random.choice(['PICKING', 'PUTAWAY', ' OTHER '])
        dict_movements[num_creati] = MOVEMENTS(itemcode, volume, weight, nodecode, idwh, whsubarea, idlocation,
                                               rack, bay, level, loccodex, loccodey, loccodez,
                                               ordercode, quantity, timestamp, inout, ordertype)

    # create inventory
    dict_inventory = {}
    for itemcode in dict_SKUs:
        # sku = dict_SKUs[itemcode]

        loc_key = random.choice(list(dict_locations.keys()))
        loc = dict_locations[loc_key]
        nodecode = loc.NODECODE
        idwh = loc.IDWH
        idlocation = loc.IDLOCATION
        quantity = np.random.lognormal(mean=2, sigma=1)
        dict_inventory[itemcode] = INVENTORY(itemcode, nodecode, idwh, idlocation, quantity, first_day)

    # save locations and export
    D_locations = pd.DataFrame()
    for loc in dict_locations:
        D_locations = D_locations.append(pd.DataFrame([vars(dict_locations[loc])]))

    # save skus
    D_SKUs = pd.DataFrame()
    for sku in dict_SKUs:
        D_SKUs = D_SKUs.append(pd.DataFrame([vars(dict_SKUs[sku])]))

    # save movements
    D_movements = pd.DataFrame()
    for mov in dict_movements:
        D_movements = D_movements.append(pd.DataFrame([vars(dict_movements[mov])]))

    # save inventory
    D_inventory = pd.DataFrame()
    for inv in dict_inventory:
        D_inventory = D_inventory.append(pd.DataFrame([vars(dict_inventory[inv])]))

    return D_locations, D_SKUs, D_movements, D_inventory
