from enum import IntEnum, auto

class Phase(IntEnum):
    SETTLER = 0
    MAYOR = 1
    BUILDER = 2
    CRAFTSMAN = 3
    TRADER = 4
    CAPTAIN = 5
    CAPTAIN_STORE = 6
    PROSPECTOR = 7
    END_ROUND = 8

class Role(IntEnum):
    SETTLER = 0
    MAYOR = 1
    BUILDER = 2
    CRAFTSMAN = 3
    TRADER = 4
    CAPTAIN = 5
    PROSPECTOR_1 = 6
    PROSPECTOR_2 = 7

class Good(IntEnum):
    COFFEE = 0
    TOBACCO = 1
    CORN = 2
    SUGAR = 3
    INDIGO = 4

class TileType(IntEnum):
    # Goods
    COFFEE_PLANTATION = 0
    TOBACCO_PLANTATION = 1
    CORN_PLANTATION = 2
    SUGAR_PLANTATION = 3
    INDIGO_PLANTATION = 4
    # Special
    QUARRY = 5
    EMPTY = 6

class BuildingType(IntEnum):
    # Small Production (Cost 1-3, VP 1, Capacity 1, Count 4)
    SMALL_INDIGO_PLANT = 0
    SMALL_SUGAR_MILL = 1
    
    # Large Production (Cost 3-6, VP 2-3, Capacity 2-3, Count 3)
    INDIGO_PLANT = 2
    SUGAR_MILL = 3
    TOBACCO_STORAGE = 4
    COFFEE_ROASTER = 5
    
    # Small Violet (Cost 1-9, VP 1-3, Capacity 1, Count 2)
    SMALL_MARKET = 6
    HACIENDA = 7
    CONSTRUCTION_HUT = 8
    SMALL_WAREHOUSE = 9
    HOSPICE = 10
    OFFICE = 11
    LARGE_MARKET = 12
    LARGE_WAREHOUSE = 13
    FACTORY = 14
    UNIVERSITY = 15
    HARBOR = 16
    WHARF = 17
    
    # Large Violet (Cost 10, VP 4, Capacity 1, Size 2, Count 1)
    GUILDHALL = 18
    RESIDENCE = 19
    FORTRESS = 20
    CUSTOMS_HOUSE = 21
    CITY_HALL = 22
    
    EMPTY = 23
    OCCUPIED_SPACE = 24

# Default counts for plantations
PLANTATION_COUNTS = {
    TileType.COFFEE_PLANTATION: 8,
    TileType.TOBACCO_PLANTATION: 9,
    TileType.CORN_PLANTATION: 10,
    TileType.SUGAR_PLANTATION: 11,
    TileType.INDIGO_PLANTATION: 12,
}

QUARRY_COUNT = 8

# Setup counts based on number of players
VP_CHIPS_SETUP = {2: 65, 3: 75, 4: 100, 5: 122}
COLONIST_SHIP_SETUP = {2: 2, 3: 3, 4: 4, 5: 5}
COLONIST_SUPPLY_SETUP = {2: 40, 3: 55, 4: 75, 5: 95}
STARTING_DOUBLOONS = {2: 3, 3: 2, 4: 3, 5: 4}

CARGO_SHIPS_SETUP = {
    2: [4, 6],
    3: [4, 5, 6],
    4: [5, 6, 7],
    5: [6, 7, 8]
}

# (Cost, VP, Capacity, Max Count, is_large, Good produced)
BUILDING_DATA = {
    BuildingType.SMALL_INDIGO_PLANT: (1, 1, 1, 4, False, Good.INDIGO),
    BuildingType.SMALL_SUGAR_MILL: (2, 1, 1, 4, False, Good.SUGAR),
    BuildingType.INDIGO_PLANT: (3, 2, 3, 3, False, Good.INDIGO),
    BuildingType.SUGAR_MILL: (4, 2, 3, 3, False, Good.SUGAR),
    BuildingType.TOBACCO_STORAGE: (5, 3, 3, 3, False, Good.TOBACCO),
    BuildingType.COFFEE_ROASTER: (6, 3, 2, 3, False, Good.COFFEE),
    BuildingType.SMALL_MARKET: (1, 1, 1, 2, False, None),
    BuildingType.HACIENDA: (2, 1, 1, 2, False, None),
    BuildingType.CONSTRUCTION_HUT: (2, 1, 1, 2, False, None),
    BuildingType.SMALL_WAREHOUSE: (3, 1, 1, 2, False, None),
    BuildingType.HOSPICE: (4, 2, 1, 2, False, None),
    BuildingType.OFFICE: (5, 2, 1, 2, False, None),
    BuildingType.LARGE_MARKET: (5, 2, 1, 2, False, None),
    BuildingType.LARGE_WAREHOUSE: (6, 2, 1, 2, False, None),
    BuildingType.FACTORY: (7, 3, 1, 2, False, None),
    BuildingType.UNIVERSITY: (8, 3, 1, 2, False, None),
    BuildingType.HARBOR: (8, 3, 1, 2, False, None),
    BuildingType.WHARF: (9, 3, 1, 2, False, None),
    BuildingType.GUILDHALL: (10, 4, 1, 1, True, None),
    BuildingType.RESIDENCE: (10, 4, 1, 1, True, None),
    BuildingType.FORTRESS: (10, 4, 1, 1, True, None),
    BuildingType.CUSTOMS_HOUSE: (10, 4, 1, 1, True, None),
    BuildingType.CITY_HALL: (10, 4, 1, 1, True, None),
    BuildingType.OCCUPIED_SPACE: (0, 0, 0, 0, False, None)
}

# Selling prices for goods at trading house
GOOD_PRICES = {
    Good.CORN: 0,
    Good.INDIGO: 1,
    Good.SUGAR: 2,
    Good.TOBACCO: 3,
    Good.COFFEE: 4
}

# Mayor Phase Strategy Enum
class MayorStrategy(IntEnum):
    CAPTAIN_FOCUS = 0        # Shipping strategy
    TRADE_FACTORY_FOCUS = 1  # Trading + Factory strategy
    BUILDING_FOCUS = 2       # Infrastructure strategy

# Large VP Buildings (always highest priority)
LARGE_VP_BUILDINGS = [
    BuildingType.GUILDHALL,
    BuildingType.RESIDENCE,
    BuildingType.FORTRESS,
    BuildingType.CUSTOMS_HOUSE,
    BuildingType.CITY_HALL,
]

# Strategy-specific building priorities
MAYOR_STRATEGY_BUILDINGS = {
    MayorStrategy.CAPTAIN_FOCUS: [
        BuildingType.WHARF,
        BuildingType.HARBOR,
        BuildingType.LARGE_WAREHOUSE,
        BuildingType.SMALL_WAREHOUSE,
    ],
    MayorStrategy.TRADE_FACTORY_FOCUS: [
        BuildingType.OFFICE,
        BuildingType.LARGE_MARKET,
        BuildingType.SMALL_MARKET,
        BuildingType.FACTORY,
    ],
    MayorStrategy.BUILDING_FOCUS: [
        BuildingType.UNIVERSITY,
        BuildingType.HOSPICE,
        BuildingType.CONSTRUCTION_HUT,
        BuildingType.HACIENDA,
    ],
}

# Production building to Good mapping (for production pairs)
PRODUCTION_BUILDINGS = {
    Good.COFFEE: [BuildingType.COFFEE_ROASTER],
    Good.TOBACCO: [BuildingType.TOBACCO_STORAGE],
    Good.SUGAR: [BuildingType.SUGAR_MILL, BuildingType.SMALL_SUGAR_MILL],
    Good.INDIGO: [BuildingType.INDIGO_PLANT, BuildingType.SMALL_INDIGO_PLANT],
}

# Plantation to Good mapping
PLANTATION_TO_GOOD = {
    TileType.COFFEE_PLANTATION: Good.COFFEE,
    TileType.TOBACCO_PLANTATION: Good.TOBACCO,
    TileType.SUGAR_PLANTATION: Good.SUGAR,
    TileType.INDIGO_PLANTATION: Good.INDIGO,
    TileType.CORN_PLANTATION: Good.CORN,
}

# Good value order (descending) for production pair priority
GOOD_VALUE_ORDER = [Good.COFFEE, Good.TOBACCO, Good.SUGAR, Good.INDIGO, Good.CORN]
