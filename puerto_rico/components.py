from dataclasses import dataclass, field
from typing import List, Optional
from puerto_rico.constants import TileType, BuildingType, Good

@dataclass
class IslandTile:
    tile_type: TileType
    is_occupied: bool = False

@dataclass
class CityBuilding:
    building_type: BuildingType
    colonists: int = 0
    
    @property
    def is_occupied(self) -> bool:
        return self.colonists > 0

@dataclass
class CargoShip:
    capacity: int
    current_load: int = 0
    good_type: Optional[Good] = None

    @property
    def is_full(self) -> bool:
        return self.current_load >= self.capacity

    @property
    def is_empty(self) -> bool:
        return self.current_load == 0
