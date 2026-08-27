from dataclasses import dataclass, field
from typing import List, Optional
from puerto_rico.constants import TileType, BuildingType, Good

# slots=True: these objects are copied thousands of times per move by planning
# agents (ForwardModel.clone), so smaller instances and faster attribute access
# translate directly into search depth.

@dataclass(slots=True)
class IslandTile:
    tile_type: TileType
    is_occupied: bool = False

    def __deepcopy__(self, memo):
        return IslandTile(self.tile_type, self.is_occupied)

@dataclass(slots=True)
class CityBuilding:
    building_type: BuildingType
    colonists: int = 0

    def __deepcopy__(self, memo):
        return CityBuilding(self.building_type, self.colonists)

    @property
    def is_occupied(self) -> bool:
        return self.colonists > 0

@dataclass(slots=True)
class CargoShip:
    capacity: int
    current_load: int = 0
    good_type: Optional[Good] = None

    def __deepcopy__(self, memo):
        return CargoShip(self.capacity, self.current_load, self.good_type)

    @property
    def is_full(self) -> bool:
        return self.current_load >= self.capacity

    @property
    def is_empty(self) -> bool:
        return self.current_load == 0
