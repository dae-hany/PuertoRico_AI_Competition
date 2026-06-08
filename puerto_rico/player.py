from typing import List, Dict
from puerto_rico.constants import TileType, BuildingType, Good
from puerto_rico.components import IslandTile, CityBuilding

class Player:
    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        self.doubloons = 0
        self.vp_chips = 0
        self.unplaced_colonists = 0
        
        # Trackers
        self.goods_produced = 0
        self.doubloons_earned = 0
        
        # 12 Island Spaces
        self.island_board: List[IslandTile] = []
        
        # 12 City Spaces
        self.city_board: List[CityBuilding] = []
        
        # Goods on windrose
        self.goods: Dict[Good, int] = {
            Good.COFFEE: 0,
            Good.TOBACCO: 0,
            Good.CORN: 0,
            Good.SUGAR: 0,
            Good.INDIGO: 0
        }

    @property
    def empty_island_spaces(self) -> int:
        return 12 - len(self.island_board)

    @property
    def empty_city_spaces(self) -> int:
        return 12 - len(self.city_board)

    def _is_large_building(self, b_type: BuildingType) -> bool:
        from puerto_rico.constants import BUILDING_DATA
        return BUILDING_DATA[b_type][4] # is_large boolean at index 4

    def add_doubloons(self, amount: int):
        self.doubloons += amount
        self.doubloons_earned += amount
        
    def pay_doubloons(self, amount: int):
        if amount > self.doubloons:
            raise ValueError("Not enough doubloons")
        self.doubloons -= amount

    def add_vp(self, amount: int):
        self.vp_chips += amount

    def add_good(self, good: Good, amount: int = 1):
        self.goods[good] += amount
        self.goods_produced += amount

    def remove_good(self, good: Good, amount: int = 1):
        if amount > self.goods[good]:
            raise ValueError(f"Not enough {good.name}")
        self.goods[good] -= amount

    def place_plantation(self, tile_type: TileType):
        if self.empty_island_spaces <= 0:
            raise ValueError("No empty island spaces")
        self.island_board.append(IslandTile(tile_type=tile_type, is_occupied=False))

    def build_building(self, building_type: BuildingType):
        spaces_needed = 2 if self._is_large_building(building_type) else 1
        if self.empty_city_spaces < spaces_needed:
            raise ValueError("Not enough empty city spaces")
        self.city_board.append(CityBuilding(building_type=building_type, colonists=0))
        if spaces_needed == 2:
            self.city_board.append(CityBuilding(building_type=BuildingType.OCCUPIED_SPACE, colonists=0))

    def has_building(self, building_type: BuildingType) -> bool:
        return any(b.building_type == building_type for b in self.city_board)

    def is_building_occupied(self, building_type: BuildingType) -> bool:
        for b in self.city_board:
            if b.building_type == building_type:
                return b.is_occupied
        return False
    
    @property
    def total_colonists_owned(self) -> int:
        count = self.unplaced_colonists
        count += sum(1 for t in self.island_board if t.is_occupied)
        count += sum(b.colonists for b in self.city_board)
        return count

    def recall_all_colonists(self):
        """Moves all colonists from board to unplaced_colonists."""
        total = self.total_colonists_owned
        for t in self.island_board:
            t.is_occupied = False
        for b in self.city_board:
            b.colonists = 0
        self.unplaced_colonists = total
