"""
Data structures and types for Six-Card Triple Flop poker simulator.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt


class Rank(IntEnum):
    """Card ranks from 2 to Ace."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    
    def __str__(self):
        if self == Rank.ACE:
            return 'A'
        elif self == Rank.KING:
            return 'K'
        elif self == Rank.QUEEN:
            return 'Q'
        elif self == Rank.JACK:
            return 'J'
        elif self == Rank.TEN:
            return 'T'
        else:
            return str(self.value)


class Suit(IntEnum):
    """Card suits."""
    SPADES = 1
    HEARTS = 2
    DIAMONDS = 3
    CLUBS = 4
    
    def __str__(self):
        return {
            Suit.SPADES: 's',
            Suit.HEARTS: 'h',
            Suit.DIAMONDS: 'd',
            Suit.CLUBS: 'c'
        }[self]


class HandRank(IntEnum):
    """Poker hand rankings from high card to straight flush."""
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    TRIPS = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    QUADS = 8
    STRAIGHT_FLUSH = 9
    
    def __str__(self):
        return self.name.replace('_', ' ').title()


@dataclass
class Card:
    """Represents a playing card."""
    rank: Rank
    suit: Suit
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def __lt__(self, other):
        """Compare cards by rank first, then suit."""
        if isinstance(other, Card):
            if self.rank != other.rank:
                return self.rank < other.rank
            return self.suit < other.suit
        return NotImplemented


@dataclass
class HandStrength:
    """
    Represents the strength of a poker hand for comparison.
    
    The hand is encoded as a vector where the first element is the hand rank
    and subsequent elements are tiebreakers in order of importance.
    """
    rank: HandRank
    tiebreakers: List[int]
    
    def to_vector(self) -> List[int]:
        """Convert to comparison vector format."""
        return [self.rank.value] + self.tiebreakers
    
    def __lt__(self, other: 'HandStrength') -> bool:
        return self.to_vector() < other.to_vector()
    
    def __le__(self, other: 'HandStrength') -> bool:
        return self.to_vector() <= other.to_vector()
    
    def __gt__(self, other: 'HandStrength') -> bool:
        return self.to_vector() > other.to_vector()
    
    def __ge__(self, other: 'HandStrength') -> bool:
        return self.to_vector() >= other.to_vector()
    
    def __eq__(self, other: 'HandStrength') -> bool:
        return self.to_vector() == other.to_vector()
    
    def __str__(self):
        vector = self.to_vector()
        
        # Format the hand name
        if self.rank == HandRank.HIGH_CARD:
            name = f"High Card, {Rank(self.tiebreakers[0])} high"
        elif self.rank == HandRank.PAIR:
            name = f"Pair of {Rank(self.tiebreakers[0])}s"
        elif self.rank == HandRank.TWO_PAIR:
            name = f"Two Pair, {Rank(self.tiebreakers[0])}s and {Rank(self.tiebreakers[1])}s"
        elif self.rank == HandRank.TRIPS:
            name = f"Three of a Kind, {Rank(self.tiebreakers[0])}s"
        elif self.rank == HandRank.STRAIGHT:
            name = f"Straight to {Rank(self.tiebreakers[0])}"
        elif self.rank == HandRank.FLUSH:
            name = f"Flush, {Rank(self.tiebreakers[0])} high"
        elif self.rank == HandRank.FULL_HOUSE:
            name = f"Full House, {Rank(self.tiebreakers[0])}s full of {Rank(self.tiebreakers[1])}s"
        elif self.rank == HandRank.QUADS:
            name = f"Four of a Kind, {Rank(self.tiebreakers[0])}s"
        elif self.rank == HandRank.STRAIGHT_FLUSH:
            name = f"Straight Flush to {Rank(self.tiebreakers[0])}"
        else:
            name = f"{self.rank}"
        
        return f"{vector} = {name}"


@dataclass
class GameState:
    """Represents the current state of a Six-Card Triple Flop game."""
    players: List[List[Card]]  # Each player has 6 cards
    boards: List[List[Optional[Card]]]  # 3 boards with up to 5 cards each
    
    @property
    def num_players(self) -> int:
        """Get the number of active players."""
        return len([p for p in self.players if p])
    
    @property
    def round_name(self) -> str:
        """Determine the current round based on board state."""
        if not self.boards[0]:
            return "preflop"
        
        cards_on_board = len([c for c in self.boards[0] if c is not None])
        if cards_on_board == 0:
            return "preflop"
        elif cards_on_board == 3:
            return "flop"
        elif cards_on_board == 4:
            return "turn"
        elif cards_on_board == 5:
            return "river"
        else:
            return "unknown"
    
    def to_arrays(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Convert to numpy array format for compatibility with existing code."""
        # Convert players
        max_players = 6
        player_array = np.zeros((max_players, 6, 2), dtype=np.int32)
        for i, player_cards in enumerate(self.players[:max_players]):
            for j, card in enumerate(player_cards[:6]):
                player_array[i, j] = card.to_array()
        
        # Convert boards
        board_array = np.zeros((3, 5, 2), dtype=np.int32)
        for i, board in enumerate(self.boards[:3]):
            for j, card in enumerate(board[:5]):
                if card is not None:
                    board_array[i, j] = card.to_array()
        
        return player_array, board_array


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    convergence_threshold: float = 0.0001
    print_iterations: bool = False
    plot_results: bool = True
    max_iterations: int = 100000


@dataclass
class SimulationResults:
    """Results from a Monte Carlo simulation."""
    player_equities: npt.NDArray[np.float64]  # Shape: (num_players, num_iterations)
    iteration_count: int
    iteration_durations: npt.NDArray[np.float64]
    convergence_history: npt.NDArray[np.float64]
    average_iteration_time_ms: float
    iterations_per_second: float
    total_time_seconds: float
    
    def get_final_equities(self) -> npt.NDArray[np.float64]:
        """Get the final equity estimates for each player."""
        return self.player_equities[:, -1]
    
    def print_summary(self):
        """Print a summary of the simulation results."""
        print("\n" + "="*50)
        print("SIMULATION RESULTS")
        print("="*50)
        print(f"Iterations: {self.iteration_count}")
        print(f"Average iteration time: {self.average_iteration_time_ms:.2f} ms")
        print(f"Iterations per second: {self.iterations_per_second:.2f}")
        print(f"Total time: {self.total_time_seconds:.2f} seconds")
        print("\nFinal Equities:")
        for i, equity in enumerate(self.get_final_equities()):
            if equity > 0:  # Only show active players
                print(f"  Player {i}: {equity:.4f} ({equity*100:.2f}%)")


# Utility functions for creating standard deck
def create_standard_deck() -> List[Card]:
    """Create a standard 52-card deck."""
    deck = []
    for suit in Suit:
        for rank in Rank:
            deck.append(Card(rank, suit))
    return deck


def create_deck_array() -> npt.NDArray[np.int32]:
    """Create deck in numpy array format for compatibility."""
    deck = create_standard_deck()
    return np.array([card.to_array() for card in deck], dtype=np.int32)