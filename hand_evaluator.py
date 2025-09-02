"""
Hand evaluation logic for Six-Card Triple Flop poker.
"""

from itertools import combinations
from typing import Tuple
import numpy as np
import numpy.typing as npt
from numba import jit


@jit(nopython=True)
def count_pairs(hand: npt.NDArray[np.int32]) -> Tuple[int, npt.NDArray[np.int32]]:
    """
    Count matching card pairs in a 5-card hand to identify poker hand types.
    
    Args:
        hand: (5,2) array where each row is [rank, suit]
    
    Returns:
        (pair_count, pair_counts_by_rank): Total pairs and per-rank counts
    """
    pair_counts_by_rank = np.zeros(15, dtype=np.int32)
    for i in range(4):
        for j in range(i + 1, 5):
            if hand[i, 0] == hand[j, 0]:
                pair_counts_by_rank[hand[i, 0]] += 1
    pair_count = np.sum(pair_counts_by_rank)
    return pair_count, pair_counts_by_rank


@jit(nopython=True)
def evaluate_hand(hand: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """
    Evaluate a 5-card poker hand.
    
    Args:
        hand: (5,2) array where each row is [rank, suit]
    
    Returns:
        Array of 6 integers: [hand_rank, tiebreaker1, tiebreaker2, ...]
    """
    pair_count, pair_counts_by_rank = count_pairs(hand)
    result = np.zeros(6, dtype=np.int32)
    
    if pair_count == 0:  # High card, straight, flush, or straight flush
        ranks = np.sort(hand[:, 0])
        suits = hand[:, 1]
        
        # Check for straight
        is_straight = False
        straight_high = 0
        
        # Regular straight check
        if (ranks[0] + 4 == ranks[1] + 3 == ranks[2] + 2 == ranks[3] + 1 == ranks[4]):
            is_straight = True
            straight_high = ranks[4]
        # Wheel straight (A-2-3-4-5)
        elif ranks[0] == 2 and ranks[1] == 3 and ranks[2] == 4 and ranks[3] == 5 and ranks[4] == 14:
            is_straight = True
            straight_high = 5
        
        # Check for flush
        is_flush = (suits[0] == suits[1] == suits[2] == suits[3] == suits[4])
        
        if is_straight and is_flush:
            result[0] = 9  # Straight flush
            result[1] = straight_high
        elif is_flush:
            result[0] = 6  # Flush
            # Fill high cards in descending order
            descending_ranks = ranks[::-1]
            for i in range(5):
                result[i + 1] = descending_ranks[i]
        elif is_straight:
            result[0] = 5  # Straight
            result[1] = straight_high
        else:
            result[0] = 1  # High card
            # Fill high cards in descending order
            descending_ranks = ranks[::-1]
            for i in range(5):
                result[i + 1] = descending_ranks[i]
    
    elif pair_count == 1:  # One pair
        result[0] = 2
        pair_rank = np.argmax(pair_counts_by_rank)
        result[1] = pair_rank

        # Get kickers
        kickers = hand[:, 0][hand[:, 0] != pair_rank]
        kickers = np.sort(kickers)[::-1]
        for i in range(len(kickers)):
            result[i + 2] = kickers[i]
    
    elif pair_count == 2:  # Two pair
        result[0] = 3
        # Find high and low pairs
        pair_ranks = np.where(pair_counts_by_rank > 0)[0]
        high_pair = max(pair_ranks)
        low_pair = min(pair_ranks)
        result[1] = high_pair
        result[2] = low_pair
        
        # Get kicker
        kicker = hand[:, 0][(hand[:, 0] != high_pair) & (hand[:, 0] != low_pair)][0]
        result[3] = kicker
    
    elif pair_count == 3:  # Three of a kind
        result[0] = 4
        trips_rank = np.argmax(pair_counts_by_rank)
        result[1] = trips_rank
        
        # Get kickers
        kickers = hand[:, 0][hand[:, 0] != trips_rank]
        kickers = np.sort(kickers)[::-1]
        for i in range(len(kickers)):
            result[i + 2] = kickers[i]
    
    elif pair_count == 4:  # Full house
        result[0] = 7
        trips_rank = np.where(pair_counts_by_rank == 3)[0][0]
        pair_rank = np.where(pair_counts_by_rank == 1)[0][0]
        result[1] = trips_rank
        result[2] = pair_rank
    
    else:  # pair_count == 6, Four of a kind
        result[0] = 8
        quads_rank = np.argmax(pair_counts_by_rank)
        result[1] = quads_rank
        
        # Get kicker
        kicker = hand[:, 0][hand[:, 0] != quads_rank][0]
        result[2] = kicker
    
    return result

@jit(nopython=True)
def find_best_six_card_triple_flop_hand(
    player_cards: npt.NDArray[np.int32],
    board_cards: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """
    Find the best 5-card hand using 2 from player's 6 cards and 3 from board.
    
    Args:
        player_cards: (6,2) array of player's cards
        board_cards: (5,2) array of board cards
    
    Returns:
        Array of 6 integers representing best hand strength
    """
    best_hand = np.zeros(6, dtype=np.int32)
    
    # Generate all combinations using indices
    for i in range(5):
        for j in range(i + 1, 6):
            for x in range(3):
                for y in range(x + 1, 4):
                    for z in range(y + 1, 5):
                        # Build 5-card hand
                        hand = np.zeros((5, 2), dtype=np.int32)
                        hand[0] = player_cards[i]
                        hand[1] = player_cards[j]
                        hand[2] = board_cards[x]
                        hand[3] = board_cards[y]
                        hand[4] = board_cards[z]
                        
                        # Evaluate and compare
                        strength = evaluate_hand(hand)
                        
                        # Compare lexicographically
                        for k in range(6):
                            if strength[k] > best_hand[k]:
                                best_hand = strength
                                break
                            elif strength[k] < best_hand[k]:
                                break
    
    return best_hand

