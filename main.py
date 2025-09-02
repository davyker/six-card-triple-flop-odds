#!/usr/bin/env python3
"""
Main entry point for Six-Card Triple Flop poker simulator.
"""

import argparse
import sys
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from simulator import TripleFlipMonteCarloSimulator
from card_types import SimulationConfig, Card, Rank, Suit
from hand_evaluator import vector_to_hand_strength


def parse_card(card_str: str) -> tuple[int, int]:
    """
    Parse a card string like 'As' or 'KH' into (rank, suit).
    
    Args:
        card_str: Card string (e.g., 'As', '2d', 'Tc')
    
    Returns:
        Tuple of (rank, suit) as integers
    """
    card_str = card_str.strip().upper()
    if len(card_str) != 2:
        raise ValueError(f"Invalid card format: {card_str}")
    
    rank_char = card_str[0]
    suit_char = card_str[1]
    
    # Parse rank
    rank_map = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'T': 10, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    if rank_char not in rank_map:
        raise ValueError(f"Invalid rank: {rank_char}")
    rank = rank_map[rank_char]
    
    # Parse suit
    suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}
    if suit_char not in suit_map:
        raise ValueError(f"Invalid suit: {suit_char}")
    suit = suit_map[suit_char]
    
    return rank, suit


def parse_player_cards(cards_str: str) -> np.ndarray:
    """
    Parse player cards from string format.
    
    Args:
        cards_str: Comma-separated hands, e.g., "AsKsQsJsTs9s,2c2d2h2s3c3d"
                  or with spaces: "As Ks Qs Js Ts 9s, 2c 2d 2h 2s 3c 3d"
    
    Returns:
        Array of shape (num_players, 6, 2)
    """
    hands = cards_str.split(',')
    num_players = len(hands)
    player_cards = np.zeros((num_players, 6, 2), dtype=np.int32)
    
    for i, hand_str in enumerate(hands):
        # Remove all spaces
        hand_str = hand_str.replace(' ', '')
        
        # Split into individual cards (every 2 characters)
        cards = []
        j = 0
        while j < len(hand_str):
            if j + 1 < len(hand_str):
                cards.append(hand_str[j:j+2])
                j += 2
            else:
                raise ValueError(f"Invalid hand format for player {i}: {hand_str}")
        
        if len(cards) != 6:
            raise ValueError(f"Player {i} must have exactly 6 cards, got {len(cards)}")
        
        for j, card_str in enumerate(cards):
            rank, suit = parse_card(card_str)
            player_cards[i, j] = [rank, suit]
    
    return player_cards


def parse_boards(boards_str: str) -> np.ndarray:
    """
    Parse board cards from string format.
    
    Args:
        boards_str: Comma-separated boards, e.g., "AhKhQh,2s3s4s,7d8d9d"
                   or with spaces: "Ah Kh Qh, 2s 3s 4s, 7d 8d 9d"
    
    Returns:
        Array of shape (3, 5, 2)
    """
    board_strs = boards_str.split(',')
    
    if len(board_strs) != 3:
        raise ValueError(f"Must specify exactly 3 boards, got {len(board_strs)}")
    
    boards = np.zeros((3, 5, 2), dtype=np.int32)
    
    for i, board_str in enumerate(board_strs):
        # Remove all spaces
        board_str = board_str.replace(' ', '')
        
        if not board_str:  # Empty board
            continue
            
        # Split into individual cards
        cards = []
        j = 0
        while j < len(board_str):
            if j + 1 < len(board_str):
                cards.append(board_str[j:j+2])
                j += 2
            else:
                raise ValueError(f"Invalid board format for board {i}: {board_str}")
        
        for j, card_str in enumerate(cards):
            rank, suit = parse_card(card_str)
            boards[i, j] = [rank, suit]
    
    return boards


def generate_random_game(num_players: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a random game setup.
    
    Args:
        num_players: Number of players (max 6)
    
    Returns:
        Tuple of (player_cards, boards)
    """
    # Create deck
    deck = []
    for rank in range(2, 15):
        for suit in range(1, 5):
            deck.append([rank, suit])
    
    # Shuffle
    np.random.shuffle(deck)
    
    # Deal player cards (6 each)
    player_cards = np.zeros((num_players, 6, 2), dtype=np.int32)
    for i in range(num_players):
        for j in range(6):
            player_cards[i, j] = deck.pop()
    
    # Create empty boards (preflop)
    boards = np.zeros((3, 5, 2), dtype=np.int32)
    
    return player_cards, boards


def print_game_state(player_cards: np.ndarray, boards: np.ndarray):
    """Print the current game state in a readable format."""
    rank_names = {2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 
                  10:'T', 11:'J', 12:'Q', 13:'K', 14:'A'}
    suit_names = {1:'s', 2:'h', 3:'d', 4:'c'}
    
    print("\n" + "="*50)
    print("GAME STATE")
    print("="*50)
    
    # Print players
    num_players = player_cards.shape[0]
    for i in range(num_players):
        print(f"Player {i}: ", end="")
        for j in range(6):
            rank = player_cards[i, j, 0]
            suit = player_cards[i, j, 1]
            if rank > 0:
                print(f"{rank_names[rank]}{suit_names[suit]} ", end="")
        print()
    
    # Print boards
    print("\nBoards:")
    for i in range(3):
        print(f"  Board {i}: ", end="")
        cards_shown = 0
        for j in range(5):
            rank = boards[i, j, 0]
            suit = boards[i, j, 1]
            if rank > 0:
                print(f"{rank_names[rank]}{suit_names[suit]} ", end="")
                cards_shown += 1
        if cards_shown == 0:
            print("(empty)", end="")
        print()
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Six-Card Triple Flop Poker Monte Carlo Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --num-players 3                    # Random 3-player game
  %(prog)s --num-players 2 --max-iter 100000  # 2 players, more iterations
  %(prog)s --num-players 4 --threshold 0.00001 # Tighter convergence
  %(prog)s --num-players 3 --verbose          # Show iteration details
        """
    )
    
    # Game setup
    parser.add_argument(
        '-p', '--num-players',
        type=int,
        default=2,
        help='Number of players (2-6, default: 2). Ignored if --cards is provided. If --cards not provided, num_players random player hands are generated'
    )
    parser.add_argument(
        '-c', '--cards',
        type=str,
        help='Player cards as comma-separated hands, e.g., "AsKsQsJsTs9s,2c2d2h2s3c3d"'
    )
    parser.add_argument(
        '-b', '--boards',
        type=str,
        help='Board cards comma-separated, e.g., "AhKhQh,2s3s4s,7d8d9d" for flop or "AhKhQhJh,2s3s4s5s,7d8d9dTd" for turn'
    )
    
    # Simulation parameters
    parser.add_argument(
        '-m', '--max-iterations',
        type=int,
        default=100000,
        help='Max iterations (default: 100000)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.002,
        help='Standard error convergence threshold (default: 0.0001)'
    )
    
    # Output options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed iteration info'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting convergence graphs'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if args.threshold <= 0 or args.threshold >= 1:
        print(f"Error: Convergence threshold must be between 0 and 1, got {args.threshold}")
        sys.exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Parse or generate player cards
    if args.cards:
        print("\nParsing player cards...")
        player_cards = parse_player_cards(args.cards)
        num_players = player_cards.shape[0]
        print(f"Parsed {num_players} players")
    else:
        print(f"\nGenerating random {args.num_players}-player game...")
        player_cards, _ = generate_random_game(args.num_players)
        num_players = args.num_players
    
    # Parse or generate boards
    if args.boards:
        print("Parsing boards...")
        boards = parse_boards(args.boards)
        cards_on_board = np.sum(boards[0, :, 0] > 0)
        round_name = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[cards_on_board]
        print(f"Starting from {round_name}")
    else:
        # Start from preflop
        boards = np.zeros((3, 5, 2), dtype=np.int32)
        print("Starting from preflop")
    
    # Print game state
    print_game_state(player_cards, boards)
    
    # Create configuration
    config = SimulationConfig(
        convergence_threshold=args.threshold,
        print_iterations=args.verbose,
        plot_results=not args.no_plot,
        max_iterations=args.max_iterations
    )
    
    # Run simulation
    print("Starting simulation...")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Convergence threshold: {args.threshold} (standard error)")
    print()
    
    simulator = TripleFlipMonteCarloSimulator(player_cards, boards, config)
    results = simulator.run_simulation()
    
    # Print results
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Converged: {'Yes' if results.iteration_count < config.max_iterations else 'No'}")
    print(f"Iterations: {results.iteration_count}")
    print(f"Time: {results.total_time_seconds:.2f} seconds")
    print(f"Speed: {results.iterations_per_second:.0f} iterations/second")
    print()
    
    print("Final Equities:")
    final_equities = results.player_equities[:, -1]
    for i in range(num_players):
        equity = final_equities[i]
        print(f"  Player {i}: {equity:.4f} ({equity*100:.2f}%)")
    
    # Sanity check
    total_equity = np.sum(final_equities[:num_players])
    if abs(total_equity - 1.0) > 0.001:
        print(f"\nWarning: Equities sum to {total_equity:.4f}, expected 1.0000")
    
    # Plot if requested
    if not args.no_plot:
        # Plot equity convergence
        plt.figure(figsize=(12, 6))
        for i in range(num_players):
            plt.plot(results.player_equities[i, :results.iteration_count+1], 
                    label=f'Player {i}')
        plt.xlabel('Iteration')
        plt.ylabel('Equity')
        plt.title('Equity Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    main()