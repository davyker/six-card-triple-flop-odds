"""
Monte Carlo simulator for Six-Card Triple Flop poker.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from hand_evaluator import find_best_six_card_triple_flop_hand
from card_types import SimulationConfig, SimulationResults


class EquityHistory:
    """Track full equity history for convergence checking and plotting."""
    
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.data = [np.zeros(num_players)]  # Start with zeros at iteration 0
        self.win_counts = np.zeros(num_players)  # Track total wins for variance calculation
        
    def add(self, equities: npt.NDArray[np.float64]):
        """Add new equity snapshot."""
        self.data.append(equities.copy())
        self.win_counts += equities  # Add to running total
    
    def get_current_estimates(self) -> npt.NDArray[np.float64]:
        """Get current equity estimates."""
        if len(self.data) <= 1:
            return np.zeros(self.num_players)
        return self.data[-1]
    
    def get_standard_errors(self, iteration: int) -> npt.NDArray[np.float64]:
        """Calculate standard error for each player's equity estimate."""
        if iteration == 0:
            return np.ones(self.num_players)  # Return large values initially
        
        # Current equity estimates
        equities = self.win_counts / iteration
        
        # Variance for Bernoulli trials: p(1-p)
        variances = equities * (1 - equities)
        
        # Standard error: sqrt(variance / n)
        std_errors = np.sqrt(variances / iteration)
        
        return std_errors
    
    def max_standard_error(self, iteration: int) -> float:
        """Get the maximum standard error across all players."""
        if iteration == 0:
            return 1.0
        std_errors = self.get_standard_errors(iteration)
        return np.max(std_errors)
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert to numpy array of shape (num_players, num_iterations+1)."""
        return np.array(self.data).T


class TripleFlipMonteCarloSimulator:
    """
    Monte Carlo simulator for Six-Card Triple Flop poker variant.
    
    This variant features:
    - Each player gets 6 cards (instead of 2)
    - 3 separate boards are dealt
    - Players make best hand using exactly 2 of their cards + 3 from each board
    - Winner on each board gets 1 point
    - Player with most total points wins the pot
    """
    
    def __init__(
        self, 
        player_cards: npt.NDArray[np.int32],
        initial_boards: npt.NDArray[np.int32],
        config: Optional[SimulationConfig] = None
    ):
        """
        Initialize the simulator.
        
        Args:
            player_cards: Array of shape (num_players, 6, 2) with player cards
                         where num_players is 2-6
            initial_boards: Array of shape (3, 5, 2) with board cards
                           Undealt cards are zeros
            config: Simulation configuration
        """
        # Validate inputs
        self._validate_inputs(player_cards, initial_boards)
        
        self.player_cards = player_cards
        self.initial_boards = initial_boards.copy()
        self.config = config or SimulationConfig()
        
        # Get number of players from shape
        self.num_players = player_cards.shape[0]
        
        # Setup deck
        self.full_deck = self._create_full_deck()
        self.available_indices = self._get_available_card_indices()
        
        # Determine cards to deal
        self.cards_to_deal_per_board = self._count_cards_to_deal()
        self.total_cards_to_deal = 3 * self.cards_to_deal_per_board
        
        # Tracking
        self.equity_history = EquityHistory(self.num_players)
        self.iteration_times = []
    
    def _validate_inputs(
        self, 
        player_cards: npt.NDArray[np.int32], 
        initial_boards: npt.NDArray[np.int32]
    ):
        """Validate that inputs follow Six-Card Triple Flop rules."""
        # Check player cards shape
        if len(player_cards.shape) != 3 or player_cards.shape[1] != 6 or player_cards.shape[2] != 2:
            raise ValueError(f"player_cards must have shape (num_players, 6, 2), got {player_cards.shape}")
        
        # Check number of players
        num_players = player_cards.shape[0]
        if num_players < 2 or num_players > 6:
            raise ValueError(f"Must have 2-6 players, got {num_players}")
        
        # Check boards shape
        if initial_boards.shape != (3, 5, 2):
            raise ValueError(f"initial_boards must have shape (3, 5, 2), got {initial_boards.shape}")
        
        # Check that all boards have the same number of cards
        cards_per_board = []
        for board_idx in range(3):
            num_cards = np.sum(initial_boards[board_idx, :, 0] > 0)
            cards_per_board.append(num_cards)
        
        if len(set(cards_per_board)) > 1:
            raise ValueError(f"All boards must have the same number of cards, got {cards_per_board}")
        
        # Check valid number of cards on boards
        num_board_cards = cards_per_board[0]
        if num_board_cards not in [0, 3, 4, 5]:
            raise ValueError(f"Boards must have 0 (preflop), 3 (flop), 4 (turn), or 5 (river) cards, got {num_board_cards}")
        
        # Check for duplicate cards
        all_cards = []
        
        # Collect player cards
        for player_idx in range(num_players):
            for card in player_cards[player_idx]:
                if card[0] > 0:  # Valid card (not empty)
                    card_tuple = (int(card[0]), int(card[1]))
                    # Validate card values
                    if card_tuple[0] < 2 or card_tuple[0] > 14:
                        raise ValueError(f"Invalid card rank {card_tuple[0]}, must be 2-14")
                    if card_tuple[1] < 1 or card_tuple[1] > 4:
                        raise ValueError(f"Invalid card suit {card_tuple[1]}, must be 1-4")
                    all_cards.append(card_tuple)
        
        # Collect board cards
        for board_idx in range(3):
            for card in initial_boards[board_idx]:
                if card[0] > 0:  # Valid card (not empty)
                    card_tuple = (int(card[0]), int(card[1]))
                    # Validate card values
                    if card_tuple[0] < 2 or card_tuple[0] > 14:
                        raise ValueError(f"Invalid card rank {card_tuple[0]}, must be 2-14")
                    if card_tuple[1] < 1 or card_tuple[1] > 4:
                        raise ValueError(f"Invalid card suit {card_tuple[1]}, must be 1-4")
                    all_cards.append(card_tuple)
        
        # Check for duplicates
        if len(all_cards) != len(set(all_cards)):
            # Find the duplicate/s
            seen = set()
            for card in all_cards:
                if card in seen:
                    rank_name = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}[card[0]]
                    suit_name = {1:'s',2:'h',3:'d',4:'c'}[card[1]]
                    raise ValueError(f"Duplicate card found: {rank_name}{suit_name}")
                seen.add(card)
        
        # Check that player cards are fully dealt (all 6 cards per player)
        for player_idx in range(num_players):
            player_card_count = np.sum(player_cards[player_idx, :, 0] > 0)
            if player_card_count != 6:
                raise ValueError(f"Player {player_idx} must have exactly 6 cards, got {player_card_count}")
        
    def _create_full_deck(self) -> npt.NDArray[np.int32]:
        """Create a full 52-card deck."""
        deck = []
        for rank in range(2, 15):  # 2 through Ace
            for suit in range(1, 5):  # Spades, Hearts, Diamonds, Clubs
                deck.append([rank, suit])
        return np.array(deck, dtype=np.int32)
    
    def _get_available_card_indices(self) -> npt.NDArray[np.int32]:
        """Get indices of cards not yet dealt."""
        used_cards = set()
        
        # Add player cards
        for player_idx in range(self.num_players):
            for card in self.player_cards[player_idx]:
                if card[0] > 0:  # Valid card
                    used_cards.add(tuple(card))
        
        # Add board cards
        for board in self.initial_boards:
            for card in board:
                if card[0] > 0:  # Valid card
                    used_cards.add(tuple(card))
        
        # Find available indices
        available = []
        for i, card in enumerate(self.full_deck):
            if tuple(card) not in used_cards:
                available.append(i)
        
        return np.array(available, dtype=np.int32)
    
    def _count_cards_to_deal(self) -> int:
        """Count how many cards need to be dealt per board."""
        # Count cards on first board (all boards have same number)
        cards_on_board = np.sum(self.initial_boards[0, :, 0] > 0)
        return 5 - cards_on_board
    
    def run_simulation(self) -> SimulationResults:
        """
        Run the Monte Carlo simulation until convergence or max iterations.
        
        Returns:
            SimulationResults object with equity estimates and metrics
        """
        start_time = time.time()
        iteration = 0
        converged = False
        max_std_error = 1.0  # Start at 1.0 (worst case)
        
        # Print initial state
        if self.config.print_iterations:
            self._print_initial_state()
        
        # Setup progress bars
        use_tqdm = not self.config.print_iterations  # Don't use tqdm if printing iterations
        
        if use_tqdm:
            # Progress bar for iterations
            iter_pbar = tqdm(
                total=self.config.max_iterations,
                desc="Iterations",
                unit="iter",
                position=0,
                leave=True
            )
            
            # Progress bar for convergence (1/std^2 scale)
            convergence_target = round(1 / (self.config.convergence_threshold ** 2))
            conv_pbar = tqdm(
                total=convergence_target,
                desc="Convergence",
                position=1,
                leave=True
            )
            conv_pbar.set_postfix({'std_error': f"{max_std_error:.6f}"})
        
        # Main simulation loop
        while not converged and iteration < self.config.max_iterations:
            iteration_start = time.time()
            iteration += 1
            
            # Run one iteration
            boards = self._complete_boards()
            hand_strengths = self._evaluate_all_hands(boards)
            iteration_equity = self._calculate_iteration_equity(hand_strengths)
            
            # Update equities
            self._update_equities(iteration_equity, iteration)
            
            # Track timing
            iteration_time = (time.time() - iteration_start) * 1000  # ms
            self.iteration_times.append(iteration_time)
            
            # Update progress
            if use_tqdm:
                iter_pbar.update(1)
                
                # Update convergence bar and check convergence every 100 iterations
                if iteration >= 100 and iteration % 100 == 0:
                    max_std_error = self.equity_history.max_standard_error(iteration)
                    
                    # Update convergence bar (1/std^2 grows linearly with iterations)
                    if max_std_error > 0:
                        convergence_metric = round(1 / (max_std_error ** 2))
                        conv_pbar.n = min(convergence_metric, convergence_target)
                        conv_pbar.refresh()
                        conv_pbar.set_postfix({'std_error': f"{max_std_error:.6f}"})
                    
                    # Check convergence
                    if max_std_error < self.config.convergence_threshold:
                        converged = True
                        conv_pbar.n = convergence_target  # Fill the bar
                        conv_pbar.refresh()
                        break
            else:
                # Original print-based progress
                if self.config.print_iterations and iteration % 100 == 0:
                    self._print_progress(iteration)
                
                # Check convergence every 100 iterations
                if iteration >= 100 and iteration % 100 == 0:
                    converged = self._check_convergence(iteration)
        
        # Close progress bars
        if use_tqdm:
            iter_pbar.close()
            conv_pbar.close()
            if converged:
                print(f"✓ Converged after {iteration} iterations (max std error: {max_std_error:.6f})")
            else:
                print(f"✗ Reached max iterations ({iteration}) without convergence (max std error: {max_std_error:.6f})")
        
        # Build results
        total_time = time.time() - start_time
        return self._build_results(iteration, total_time, converged)
    
    def _complete_boards(self) -> npt.NDArray[np.int32]:
        """Complete all three boards with random cards."""
        if self.cards_to_deal_per_board == 0:
            return self.initial_boards
        
        # Copy initial boards
        boards = self.initial_boards.copy()
        
        # Sample cards for all boards at once
        if self.total_cards_to_deal > 0:
            sampled_indices = np.random.choice(
                self.available_indices,
                size=self.total_cards_to_deal,
                replace=False
            )
            
            # Assign to boards
            card_idx = 0
            for board_idx in range(3):
                # Find empty slots on this board
                empty_slots = np.where(boards[board_idx, :, 0] == 0)[0]
                
                # Fill them
                for slot in empty_slots[:self.cards_to_deal_per_board]:
                    deck_idx = sampled_indices[card_idx]
                    boards[board_idx, slot] = self.full_deck[deck_idx]
                    card_idx += 1
        
        return boards
    
    def _evaluate_all_hands(self, boards: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        """
        Evaluate all player hands on all boards.
        
        Returns:
            Array of shape (3, num_players, 6) with hand strengths
        """
        hand_strengths = np.zeros((3, self.num_players, 6), dtype=np.int32)
        
        for board_idx in range(3):
            for player_idx in range(self.num_players):
                hand_strengths[board_idx, player_idx] = find_best_six_card_triple_flop_hand(
                    self.player_cards[player_idx],
                    boards[board_idx]
                )
        
        return hand_strengths
    
    def _calculate_iteration_equity(self, hand_strengths: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
        """
        Determine who wins this iteration.
        
        Args:
            hand_strengths: Array of shape (3, num_players, 6)
        
        Returns:
            Equity array showing each player's share (sums to 1.0)
        """
        points = np.zeros(self.num_players)
        
        # Award points for each board
        for board_idx in range(3):
            board_hands = hand_strengths[board_idx]
            winner_mask = self._find_winners_on_board(board_hands)
            num_winners = np.sum(winner_mask)
            
            # Split 1 point among winners
            if num_winners > 0:
                points[winner_mask] += 1.0 / num_winners
        
        # Find overall winner(s) - most total points
        max_points = np.max(points)
        if max_points > 0:
            overall_winners = (points == max_points)
            num_overall_winners = np.sum(overall_winners)
            
            # Equity for this iteration
            equity = np.zeros(self.num_players)
            equity[overall_winners] = 1.0 / num_overall_winners
        else:
            # Shouldn't happen, but handle gracefully
            equity = np.ones(self.num_players) / self.num_players
        
        return equity
    
    def _find_winners_on_board(self, hands: npt.NDArray[np.int32]) -> npt.NDArray[np.bool_]:
        """
        Find winner(s) on a single board.
        
        Args:
            hands: Array of shape (num_players, 6) with hand strengths
        
        Returns:
            Boolean mask indicating winners
        """
        # Find the best hand using lexicographic comparison
        best_idx = 0
        for i in range(1, len(hands)):
            # Compare hands element by element
            for j in range(6):
                if hands[i, j] > hands[best_idx, j]:
                    best_idx = i
                    break
                elif hands[i, j] < hands[best_idx, j]:
                    break
        
        # Find all players with hand equal to best
        winner_mask = np.zeros(len(hands), dtype=bool)
        best_hand = hands[best_idx]
        for i in range(len(hands)):
            if np.array_equal(hands[i], best_hand):
                winner_mask[i] = True
        
        return winner_mask
    
    def _update_equities(self, iteration_equity: npt.NDArray[np.float64], iteration: int):
        """Update equity tracking."""
        # Add win/loss result to history
        self.equity_history.add(iteration_equity)
        
        # Current estimates are maintained in equity_history via win_counts
    
    def _check_convergence(self, iteration: int) -> bool:
        """Check if standard errors are below threshold."""
        if iteration < 100:  # Need minimum samples for meaningful std error
            return False
        
        max_std_error = self.equity_history.max_standard_error(iteration)
        
        if self.config.print_iterations and iteration % 100 == 0:
            print(f"Iteration {iteration}: max std error = {max_std_error:.6f}")
        
        return max_std_error < self.config.convergence_threshold
    
    def _build_results(self, iterations: int, total_time: float, converged: bool) -> SimulationResults:
        """Build simulation results object."""
        # Calculate timing statistics
        if self.iteration_times:
            avg_iteration_time_ms = np.mean(self.iteration_times)
            iterations_per_second = 1000.0 / avg_iteration_time_ms if avg_iteration_time_ms > 0 else 0
        else:
            avg_iteration_time_ms = 0
            iterations_per_second = 0
        
        # Build proper equity history (running averages)
        equity_history_raw = self.equity_history.to_array()
        equity_history = np.zeros_like(equity_history_raw)
        for i in range(1, equity_history_raw.shape[1]):
            equity_history[:, i] = np.sum(equity_history_raw[:, :i+1], axis=1) / i
        
        return SimulationResults(
            player_equities=equity_history,
            iteration_count=iterations,
            iteration_durations=np.array(self.iteration_times),
            convergence_history=np.array([]),  # Could track max_change values if needed
            average_iteration_time_ms=avg_iteration_time_ms,
            iterations_per_second=iterations_per_second,
            total_time_seconds=total_time
        )
    
    def _print_initial_state(self):
        """Print initial game state."""
        print("\n" + "="*50)
        print("SIMULATION STARTING")
        print("="*50)
        print(f"Number of players: {self.num_players}")
        print(f"Cards to deal per board: {self.cards_to_deal_per_board}")
        print(f"Available cards in deck: {len(self.available_indices)}")
        print(f"Convergence threshold: {self.config.convergence_threshold} (standard error)")
        print()
    
    def _print_progress(self, iteration: int):
        """Print progress update."""
        equities = self.equity_history.win_counts / iteration if iteration > 0 else np.zeros(self.num_players)
        std_errors = self.equity_history.get_standard_errors(iteration)
        max_std = self.equity_history.max_standard_error(iteration)
        
        print(f"Iteration {iteration} (max std: {max_std:.6f}): ", end="")
        for i in range(self.num_players):
            print(f"P{i}: {equities[i]:.4f}±{std_errors[i]:.4f} ", end="")
        print()