import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
# Create PyTorch datasets and dataloaders.
from Models import FastOthelloNet
from collections import Counter


# dataset is taken from: https://github.com/dimitri-rusin/othello/tree/main
class OthelloDataset(Dataset):

    def __init__(self, examples):
        """
        examples: list of tuples (board_state, action, outcome)
        """
        self.boards = [
            torch.tensor(e[0], dtype=torch.float32) for e in examples
        ]
        self.actions = [e[1] for e in examples]
        self.outcomes = [e[2] for e in examples]

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        outcome = torch.tensor(self.outcomes[idx], dtype=torch.float32)

        return board, action, outcome


BOARD_SIZE = 8


def board_to_key(board):
    """
    Convert a board state (numpy array or similar) to a hashable tuple.
    You can use flatten() to create a 1D tuple representation.
    """
    return tuple(board.flatten().tolist())


def remove_conflicting_duplicates(examples):
    """
    Filter the examples list (each item is a tuple (board_state, action, outcome)).
    Remove board states that have conflicting moves (and outcomes).
    
    Returns a list with filtered examples.
    """
    board_groups = {}
    for board, action, outcome in examples:
        # Generate a hashable key; if board is a tensor, convert to numpy first.
        if hasattr(board, 'numpy'):
            board_array = board.cpu().numpy()
        else:
            board_array = board
        key = board_to_key(board_array)

        if key not in board_groups:
            board_groups[key] = []
        board_groups[key].append((action, outcome))

    # Now filter: keep only board states where the move (and outcome, if desired) is unique.
    filtered_examples = []
    for key, data in board_groups.items():
        # Extract all actions and outcomes observed for this board state.
        actions = [x[0] for x in data]
        outcomes = [x[1] for x in data]

        # Use majority voting on actions.
        action_counter = Counter(actions)
        majority_action, _ = action_counter.most_common(1)[0]

        # Use majority voting on outcomes.
        outcome_counter = Counter(outcomes)
        majority_outcome, _ = outcome_counter.most_common(1)[0]

        # If the board had conflicting data, print the details.
        if len(action_counter) > 1 or len(outcome_counter) > 1:
            print(
                f"Conflict for board {key}:\n"
                f"  actions: {actions}, outcomes: {outcomes}\n"
                f"  Majority vote => action: {majority_action}, outcome: {majority_outcome}"
            )

        # Reconstruct the board from its key.
        board_array = np.array(key).reshape(BOARD_SIZE, BOARD_SIZE)
        filtered_examples.append(
            (board_array, majority_action, majority_outcome))

    return filtered_examples


def initial_board(board_size=BOARD_SIZE):
    """
    Create an 8x8 Othello board with the standard initial configuration.
    Convention: use -1 for Black, +1 for White, and 0 for empty.
    Standard starting position (with Black to move first):
      - board[3,3] = +1, board[3,4] = -1
      - board[4,3] = -1, board[4,4] = +1
    """
    board = np.zeros((board_size, board_size), dtype=np.int32)
    mid = board_size // 2
    board[mid - 1, mid - 1] = -1
    board[mid - 1, mid] = 1
    board[mid, mid - 1] = 1
    board[mid, mid] = -1
    return board


def move_to_coord(move_str):
    """
    Convert a move in algebraic notation (e.g. "f5") into board coordinates (row, col).
    Assumes columns are labeled a-h and rows are 1-8.
    """
    if move_str.lower() == "pass":
        return None  # designate pass move
    col = ord(move_str[0].lower()) - ord('a')
    row = int(move_str[1]) - 1
    return row, col


def apply_move(board, row, col, player):
    """
    Given the current board, apply the move at (row, col) for the given player.
    This function will flip the opponent's pieces according to Othello rules.
    It returns a new board state (a copy with the move applied).
    Assumes the move is legal.
    """
    new_board = copy.deepcopy(board)
    new_board[row, col] = player
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1),
                  (1, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        flips = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and new_board[
                r, c] == -player:
            flips.append((r, c))
            r += dr
            c += dc
        # If we are still on board and end on a piece of the same color, flip pieces.
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and new_board[
                r, c] == player:
            for fr, fc in flips:
                new_board[fr, fc] = player
    return new_board


def process_game(moves_str, outcome, board_size=BOARD_SIZE):
    """
    Processes a single game into training examples.
    
    Parameters:
      moves_str: a string of concatenated moves; each move is 2 characters (e.g. "f5")
      outcome: game outcome value from Black's perspective (+1 if black won, -1 if lost, 0 if draw)
      board_size: default 8
      
    Returns:
      A list of examples, each tuple: (board_state, target_action, value_target)
        - board_state: numpy array shape (board_size, board_size) with values in {-1, 0, 1}
        - target_action: integer index; computed as row*board_size+col for a regular move, 
                         or board_size*board_size for a pass move.
        - value_target: outcome from the perspective of the current player on that move.
                      (if current player is black, use outcome; if white, use -outcome)
    """
    examples = []
    # Starting board (using standard initial configuration)
    board = initial_board(board_size)
    player = 1

    # Process moves in strides of 2 (since each move is two characters)
    num_moves = len(moves_str) // 2
    for i in range(num_moves):
        move = moves_str[2 * i:2 * i + 2]
        coord = move_to_coord(move)
        # Compute action target: if pass, use index board_size*board_size, else compute index.
        if coord is None:
            action = board_size * board_size
        else:
            row, col = coord
            action = row * board_size + col

        # Set value target: from the perspective of the current player.
        value_target = outcome if player == 1 else -outcome
        # Copy board state to avoid later mutation.
        board_copy = board.copy()
        examples.append((board_copy * player, action, value_target))
        # Apply the move only if not a pass.
        if coord is not None:
            board = apply_move(board, row, col, player)
        # Alternate players.
        player = -player
    return examples


def train_epoch(model, dataloader, optimizer, criterion_policy,
                criterion_value, device):
    model.train()
    running_loss = 0.0
    for board, move, outcome in tqdm(dataloader, desc='Training', leave=False):
        board = board.to(device)
        move = move.to(device)
        outcome = outcome.to(device).view(-1, 1)
        optimizer.zero_grad()
        policy_logits, value = model(board)
        loss_policy = criterion_policy(policy_logits, move)
        loss_value = criterion_value(value, outcome)
        loss = loss_policy + loss_value
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion_policy, criterion_value, device):
    model.eval()
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    correct_policy = 0
    total_samples = 0
    with torch.no_grad():
        for board, move, outcome in tqdm(dataloader,
                                         desc='Evaluating',
                                         leave=False):
            board = board.to(device)
            move = move.to(device)
            outcome = outcome.to(device).view(-1, 1)
            policy_logits, value = model(board)
            loss_policy = criterion_policy(policy_logits, move)
            loss_value = criterion_value(value, outcome)
            loss = loss_policy + loss_value
            running_loss += loss.item()
            running_policy_loss += loss_policy.item()
            running_value_loss += loss_value.item()
            predicted_moves = torch.argmax(policy_logits, dim=1)

            correct_policy += (predicted_moves == move).sum().item()
            total_samples += move.size(0)
    avg_loss = running_loss / len(dataloader)
    policy_acc = correct_policy / total_samples
    avg_policy_loss = running_policy_loss / len(dataloader)
    avg_value_loss = running_value_loss / len(dataloader)
    return avg_loss, avg_policy_loss, avg_value_loss, policy_acc


def load_and_process_csv(csv_file,
                         board_size=BOARD_SIZE,
                         test_size=0.2,
                         random_state=42):
    """
    Reads the CSV file, splits it into training and testing rows, and processes each game into examples.
    
    The CSV file is expected to have a header with:
      - "eOthello_game_id": unique identifier for each game,
      - "winner": game outcome from Black's perspective (+1, -1, or 0),
      - "game_moves": concatenated string of moves (each move is 2 characters).
    
    Splitting is performed at the row level before game processing.
    
    Returns:
      train_examples, test_examples: two lists of examples (board_state, action, value_target).
    """
    df = pd.read_csv(csv_file)
    print(f"CSV read with shape: {df.shape}")

    # First split the dataframe by rows (each row = one game).
    train_df, test_df = train_test_split(df,
                                         test_size=test_size,
                                         random_state=random_state)

    train_examples = []
    for _, row in train_df.iterrows():
        outcome = float(row["winner"])
        moves_str = str(row["game_moves"]).strip()
        game_examples = process_game(moves_str, outcome, board_size=board_size)
        train_examples.extend(game_examples)

    test_examples = []
    for _, row in test_df.iterrows():
        outcome = float(row["winner"])
        moves_str = str(row["game_moves"]).strip()
        game_examples = process_game(moves_str, outcome, board_size=board_size)
        test_examples.extend(game_examples)

    return remove_conflicting_duplicates(
        train_examples), remove_conflicting_duplicates(test_examples)


if __name__ == '__main__':
    path = "/Users/andreiafonin/Downloads/othello_dataset.csv"

    train_examples, test_examples = load_and_process_csv(path, BOARD_SIZE, 0.1)

    print(f"Total training examples: {len(train_examples)}")
    print(f"Total testing examples: {len(test_examples)}")

    train_dataset = OthelloDataset(train_examples)
    val_dataset = OthelloDataset(test_examples)
    train_loader = DataLoader(train_dataset,
                              batch_size=1024,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1024,
                            shuffle=False,
                            pin_memory=True)

    # Set device.
    action_size = BOARD_SIZE * BOARD_SIZE + 1  # Extra action for "pass"
    model = FastOthelloNet(BOARD_SIZE, action_size)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    num_epochs = 500
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer,
                                 criterion_policy, criterion_value, device)
        val_loss, val_policy_loss, val_value_loss, policy_acc = evaluate(
            model, val_loader, criterion_policy, criterion_value, device)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Policy Loss: {val_policy_loss:.4f} | "
            f"Value Loss: {val_value_loss:.4f} | Policy Accuracy: {policy_acc:.4f}"
        )
        torch.save(model, 'othello_policy_supervised.pt')
