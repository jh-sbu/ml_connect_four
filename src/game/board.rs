pub const ROWS: usize = 6;
pub const COLS: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty,
    Red,
    Yellow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    cells: [[Cell; COLS]; ROWS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveError {
    ColumnFull,
    InvalidColumn,
}

impl Board {
    /// Create a new empty board
    pub fn new() -> Self {
        Board {
            cells: [[Cell::Empty; COLS]; ROWS],
        }
    }

    /// Get the cell at a specific position
    /// Row 0 is the top, row 5 is the bottom
    pub fn get(&self, row: usize, col: usize) -> Cell {
        self.cells[row][col]
    }

    /// Check if a column is full
    pub fn is_column_full(&self, col: usize) -> bool {
        if col >= COLS {
            return true;
        }
        self.cells[0][col] != Cell::Empty
    }

    /// Drop a piece in a column, returns the row where it landed
    pub fn drop_piece(&mut self, col: usize, cell: Cell) -> Result<usize, MoveError> {
        if col >= COLS {
            return Err(MoveError::InvalidColumn);
        }

        if self.is_column_full(col) {
            return Err(MoveError::ColumnFull);
        }

        // Find the lowest empty row in this column
        for row in (0..ROWS).rev() {
            if self.cells[row][col] == Cell::Empty {
                self.cells[row][col] = cell;
                return Ok(row);
            }
        }

        unreachable!("Column should not be full if is_column_full returned false");
    }

    /// Check if the board is completely full
    pub fn is_full(&self) -> bool {
        (0..COLS).all(|col| self.is_column_full(col))
    }

    /// Check if the last move at (row, col) resulted in a win
    pub fn check_win(&self, row: usize, col: usize) -> bool {
        let cell = self.get(row, col);
        if cell == Cell::Empty {
            return false;
        }

        self.check_horizontal(row, col, cell)
            || self.check_vertical(row, col, cell)
            || self.check_diagonal_up(row, col, cell)
            || self.check_diagonal_down(row, col, cell)
    }

    /// Check horizontal win (left-right through the position)
    fn check_horizontal(&self, row: usize, col: usize, cell: Cell) -> bool {
        let mut count = 1; // Count the current piece

        // Check left
        let mut c = col as i32 - 1;
        while c >= 0 && self.cells[row][c as usize] == cell {
            count += 1;
            c -= 1;
        }

        // Check right
        let mut c = col + 1;
        while c < COLS && self.cells[row][c] == cell {
            count += 1;
            c += 1;
        }

        count >= 4
    }

    /// Check vertical win (down from the position)
    fn check_vertical(&self, row: usize, col: usize, cell: Cell) -> bool {
        let mut count = 1;

        // Only need to check downward (pieces fall down)
        let mut r = row + 1;
        while r < ROWS && self.cells[r][col] == cell {
            count += 1;
            r += 1;
        }

        count >= 4
    }

    /// Check diagonal win (bottom-left to top-right, /)
    fn check_diagonal_up(&self, row: usize, col: usize, cell: Cell) -> bool {
        let mut count = 1;

        // Check down-left
        let mut r = row as i32 + 1;
        let mut c = col as i32 - 1;
        while r < ROWS as i32 && c >= 0 && self.cells[r as usize][c as usize] == cell {
            count += 1;
            r += 1;
            c -= 1;
        }

        // Check up-right
        let mut r = row as i32 - 1;
        let mut c = col as i32 + 1;
        while r >= 0 && c < COLS as i32 && self.cells[r as usize][c as usize] == cell {
            count += 1;
            r -= 1;
            c += 1;
        }

        count >= 4
    }

    /// Check diagonal win (top-left to bottom-right, \)
    fn check_diagonal_down(&self, row: usize, col: usize, cell: Cell) -> bool {
        let mut count = 1;

        // Check up-left
        let mut r = row as i32 - 1;
        let mut c = col as i32 - 1;
        while r >= 0 && c >= 0 && self.cells[r as usize][c as usize] == cell {
            count += 1;
            r -= 1;
            c -= 1;
        }

        // Check down-right
        let mut r = row as i32 + 1;
        let mut c = col as i32 + 1;
        while r < ROWS as i32 && c < COLS as i32 && self.cells[r as usize][c as usize] == cell {
            count += 1;
            r += 1;
            c += 1;
        }

        count >= 4
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_board_is_empty() {
        let board = Board::new();
        for row in 0..ROWS {
            for col in 0..COLS {
                assert_eq!(board.get(row, col), Cell::Empty);
            }
        }
    }

    #[test]
    fn test_drop_piece() {
        let mut board = Board::new();

        // Drop first piece in column 3
        let row = board.drop_piece(3, Cell::Red).unwrap();
        assert_eq!(row, 5); // Should land at bottom
        assert_eq!(board.get(5, 3), Cell::Red);

        // Drop second piece in same column
        let row = board.drop_piece(3, Cell::Yellow).unwrap();
        assert_eq!(row, 4); // Should land on top of first piece
        assert_eq!(board.get(4, 3), Cell::Yellow);
    }

    #[test]
    fn test_column_full() {
        let mut board = Board::new();

        // Fill column 0
        for _ in 0..ROWS {
            board.drop_piece(0, Cell::Red).unwrap();
        }

        assert!(board.is_column_full(0));
        assert_eq!(board.drop_piece(0, Cell::Yellow), Err(MoveError::ColumnFull));
    }

    #[test]
    fn test_invalid_column() {
        let mut board = Board::new();
        assert_eq!(board.drop_piece(7, Cell::Red), Err(MoveError::InvalidColumn));
    }

    #[test]
    fn test_full_board() {
        let mut board = Board::new();
        for col in 0..COLS {
            for _ in 0..ROWS {
                board.drop_piece(col, Cell::Red).unwrap();
            }
        }
        assert!(board.is_full());
    }

    #[test]
    fn test_horizontal_win() {
        let mut board = Board::new();
        // Create horizontal line at bottom row
        for col in 0..4 {
            board.drop_piece(col, Cell::Red).unwrap();
        }
        assert!(board.check_win(5, 2)); // Check middle of the line
    }

    #[test]
    fn test_vertical_win() {
        let mut board = Board::new();
        // Create vertical line in column 3
        for _ in 0..4 {
            board.drop_piece(3, Cell::Yellow).unwrap();
        }
        assert!(board.check_win(2, 3)); // Check the 4th piece
    }

    #[test]
    fn test_diagonal_up_win() {
        let mut board = Board::new();
        // Create diagonal / pattern
        board.drop_piece(0, Cell::Red).unwrap();

        board.drop_piece(1, Cell::Yellow).unwrap();
        board.drop_piece(1, Cell::Red).unwrap();

        board.drop_piece(2, Cell::Yellow).unwrap();
        board.drop_piece(2, Cell::Yellow).unwrap();
        board.drop_piece(2, Cell::Red).unwrap();

        board.drop_piece(3, Cell::Yellow).unwrap();
        board.drop_piece(3, Cell::Yellow).unwrap();
        board.drop_piece(3, Cell::Yellow).unwrap();
        let row = board.drop_piece(3, Cell::Red).unwrap();

        assert!(board.check_win(row, 3));
    }

    #[test]
    fn test_diagonal_down_win() {
        let mut board = Board::new();
        // Create diagonal \ pattern
        board.drop_piece(6, Cell::Red).unwrap();

        board.drop_piece(5, Cell::Yellow).unwrap();
        board.drop_piece(5, Cell::Red).unwrap();

        board.drop_piece(4, Cell::Yellow).unwrap();
        board.drop_piece(4, Cell::Yellow).unwrap();
        board.drop_piece(4, Cell::Red).unwrap();

        board.drop_piece(3, Cell::Yellow).unwrap();
        board.drop_piece(3, Cell::Yellow).unwrap();
        board.drop_piece(3, Cell::Yellow).unwrap();
        let row = board.drop_piece(3, Cell::Red).unwrap();

        assert!(board.check_win(row, 3));
    }

    #[test]
    fn test_no_win_with_three() {
        let mut board = Board::new();
        for col in 0..3 {
            board.drop_piece(col, Cell::Red).unwrap();
        }
        assert!(!board.check_win(5, 1)); // Only 3 in a row
    }
}
