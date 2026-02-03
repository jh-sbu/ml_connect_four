use super::board::Cell;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Red,
    Yellow,
}

impl Player {
    /// Get the other player
    pub fn other(self) -> Player {
        match self {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        }
    }

    /// Convert player to cell type
    pub fn to_cell(self) -> Cell {
        match self {
            Player::Red => Cell::Red,
            Player::Yellow => Cell::Yellow,
        }
    }

    /// Get player name for display
    pub fn name(self) -> &'static str {
        match self {
            Player::Red => "Red",
            Player::Yellow => "Yellow",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_other_player() {
        assert_eq!(Player::Red.other(), Player::Yellow);
        assert_eq!(Player::Yellow.other(), Player::Red);
    }

    #[test]
    fn test_player_name() {
        assert_eq!(Player::Red.name(), "Red");
        assert_eq!(Player::Yellow.name(), "Yellow");
    }
}
