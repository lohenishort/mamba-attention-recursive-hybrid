use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn select_consensus(candidates: Vec<Vec<i64>>, confidences: Vec<f64>) -> PyResult<usize> {
    if candidates.is_empty() {
        return Err(PyValueError::new_err("candidates must not be empty"));
    }
    if candidates.len() != confidences.len() {
        return Err(PyValueError::new_err(
            "candidates and confidences must have equal length",
        ));
    }
    if confidences.iter().any(|score| !score.is_finite()) {
        return Err(PyValueError::new_err("confidences must be finite"));
    }

    let mut counts: HashMap<&[i64], usize> = HashMap::new();
    for candidate in &candidates {
        *counts.entry(candidate).or_default() += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(1);

    let mut best = 0;
    for index in 1..candidates.len() {
        let eligible = counts[candidates[index].as_slice()] == max_count;
        let best_eligible = counts[candidates[best].as_slice()] == max_count;
        if eligible && (!best_eligible || confidences[index] > confidences[best]) {
            best = index;
        }
    }
    Ok(best)
}

#[pyfunction]
#[pyo3(signature = (grid, path, start=(0, 0), goal=None, wall_value=1))]
fn validate_maze_path(
    grid: Vec<Vec<i64>>,
    path: Vec<(isize, isize)>,
    start: (isize, isize),
    goal: Option<(isize, isize)>,
    wall_value: i64,
) -> bool {
    let rows = grid.len();
    let cols = grid.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 || grid.iter().any(|row| row.len() != cols) || path.is_empty() {
        return false;
    }
    let expected_goal = goal.unwrap_or((rows as isize - 1, cols as isize - 1));
    if path.first() != Some(&start) || path.last() != Some(&expected_goal) {
        return false;
    }

    let valid_cell = |&(row, col): &(isize, isize)| {
        row >= 0
            && col >= 0
            && (row as usize) < rows
            && (col as usize) < cols
            && grid[row as usize][col as usize] != wall_value
    };
    path.iter().all(valid_cell)
        && path
            .windows(2)
            .all(|step| (step[0].0 - step[1].0).abs() + (step[0].1 - step[1].1).abs() == 1)
}

#[pyfunction]
#[pyo3(signature = (board, puzzle=None))]
fn validate_sudoku_board(board: Vec<i64>, puzzle: Option<Vec<i64>>) -> bool {
    if board.len() != 81 || board.iter().any(|value| !(1..=9).contains(value)) {
        return false;
    }
    if let Some(clues) = puzzle {
        if clues.len() != 81
            || clues.iter().any(|value| !(0..=9).contains(value))
            || clues
                .iter()
                .zip(&board)
                .any(|(clue, value)| *clue != 0 && clue != value)
        {
            return false;
        }
    }

    let complete = |values: [i64; 9]| {
        let mut seen = [false; 10];
        values.into_iter().all(|value| {
            let unique = !seen[value as usize];
            seen[value as usize] = true;
            unique
        })
    };
    for index in 0..9 {
        if !complete(std::array::from_fn(|col| board[index * 9 + col]))
            || !complete(std::array::from_fn(|row| board[row * 9 + index]))
        {
            return false;
        }
    }
    for box_row in 0..3 {
        for box_col in 0..3 {
            if !complete(std::array::from_fn(|offset| {
                let row = box_row * 3 + offset / 3;
                let col = box_col * 3 + offset % 3;
                board[row * 9 + col]
            })) {
                return false;
            }
        }
    }
    true
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(select_consensus, module)?)?;
    module.add_function(wrap_pyfunction!(validate_maze_path, module)?)?;
    module.add_function(wrap_pyfunction!(validate_sudoku_board, module)?)?;
    Ok(())
}
