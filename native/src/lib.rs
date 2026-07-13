use std::collections::HashMap;

use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Coordinate = (isize, isize);

fn consensus_index(candidates: &[Vec<i64>], confidences: &[f64]) -> Result<usize, &'static str> {
    if candidates.is_empty() {
        return Err("candidates must not be empty");
    }
    if candidates.len() != confidences.len() {
        return Err("candidates and confidences must have equal length");
    }
    if confidences.iter().any(|score| !score.is_finite()) {
        return Err("confidences must be finite");
    }

    let mut counts: HashMap<&[i64], usize> = HashMap::new();
    for candidate in candidates {
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
fn select_consensus(candidates: Vec<Vec<i64>>, confidences: Vec<f64>) -> PyResult<usize> {
    consensus_index(&candidates, &confidences).map_err(PyValueError::new_err)
}

#[pyfunction]
fn select_consensus_array(
    candidates: PyReadonlyArray3<'_, i64>,
    confidences: PyReadonlyArray2<'_, f32>,
) -> PyResult<Vec<usize>> {
    let candidate_shape = candidates.shape();
    let confidence_shape = confidences.shape();
    if candidate_shape[0] == 0 || candidate_shape[1] == 0 {
        return Err(PyValueError::new_err(
            "candidate rollouts and batches must not be empty",
        ));
    }
    if confidence_shape != &candidate_shape[..2] {
        return Err(PyValueError::new_err(
            "confidences must have shape [rollouts, batch]",
        ));
    }
    let candidate_values = candidates
        .as_slice()
        .map_err(|_| PyValueError::new_err("candidates must be C-contiguous"))?;
    let confidence_values = confidences
        .as_slice()
        .map_err(|_| PyValueError::new_err("confidences must be C-contiguous"))?;
    if confidence_values.iter().any(|score| !score.is_finite()) {
        return Err(PyValueError::new_err("confidences must be finite"));
    }
    let rollouts = candidate_shape[0];
    let batch_size = candidate_shape[1];
    let sequence_length = candidate_shape[2];

    Ok((0..batch_size)
        .map(|batch_index| {
            let candidate_at = |rollout: usize| {
                let offset = (rollout * batch_size + batch_index) * sequence_length;
                &candidate_values[offset..offset + sequence_length]
            };
            let mut counts: HashMap<&[i64], usize> = HashMap::new();
            for rollout in 0..rollouts {
                *counts.entry(candidate_at(rollout)).or_default() += 1;
            }
            let max_count = counts.values().copied().max().unwrap_or(1);
            let mut best = 0;
            for rollout in 1..rollouts {
                let eligible = counts[candidate_at(rollout)] == max_count;
                let best_eligible = counts[candidate_at(best)] == max_count;
                let score = confidence_values[rollout * batch_size + batch_index];
                let best_score = confidence_values[best * batch_size + batch_index];
                if eligible && (!best_eligible || score > best_score) {
                    best = rollout;
                }
            }
            best
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (grid, path, start=(0, 0), goal=None, wall_value=1))]
fn validate_maze_path(
    grid: Vec<Vec<i64>>,
    path: Vec<Coordinate>,
    start: Coordinate,
    goal: Option<Coordinate>,
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
fn validate_maze_moves_array(
    predictions: PyReadonlyArray2<'_, i64>,
    grids: PyReadonlyArray3<'_, i64>,
    starts: PyReadonlyArray2<'_, i64>,
    goals: PyReadonlyArray2<'_, i64>,
) -> PyResult<Vec<bool>> {
    let prediction_shape = predictions.shape();
    let grid_shape = grids.shape();
    let start_shape = starts.shape();
    let goal_shape = goals.shape();
    let batch_size = prediction_shape[0];
    if grid_shape[0] != batch_size
        || start_shape != [batch_size, 2]
        || goal_shape != [batch_size, 2]
    {
        return Err(PyValueError::new_err(
            "predictions, grids, starts, and goals must have equal batch size",
        ));
    }
    let prediction_values = predictions
        .as_slice()
        .map_err(|_| PyValueError::new_err("predictions must be C-contiguous"))?;
    let grid_values = grids
        .as_slice()
        .map_err(|_| PyValueError::new_err("grids must be C-contiguous"))?;
    let start_values = starts
        .as_slice()
        .map_err(|_| PyValueError::new_err("starts must be C-contiguous"))?;
    let goal_values = goals
        .as_slice()
        .map_err(|_| PyValueError::new_err("goals must be C-contiguous"))?;
    let sequence_length = prediction_shape[1];
    let rows = grid_shape[1];
    let cols = grid_shape[2];

    Ok((0..batch_size)
        .map(|index| {
            let start = (
                start_values[index * 2] as isize,
                start_values[index * 2 + 1] as isize,
            );
            let goal = (
                goal_values[index * 2] as isize,
                goal_values[index * 2 + 1] as isize,
            );
            let token_offset = index * sequence_length;
            let grid_offset = index * rows * cols;
            validate_maze_moves_flat(
                &prediction_values[token_offset..token_offset + sequence_length],
                &grid_values[grid_offset..grid_offset + rows * cols],
                rows,
                cols,
                start,
                goal,
            )
        })
        .collect())
}

fn validate_maze_moves_flat(
    tokens: &[i64],
    grid: &[i64],
    rows: usize,
    cols: usize,
    start: Coordinate,
    goal: Coordinate,
) -> bool {
    if rows == 0
        || cols == 0
        || start.0 < 0
        || start.1 < 0
        || start.0 as usize >= rows
        || start.1 as usize >= cols
        || grid[start.0 as usize * cols + start.1 as usize] != 0
    {
        return false;
    }
    let (mut row, mut col) = start;
    for token in tokens {
        if *token == 1 {
            return (row, col) == goal;
        }
        let (row_delta, col_delta) = match token {
            2 => (-1, 0),
            3 => (1, 0),
            4 => (0, -1),
            5 => (0, 1),
            _ => return false,
        };
        row += row_delta;
        col += col_delta;
        if row < 0
            || col < 0
            || row as usize >= rows
            || col as usize >= cols
            || grid[row as usize * cols + col as usize] != 0
        {
            return false;
        }
    }
    false
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
    module.add_function(wrap_pyfunction!(select_consensus_array, module)?)?;
    module.add_function(wrap_pyfunction!(validate_maze_path, module)?)?;
    module.add_function(wrap_pyfunction!(validate_maze_moves_array, module)?)?;
    module.add_function(wrap_pyfunction!(validate_sudoku_board, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{consensus_index, validate_maze_moves_flat};

    #[test]
    fn consensus_prefers_majority_then_confidence() {
        let candidates = vec![vec![1, 2], vec![9], vec![1, 2]];
        assert_eq!(consensus_index(&candidates, &[0.2, 0.99, 0.8]), Ok(2));
    }

    #[test]
    fn maze_moves_require_legal_goal_reaching_eos() {
        let grid = [0, 0, 1, 0];
        assert!(validate_maze_moves_flat(
            &[5, 3, 1],
            &grid,
            2,
            2,
            (0, 0),
            (1, 1)
        ));
        assert!(!validate_maze_moves_flat(
            &[5, 3],
            &grid,
            2,
            2,
            (0, 0),
            (1, 1)
        ));
        assert!(!validate_maze_moves_flat(
            &[3, 5, 1],
            &grid,
            2,
            2,
            (0, 0),
            (1, 1)
        ));
    }
}
