use crate::mol::molecule::Molecule;
use smallvec::SmallVec;

#[derive(Default)]
pub struct CalculationContext {
    pub dist_u32: Vec<u32>,
    pub has_dist_u32: bool,
    pub dist_bo: Vec<f64>,
    pub has_dist_bo: bool,
    
    // Distance matrix workspace
    pub dist_f64: Vec<f64>,
    pub row_f64: Vec<f64>,
    pub visited_bool: Vec<bool>,
    pub weighted_neighbors: Vec<SmallVec<[(usize, f64); 4]>>,

    pub bfs_queue: Vec<usize>,
    pub bfs_dist: Vec<usize>,
}

impl CalculationContext {
    pub fn new() -> Self {
        Self {
            dist_u32: Vec::new(),
            has_dist_u32: false,
            dist_bo: Vec::new(),
            has_dist_bo: false,
            dist_f64: Vec::new(),
            row_f64: Vec::new(),
            visited_bool: Vec::new(),
            weighted_neighbors: Vec::new(),
            bfs_queue: Vec::new(),
            bfs_dist: Vec::new(),
        }
    }

    pub fn reset_for_new_molecule(&mut self) {
        self.has_dist_u32 = false;
        self.has_dist_bo = false;
        self.bfs_queue.clear();
        self.bfs_dist.clear();
        
        self.dist_f64.clear();
        self.row_f64.clear();
        self.visited_bool.clear();
        self.weighted_neighbors.clear();
    }
}
