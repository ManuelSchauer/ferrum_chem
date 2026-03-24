use crate::mol::molecule::Molecule;
use smallvec::SmallVec;

#[derive(Default)]
pub struct CalculationContext {
    pub dist_u8: Vec<u8>,
    pub has_dist_u8: bool,
    pub dist_bo: Vec<f64>,
    pub has_dist_bo: bool,
    
    // Distance matrix workspace
    pub dist_f64: Vec<f64>,
    pub row_f64: Vec<f64>,
    pub visited_bool: Vec<bool>,
    pub weighted_neighbors: Vec<SmallVec<[(usize, f64); 4]>>,

    pub bfs_queue: Vec<usize>,
    pub bfs_dist: Vec<usize>,

    // Reusable float workspaces for matrix operations (e.g. Ipc)
    pub mat_a: Vec<f64>,
    pub mat_b: Vec<f64>,
    pub mat_c: Vec<f64>,
    pub mat_d: Vec<f64>,
    pub vec_a: Vec<f64>,
    pub vec_b: Vec<f64>,

    // BertzCT symmetry class keys
    pub bertz_keys: Vec<(SmallVec<[i32; 64]>, usize)>,
}

impl CalculationContext {
    pub fn new() -> Self {
        Self {
            dist_u8: Vec::new(),
            has_dist_u8: false,
            dist_bo: Vec::new(),
            has_dist_bo: false,
            dist_f64: Vec::new(),
            row_f64: Vec::new(),
            visited_bool: Vec::new(),
            weighted_neighbors: Vec::new(),
            bfs_queue: Vec::new(),
            bfs_dist: Vec::new(),
            mat_a: Vec::new(),
            mat_b: Vec::new(),
            mat_c: Vec::new(),
            mat_d: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            bertz_keys: Vec::new(),
        }
    }

    pub fn reset_for_new_molecule(&mut self) {
        self.has_dist_u8 = false;
        self.has_dist_bo = false;
        self.bfs_queue.clear();
        self.bfs_dist.clear();
        
        self.dist_f64.clear();
        self.row_f64.clear();
        self.visited_bool.clear();
        self.weighted_neighbors.clear();
        self.mat_a.clear();
        self.mat_b.clear();
        self.mat_c.clear();
        self.mat_d.clear();
        self.vec_a.clear();
        self.vec_b.clear();
        self.bertz_keys.clear();
    }
}
