use crate::algo::canon::rank_atoms;
use crate::mol::elements::ELEMENT_SYMBOLS;
use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// Convert a Molecule into a Canonical SMILES string.
pub fn write_canonical_smiles(mol: &Molecule) -> String {
    let ranks = rank_atoms(mol);
    write_smiles_with_order(mol, Some(&ranks))
}

/// Convert a Molecule into a SMILES string using depth-first search.
pub fn write_smiles(mol: &Molecule) -> String {
    write_smiles_with_order(mol, None)
}

/// Convert a Molecule into a SMILES string using an optional traversal rank ordering.
fn write_smiles_with_order(mol: &Molecule, order: Option<&[usize]>) -> String {
    if mol.atoms.is_empty() {
        return String::new();
    }
    
    let n = mol.atoms.len();
    let mut visited = vec![false; n];
    let mut tree_edges: Vec<Vec<usize>> = vec![vec![]; n];
    let mut ring_edges: Vec<(usize, usize)> = Vec::new();

    let mut atom_order: Vec<usize> = (0..n).collect();
    if let Some(ranks) = order {
        atom_order.sort_unstable_by(|&a, &b| ranks[a].cmp(&ranks[b]).then(a.cmp(&b)));
    }

    // Pass 1: Build spanning forest and find ring closure edges
    for &start_atom in &atom_order {
        if !visited[start_atom] {
            build_spanning_tree(mol, start_atom, usize::MAX, &mut visited, &mut tree_edges, &mut ring_edges, order);
        }
    }

    // Assign ring digits
    let mut node_ring_closures: Vec<Vec<(Option<BondType>, usize)>> = vec![vec![]; n];
    for (i, &(u, v)) in ring_edges.iter().enumerate() {
        let digit = i + 1;
        let bond_type = get_bond_type(mol, u, v);
        
        let mut explicit_bond = None;
        if bond_type != BondType::Single && bond_type != BondType::Aromatic {
            explicit_bond = Some(bond_type);
        } else if bond_type == BondType::Single && mol.atoms[u].is_aromatic() && mol.atoms[v].is_aromatic() {
            explicit_bond = Some(BondType::Single);
        }

        // v was visited before u, so v is the "open" ring and u is the "close" ring.
        node_ring_closures[v].push((None, digit));
        node_ring_closures[u].push((explicit_bond, digit));
    }

    // Pass 2: Generate SMILES
    let mut out = String::new();
    for &start_atom in &atom_order {
        // Since we process in order, and tree_edges form components, we just check if it's a root.
        // A root is a node that is never a target in tree_edges.
        let mut is_root = true;
        for edges in &tree_edges {
            if edges.contains(&start_atom) {
                is_root = false;
                break;
            }
        }

        if is_root {
            if !out.is_empty() {
                out.push('.');
            }
            write_tree(mol, start_atom, None, &tree_edges, &node_ring_closures, &mut out, order);
        }
    }

    out
}

/// Implements build spanning tree.
///
/// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
/// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
fn build_spanning_tree(
    mol: &Molecule,
    u: usize,
    p: usize,
    visited: &mut Vec<bool>,
    tree_edges: &mut [Vec<usize>],
    ring_edges: &mut Vec<(usize, usize)>,
    order: Option<&[usize]>,
) {
    visited[u] = true;
    
    let mut neighbors: Vec<usize> = mol.adjacency[u]
        .iter()
        .map(|&bi| {
            let b = &mol.bonds[bi as usize];
            if b.source as usize == u { b.target as usize } else { b.source as usize }
        })
        .collect();
        
    neighbors.sort_unstable_by(|&a, &b| {
        if let Some(ord) = order {
            ord[a].cmp(&ord[b]).then(a.cmp(&b))
        } else {
            a.cmp(&b)
        }
    }); // Ensure canonical traversal order

    for &v in &neighbors {
        if v == p {
            continue;
        }
        if visited[v] {
            // Back edge: only add it once (when u is "greater" than v in canonical order)
            let cmp = if let Some(ord) = order {
                ord[u].cmp(&ord[v]).then(u.cmp(&v))
            } else {
                u.cmp(&v)
            };
            
            if cmp == std::cmp::Ordering::Greater {
                ring_edges.push((u, v));
            }
        } else {
            tree_edges[u].push(v);
            build_spanning_tree(mol, v, u, visited, tree_edges, ring_edges, order);
        }
    }
}

/// Implements get bond type.
///
/// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
/// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
fn get_bond_type(mol: &Molecule, u: usize, v: usize) -> BondType {
    mol.bonds
        .iter()
        .find(|b| (b.source as usize == u && b.target as usize == v) || (b.source as usize == v && b.target as usize == u))
        .unwrap()
        .bond_type
}

/// Implements write tree.
///
/// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
/// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
fn write_tree(
    mol: &Molecule,
    u: usize,
    parent_bond: Option<BondType>,
    tree_edges: &[Vec<usize>],
    node_ring_closures: &[Vec<(Option<BondType>, usize)>],
    out: &mut String,
    order: Option<&[usize]>,
) {
    // Write incident bond from parent
    if let Some(bt) = parent_bond {
        write_bond(bt, out);
    }

    // Write atom symbol and properties
    write_atom(mol, u, out);

    // Write ring closures starting/ending at this atom
    for &(bt_opt, digit) in &node_ring_closures[u] {
        if let Some(bt) = bt_opt {
            write_bond(bt, out);
        }
        if digit < 10 {
            out.push_str(&digit.to_string());
        } else {
            out.push_str(&format!("%{:02}", digit));
        }
    }

    // Recursively write children
    let children = &tree_edges[u];
    for (i, &v) in children.iter().enumerate() {
        let bt = get_bond_type(mol, u, v);
        
        // Determine if we need to write the bond explicitly
        let mut pb = None;
        if bt != BondType::Single && bt != BondType::Aromatic {
            pb = Some(bt);
        } else if bt == BondType::Single && mol.atoms[u].is_aromatic() && mol.atoms[v].is_aromatic() {
            pb = Some(BondType::Single);
        }

        if i < children.len() - 1 {
            out.push('(');
            write_tree(mol, v, pb, tree_edges, node_ring_closures, out, order);
            out.push(')');
        } else {
            write_tree(mol, v, pb, tree_edges, node_ring_closures, out, order);
        }
    }
}

/// Implements write bond.
///
/// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
/// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
fn write_bond(bt: BondType, out: &mut String) {
    match bt {
        BondType::Single => out.push('-'),
        BondType::Double => out.push('='),
        BondType::Triple => out.push('#'),
        BondType::Aromatic => {}, // implicit
        BondType::Dative => out.push('>'),
        BondType::Unknown => out.push('~'),
    }
}

/// Implements write atom.
///
/// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
/// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
fn write_atom(mol: &Molecule, u: usize, out: &mut String) {
    let atom = &mol.atoms[u];
    let symbol = if (atom.atomic_number as usize) < ELEMENT_SYMBOLS.len() {
        ELEMENT_SYMBOLS[atom.atomic_number as usize]
    } else {
        "*"
    };
    
    // Check if it qualifies for the organic subset without brackets
    let is_organic = matches!(atom.atomic_number, 5|6|7|8|9|15|16|17|35|53);
    
    let needs_brackets = !is_organic 
        || atom.formal_charge != 0 
        || atom.isotope != 0 
        || atom.num_explicit_h > 0 
        // Aromatic heteroatoms outside standard valency rules often need brackets, 
        // but for safety, we bracket if it has unusual state. Actually, RDKit brackets 
        // anything that has explicit H. Let's rely on standard subset.
        || false;

    let mut sym = symbol.to_string();
    if atom.is_aromatic() {
        // Special case: Boron is 'b', Carbon is 'c', N is 'n', O is 'o', P is 'p', S is 's'
        if is_organic && atom.atomic_number != 9 && atom.atomic_number != 17 && atom.atomic_number != 35 && atom.atomic_number != 53 {
            sym = sym.to_lowercase();
        }
    }

    if needs_brackets {
        out.push('[');
        if atom.isotope != 0 {
            out.push_str(&atom.isotope.to_string());
        }
        out.push_str(&sym);
        
        let total_h = atom.num_explicit_h + atom.num_implicit_h;
        if total_h > 0 {
            out.push('H');
            if total_h > 1 {
                out.push_str(&total_h.to_string());
            }
        }
        
        if atom.formal_charge > 0 {
            out.push('+');
            if atom.formal_charge > 1 {
                out.push_str(&atom.formal_charge.to_string());
            }
        } else if atom.formal_charge < 0 {
            out.push('-');
            if atom.formal_charge < -1 {
                out.push_str(&(-atom.formal_charge).to_string());
            }
        }
        out.push(']');
    } else {
        out.push_str(&sym);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Implements test write smiles basic.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_write_smiles_basic() {
        let mol = parse_smiles("CCO").unwrap();
        let s = write_smiles(&mol);
        // Note: the order depends on parsing and tree traversal.
        // It could be C(C)O or OCC or CCO.
        assert!(!s.is_empty());
        println!("CCO -> {}", s);
    }

    #[test]
    /// Implements test write canonical smiles benzene.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_write_canonical_smiles_benzene() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        let s = write_canonical_smiles(&mol);
        assert_eq!(s, "c1ccccc1");
    }

    #[test]
    /// Implements test write canonical smiles aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles_writer.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_write_canonical_smiles_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
        let s = write_canonical_smiles(&mol);
        println!("Aspirin -> {}", s);
        assert!(s.contains("c1ccccc1") || s.contains("c1cccc(c1"));
    }
}
