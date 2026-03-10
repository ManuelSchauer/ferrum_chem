use ferrum_core::algo::electronic::effective_hybridization_pub;
use ferrum_core::algo::rings::find_sssr;
use ferrum_core::desc::extra_counts::HallKierAlpha;
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::desc::typed::Descriptor;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::types::BondType;
use ferrum_core::types::Hybridization;

#[test]
fn debug_molecules() {
    for (name, smiles) in [
        (
            "CNP0034094.1",
            "COCCC(=O)N1C[C@H]2C[C@H](C1)C1=CC=C(NC(=O)CC3=CC=C(OC)C=C3)C(=O)N1C2",
        ),
        (
            "CNP0459540.0",
            "COC1=CC=C(CCN2C=C(C(=O)N=C3N=CC=N3)C3=CC=CC=C3C2=O)C=C1",
        ),
        ("CNP0064403.0", "CN1C(=O)N=C2N=CN=C2C1=O"),
        (
            "CNP0011734.1",
            "CN1CCC(N2C[C@@H]3C[C@H](C2)C2=CC=C(C4=CC=C5C=CC=CC5=C4)C(=O)N2C3)CC1",
        ),
        ("CNP0288989.0", "O=C1C2=CC=CC=C2N=C2C3=NC4=CC=CC=C4C=C3CN21"),
        (
            "CNP0538862.1",
            "CC[C@H](C)/C=C(C)/C=C/C1=CC2=C(Cl)[C@H](OC)[C@]3(C)O[C@@](C)(OC)[C@H](C(C)=O)[C@H]3C2=CO1",
        ),
        (
            "CNP0208621.2",
            "CC1=CC(O)=C2C(=O)C3=C(O)C=C(O[C@@H]4O[C@@H](C)[C@H](O)[C@@H](O)[C@H]4O)C=C3C(=O)C2=C1",
        ),
    ] {
        let mut mol = parse_smiles(smiles).unwrap();
        let rings = find_sssr(&mut mol);
        eprintln!("\n=== {name} ({smiles}) ===");
        eprintln!("Heavy atoms: {}", mol.num_atoms());
        eprintln!("Bonds: {}", mol.num_bonds());
        let total_h: u8 = mol.atoms.iter().map(|a| a.num_implicit_h).sum();
        eprintln!("Total implicit H: {total_h}");
        eprintln!(
            "Total atoms (heavy+H): {}",
            mol.num_atoms() as u16 + total_h as u16
        );
        let aromatic: usize = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        eprintln!("Aromatic atoms: {aromatic}");
        eprintln!("SSSR rings: {}", rings.len());
        for (ri, ring) in rings.iter().enumerate() {
            let arom_atoms = ring
                .iter()
                .filter(|&&ai| mol.atoms[ai as usize].is_aromatic())
                .count();
            eprintln!(
                "  ring {ri}: size={} arom_atoms={}/{} atoms={:?}",
                ring.len(),
                arom_atoms,
                ring.len(),
                ring
            );
        }
        let mut mw_mol = parse_smiles(smiles).unwrap();
        let mw_val = MolecularWeight.calculate(&mut mw_mol);
        eprintln!("MW: {:?}", mw_val);
        eprintln!("Candidate C(=O)-N single bonds:");
        for (bi, b) in mol.bonds.iter().enumerate() {
            if b.bond_type != BondType::Single {
                continue;
            }
            let a = b.source as usize;
            let c = b.target as usize;
            let (n_idx, c_idx) =
                if mol.atoms[a].atomic_number == 7 && mol.atoms[c].atomic_number == 6 {
                    (a, c)
                } else if mol.atoms[c].atomic_number == 7 && mol.atoms[a].atomic_number == 6 {
                    (c, a)
                } else {
                    continue;
                };
            let has_c_eq_o = mol.adjacency[c_idx].iter().any(|&bi2| {
                let b2 = &mol.bonds[bi2 as usize];
                if b2.bond_type != BondType::Double {
                    return false;
                }
                let o = if b2.source as usize == c_idx {
                    b2.target as usize
                } else {
                    b2.source as usize
                };
                mol.atoms[o].atomic_number == 8
            });
            if !has_c_eq_o {
                continue;
            }
            eprintln!(
                "  bond {bi}: N{}(arom={},h={},chg={},deg={}) - C{}(arom={},chg={},deg={})",
                n_idx,
                mol.atoms[n_idx].is_aromatic(),
                mol.atoms[n_idx].total_h(),
                mol.atoms[n_idx].formal_charge,
                mol.get_degree(n_idx as u16),
                c_idx,
                mol.atoms[c_idx].is_aromatic(),
                mol.atoms[c_idx].formal_charge,
                mol.get_degree(c_idx as u16)
            );
            for &bi2 in &mol.adjacency[n_idx] {
                let b2 = &mol.bonds[bi2 as usize];
                let other = if b2.source as usize == n_idx {
                    b2.target as usize
                } else {
                    b2.source as usize
                };
                if other == c_idx {
                    continue;
                }
                eprintln!(
                    "    N neighbor {}: Z={} arom={} bond={:?}",
                    other,
                    mol.atoms[other].atomic_number,
                    mol.atoms[other].is_aromatic(),
                    b2.bond_type
                );
            }
            for &bi2 in &mol.adjacency[c_idx] {
                let b2 = &mol.bonds[bi2 as usize];
                let other = if b2.source as usize == c_idx {
                    b2.target as usize
                } else {
                    b2.source as usize
                };
                if other == n_idx {
                    continue;
                }
                eprintln!(
                    "    C neighbor {}: Z={} arom={} bond={:?}",
                    other,
                    mol.atoms[other].atomic_number,
                    mol.atoms[other].is_aromatic(),
                    b2.bond_type
                );
            }
        }
        for (i, a) in mol.atoms.iter().enumerate() {
            eprintln!(
                "  Atom {i}: Z={} arom={} impl_h={} charge={} degree={}",
                a.atomic_number,
                a.is_aromatic(),
                a.num_implicit_h,
                a.formal_charge,
                mol.get_degree(i as u16)
            );
        }
    }
}

/// Dumps every bond with its keep/exclude decision for the 12 RotatableBonds
/// failing molecules, so we can see exactly which bond differs from RDKit.
#[test]
fn debug_rotatable_failures() {
    use ferrum_core::desc::bond_count::RotatableBondCount;
    use ferrum_core::desc::typed::Descriptor;

    let cases: &[(&str, &str, u32, u32)] = &[
        (
            "CNP0128599.1 [ferrum LOW: 3 vs rdkit 4]",
            "COC1=CC(O)=C2C=CN(C=O)[C@@H](COC(C)=O)C2=C1",
            3,
            4,
        ),
        (
            "CNP0274832.0 [ferrum HIGH: 6 vs rdkit 5]",
            "CC(C)(C)C(O)C(OC1=CC=C(C2=CC=CC=C2)C=C1)N1C=NC=N1",
            6,
            5,
        ),
        (
            "CNP0584960.0 [ferrum HIGH: 12 vs rdkit 11]",
            "CC1=CC(C(C)(C)CC(C)(C)C)=CC=C1OCCOCC[N+](C)(C)CC1=CC=CC=C1",
            12,
            11,
        ),
        (
            "CNP0065983.0 [ferrum HIGH: 4 vs rdkit 3]",
            "CC(C)(C)C1=CC=C(C2=CC(=O)C3=CC=C(NC(=O)C4=CC=CC=C4Cl)C=C3O2)C=C1",
            4,
            3,
        ),
        (
            "CNP0059824.1 [ferrum HIGH: 6 vs rdkit 5]",
            "CC(C)(C)OC(=O)[C@@H](C(=O)CC1=CC=CC=C1)C1=CC=CC=C1",
            6,
            5,
        ),
        (
            "CNP0246017.1 [ferrum LOW: 11 vs rdkit 12]",
            "C=C1[C@@H]([C@@]2(C)[C@@H](OC(C)=O)C[C@]3(O)OC[C@@](C)(O3)[C@@H]2CC(=O)OC)[C@@H](OC=O)[C@H](OC(=O)C(=O)[C@H](C)CC)[C@@]2(C)[C@H](C3=COC=C3)C[C@@H](OC(C)=O)[C@]12O",
            11,
            12,
        ),
        (
            "CNP0248430.1 [ferrum HIGH: 8 vs rdkit 7]",
            "CC(C)(C)CC(=O)N1CC[C@@H](CC(=O)O)[C@@H](CCN2CCN(C3=CC=CC=C3)CC2)C1",
            8,
            7,
        ),
        (
            "CNP0071270.1 [ferrum HIGH: 2 vs rdkit 1]",
            "CC(C)(C)OC(=O)N[C@@H]1C=CC(=O)C1",
            2,
            1,
        ),
        (
            "CNP0519304.1 [ferrum HIGH: 5 vs rdkit 4]",
            "CC(=O)NC[C@H]1C(=O)N2C[C@@H](NC(C)=O)C[C@H]2CN1CC(C)(C)C",
            5,
            4,
        ),
        (
            "CNP0059848.0 [ferrum HIGH: 6 vs rdkit 4]",
            "C#CCOC(=O)NNC(=O)C1=CC=CC(C(=O)NNC(=O)OCC#C)=C1",
            6,
            4,
        ),
        (
            "CNP0434863.1 [ferrum HIGH: 3 vs rdkit 2]",
            "CC(=O)N1C(=O)C=CN([C@H]2C[C@H](O)[C@@H](CO)O2)C1=O",
            3,
            2,
        ),
        (
            "CNP0177170.1 [ferrum HIGH: 13 vs rdkit 11]",
            "CC(C)[C@@H]1NC(=O)[C@@H]2[C@H](C)CCN2C(=O)CN/C(=N\\[C@H](C(=O)N[C@H](C(=O)N[C@H](CC(=O)O)C2=NC=CS2)[C@@H](C)C2=CC=CC=C2)C(C)(C)C)[C@H](C(C)(C)C)NC1=O",
            13,
            11,
        ),
    ];

    // Replicate the filtering logic from compute_rotatable_bond_count so we can
    // annotate why each bond is kept or skipped.
    fn heavy_degrees(mol: &ferrum_core::mol::molecule::Molecule) -> Vec<usize> {
        (0..mol.num_atoms())
            .map(|i| {
                mol.adjacency[i]
                    .iter()
                    .filter(|&&bi| {
                        let b = &mol.bonds[bi as usize];
                        let nb = if b.source == i as u16 {
                            b.target
                        } else {
                            b.source
                        } as usize;
                        mol.atoms[nb].atomic_number > 1
                    })
                    .count()
            })
            .collect()
    }

    fn bond_in_cycle(mol: &ferrum_core::mol::molecule::Molecule) -> Vec<bool> {
        use ferrum_core::types::BondType;
        let n = mol.num_atoms();
        let m = mol.num_bonds();
        let mut disc = vec![usize::MAX; n];
        let mut low = vec![0usize; n];
        let mut is_bridge = vec![false; m];
        let mut time = 0usize;

        fn dfs(
            mol: &ferrum_core::mol::molecule::Molecule,
            u: usize,
            parent_bond: Option<usize>,
            time: &mut usize,
            disc: &mut [usize],
            low: &mut [usize],
            is_bridge: &mut [bool],
        ) {
            disc[u] = *time;
            low[u] = *time;
            *time += 1;
            for &bi in &mol.adjacency[u] {
                let bidx = bi as usize;
                if Some(bidx) == parent_bond {
                    continue;
                }
                let bond = &mol.bonds[bidx];
                let v = if bond.source == u as u16 {
                    bond.target
                } else {
                    bond.source
                } as usize;
                if disc[v] == usize::MAX {
                    dfs(mol, v, Some(bidx), time, disc, low, is_bridge);
                    low[u] = low[u].min(low[v]);
                    if low[v] > disc[u] {
                        is_bridge[bidx] = true;
                    }
                } else {
                    low[u] = low[u].min(disc[v]);
                }
            }
        }
        for u in 0..n {
            if disc[u] == usize::MAX {
                dfs(mol, u, None, &mut time, &mut disc, &mut low, &mut is_bridge);
            }
        }
        is_bridge.into_iter().map(|b| !b).collect()
    }

    fn has_triple_nbr(mol: &ferrum_core::mol::molecule::Molecule) -> Vec<bool> {
        use ferrum_core::types::BondType;
        (0..mol.num_atoms())
            .map(|i| {
                mol.adjacency[i]
                    .iter()
                    .any(|&bi| mol.bonds[bi as usize].bond_type == BondType::Triple)
            })
            .collect()
    }

    fn is_carbonyl_hetero(mol: &ferrum_core::mol::molecule::Molecule, bond_idx: usize) -> bool {
        use ferrum_core::types::BondType;
        let bond = &mol.bonds[bond_idx];
        if bond.bond_type != BondType::Single {
            return false;
        }
        let a = bond.source as usize;
        let b = bond.target as usize;
        let za = mol.atoms[a].atomic_number;
        let zb = mol.atoms[b].atomic_number;
        let (hetero_idx, c_idx) = if za == 6 && matches!(zb, 7 | 8 | 16) {
            (b, a)
        } else if zb == 6 && matches!(za, 7 | 8 | 16) {
            (a, b)
        } else {
            return false;
        };
        if mol.atoms[hetero_idx].is_aromatic() {
            return false;
        }
        for &bi in &mol.adjacency[c_idx] {
            if bi as usize == bond_idx {
                continue;
            }
            let b2 = &mol.bonds[bi as usize];
            if b2.bond_type != BondType::Double {
                continue;
            }
            let other = if b2.source == c_idx as u16 {
                b2.target
            } else {
                b2.source
            } as usize;
            if matches!(mol.atoms[other].atomic_number, 7 | 8 | 16) {
                return true;
            }
        }
        false
    }

    for (name, smiles, ferrum_count, rdkit_count) in cases {
        let mut mol = parse_smiles(smiles).expect("parse");
        let in_cycle = bond_in_cycle(&mol);
        let hdeg = heavy_degrees(&mol);
        let has_triple = has_triple_nbr(&mol);

        eprintln!("\n══════════════════════════════════════════════════");
        eprintln!("  {name}");
        eprintln!("  SMILES: {smiles}");
        eprintln!("  ferrum={ferrum_count}  rdkit={rdkit_count}");
        eprintln!("──────────────────────────────────────────────────");
        eprintln!(
            "  {:<4} {:<6} {:<18} {:>5} {:>5}  reason",
            "bi", "type", "atoms(Z,ar,deg)", "cycle", "kept"
        );

        let mut kept = 0u32;
        for (bi, bond) in mol.bonds.iter().enumerate() {
            use ferrum_core::types::BondType;
            let a = bond.source as usize;
            let b = bond.target as usize;
            let za = mol.atoms[a].atomic_number;
            let zb = mol.atoms[b].atomic_number;

            // reasons for exclusion
            let not_single = bond.bond_type != BondType::Single || bond.is_aromatic();
            let cycle = in_cycle[bi];
            let hydrogen = za <= 1 || zb <= 1;
            let terminal = hdeg[a] <= 1 || hdeg[b] <= 1;
            let triple = has_triple[a] || has_triple[b];
            let carbonyl = is_carbonyl_hetero(&mol, bi);

            let keep = !not_single && !cycle && !hydrogen && !terminal && !triple && !carbonyl;
            if keep {
                kept += 1;
            }

            // Only print bonds that involve at least one non-H atom and are not obvious H-bonds
            if za <= 1 && zb <= 1 {
                continue;
            }
            if (za <= 1 || zb <= 1) && !keep {
                continue;
            } // skip H bonds that are excluded trivially

            let reason = if not_single {
                "not-single"
            } else if hydrogen {
                "has-H"
            } else if cycle {
                "in-ring"
            } else if terminal {
                "terminal"
            } else if triple {
                "triple-adj"
            } else if carbonyl {
                "carbonyl-X"
            } else {
                "KEPT"
            };

            let flag = if keep { "✓" } else { "✗" };
            eprintln!(
                "  [{bi:3}] {:<8} A{a}(Z={za},ar={},hd={}) — A{b}(Z={zb},ar={},hd={})  cycle={cycle}  {flag}  {reason}",
                format!("{:?}", bond.bond_type),
                mol.atoms[a].is_aromatic() as u8,
                hdeg[a],
                mol.atoms[b].is_aromatic() as u8,
                hdeg[b],
            );
        }

        eprintln!("──────────────────────────────────────────────────");
        eprintln!(
            "  ferrum total (recomputed): {kept}  descriptor: {ferrum_count}  rdkit: {rdkit_count}"
        );

        // Also confirm via descriptor
        let mut mol2 = parse_smiles(smiles).expect("parse2");
        let desc_val = RotatableBondCount.calculate(&mut mol2);
        eprintln!("  descriptor output: {:?}", desc_val);
    }
}

/// Dumps per-atom hybridization and individual alpha contributions for the
/// 11 molecules that still fail the HallKierAlpha RDKit comparison, so we
/// can identify exactly which atom is misclassified.
#[test]
fn debug_hka_failures() {
    // (name, smiles, rdkit_hka, ferrum_hka_expected_diff)
    let cases: &[(&str, &str, f64, f64)] = &[
        (
            "CNP0537267.2 (diff=0.13)",
            "CCCCOC[C@@H]1C[C@@]2(C[C@](C)(C(=O)CSC3=NN=C(C4=CC=CC=C4Cl)N3)OC2=O)C(=O)O1",
            -2.43,
            -0.13,
        ),
        (
            "CNP0446854.2 (diff=0.13)",
            "C#CCNC(=O)[C@H]1C[C@@H](O)[C@H](O)[C@@H]2N=C(NC3=CC=C4OCOC4=C3)S[C@@H]21",
            -1.40,
            -0.13,
        ),
        (
            "CNP0057583.1 (diff=0.16)",
            "COC(=O)[C@@H](C)NC(=O)[C@H]1CN(S(=O)(=O)C2=CC=CC=C2)CCN1C(=O)N1CCOCC1",
            -2.70,
            -0.16,
        ),
        (
            "CNP0342356.0 (diff=0.13)",
            "CCOC(=O)C1=C(N)N(S(=O)(=O)C2=CC=C(OC)C=C2)N=C1SC",
            -2.20,
            -0.13,
        ),
        (
            "CNP0064623.2 (diff=0.16)",
            "CC=CC1=CC=C2[C@@H]3C[C@H](CN(S(=O)(=O)CCC)C3)CN2C1=O",
            -1.40,
            -0.16,
        ),
        (
            "CNP0500772.4 (diff=0.13)",
            "CSC1=CC=CC(NC(=O)[C@@H]2[C@@H]3C=C[C@]4(O3)[C@@H]2C(=O)N([C@@H]2CCC[C@@H](C)[C@H]2C)[C@H]4C(=O)NC2CCCCC2)=C1",
            -2.32,
            -0.13,
        ),
        (
            "CNP0014928.1 (diff=0.16)",
            "CC1=CC=C2C(=C1)N=C1COC[C@H](CNS(=O)(=O)C3=CC=CC=C3)N12",
            -2.22,
            -0.16,
        ),
        (
            "CNP0511405.1 (diff=0.16)",
            "COCCN1[C@@H]2CN(C(=O)NCC3=CC=CC=C3)C[C@@H]2OCCS1(=O)=O",
            -1.68,
            -0.16,
        ),
        (
            "CNP0513418.4 (diff=0.13)",
            "O=C1C2=N[C@H]3C[C@@]45C(=CC(=O)[C@H]([C@@H]4O)N4CCC6=CNC1=C6C4=C25)S3",
            -2.12,
            -0.13,
        ),
        (
            "CNP0496638.0 (diff=0.13)",
            "CC1(C)C(=CC=C2CCCC(C=CC3=[N+](CCCCS(=O)(=O)O)C4=CC=CC=C4C3(C)C)=C2SCCC(=O)NCCOCCNC(=O)OCC2C3=CC=CC=C3C3=CC=CC=C32)N(CCCCS(=O)(=O)O)C2=CC=CC=C21",
            -5.82,
            -0.13,
        ),
        (
            "CNP0084016.1 (diff=0.13)",
            "CC(=O)N1CCN(C(=O)C[C@@H]2CCNC[C@@H]2CC2=NOC(CSC3=CC=CC=C3)=C2)CC1",
            -2.32,
            -0.13,
        ),
    ];

    // alpha lookup mirrors compute_hall_kier_alpha exactly
    let atom_alpha = |z: u8, hyb: Hybridization| -> f64 {
        let hyb_idx = match hyb {
            Hybridization::SP => 0usize,
            Hybridization::SP2 => 1,
            _ => 2,
        };
        match z {
            6 => [-0.22, -0.13, 0.0][hyb_idx],
            7 => [-0.29, -0.20, -0.04][hyb_idx],
            8 => {
                if hyb_idx == 1 {
                    -0.20
                } else {
                    -0.04
                }
            }
            16 => {
                if hyb_idx == 1 {
                    0.22
                } else {
                    0.35
                }
            }
            15 => {
                if hyb_idx == 1 {
                    0.30
                } else {
                    0.43
                }
            }
            9 => -0.07,
            17 => 0.29,
            35 => 0.48,
            53 => 0.73,
            _ => 0.0,
        }
    };

    for (name, smiles, rdkit_val, expected_diff) in cases {
        let mut mol = parse_smiles(smiles).expect("parse");
        eprintln!("\n══════════════════════════════════════════════════");
        eprintln!("  {name}");
        eprintln!("  SMILES: {smiles}");
        eprintln!("  RDKit HKA: {rdkit_val:.4}   expected ferrum diff: {expected_diff:+.2}");
        eprintln!("──────────────────────────────────────────────────");

        let mut total_alpha = 0.0f64;
        for i in 0..mol.atoms.len() {
            let a = &mol.atoms[i];
            if a.atomic_number == 0 || a.atomic_number == 1 {
                continue;
            }
            let hyb = effective_hybridization_pub(&mol, i);
            let alpha = atom_alpha(a.atomic_number, hyb);
            total_alpha += alpha;

            // Collect neighbor info for context
            let nbr_info: Vec<String> = mol.adjacency[i]
                .iter()
                .map(|&bi| {
                    let b = &mol.bonds[bi as usize];
                    let ni = if b.source as usize == i {
                        b.target as usize
                    } else {
                        b.source as usize
                    };
                    format!(
                        "{}(Z={},ar={},{:?})",
                        ni,
                        mol.atoms[ni].atomic_number,
                        mol.atoms[ni].is_aromatic(),
                        b.bond_type
                    )
                })
                .collect();

            // Flag atoms that contribute a non-SP3 value for heteratoms and non-zero for C/N
            let flag = match a.atomic_number {
                6 | 7 => {
                    if alpha != 0.0
                        && a.atomic_number == 6
                        && matches!(hyb, Hybridization::SP2 | Hybridization::SP)
                    {
                        "  ←C-pi"
                    } else if a.atomic_number == 7
                        && matches!(hyb, Hybridization::SP2 | Hybridization::SP)
                    {
                        "  ←N-pi"
                    } else {
                        ""
                    }
                }
                8 | 16 | 15 => {
                    if matches!(hyb, Hybridization::SP2) {
                        "  ←HETERO-SP2"
                    } else {
                        ""
                    }
                }
                _ => "",
            };
            eprintln!(
                "  [{i:2}] Z={:2} ar={} hyb={:4?}  alpha={:+.4}{}",
                a.atomic_number,
                a.is_aromatic() as u8,
                hyb,
                alpha,
                flag
            );
            if !flag.is_empty() {
                eprintln!("        neighbors: {}", nbr_info.join(", "));
            }
        }

        // Also invoke the descriptor directly
        let mut mol2 = parse_smiles(smiles).expect("parse2");
        let ferrum_hka = match HallKierAlpha.calculate(&mut mol2) {
            ferrum_core::desc::typed::DescriptorValue::Float(v) => v as f64,
            _ => f64::NAN,
        };

        eprintln!("──────────────────────────────────────────────────");
        eprintln!("  atom-loop total alpha : {total_alpha:+.4}");
        eprintln!("  descriptor HKA        : {ferrum_hka:+.4}");
        eprintln!("  rdkit HKA             : {rdkit_val:+.4}");
        eprintln!("  actual diff (f-r)     : {:+.4}", ferrum_hka - rdkit_val);
    }
}
