//! Shared column-by-column parity harness (see rdkit_parity.rs / cdk_parity.rs).
#![allow(dead_code)]

use ferrum_core::desc::batch::BatchDescriptor;
use ferrum_core::desc::context::CalculationContext;
use ferrum_core::desc::typed::DescriptorValue;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::mol::molecule::Molecule;
use ferrum_core::types::Hybridization;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

/// Values are f64 end to end, so references must match to ~double precision.
const ABS_TOL: f64 = 1e-9;
const REL_TOL: f64 = 1e-9;
const MAX_EXAMPLES: usize = 5;

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn to_f64s(v: DescriptorValue) -> Vec<f64> {
    match v {
        DescriptorValue::Int(x) => vec![x as f64],
        DescriptorValue::Float(x) => vec![x],
        DescriptorValue::IntVec(v) => v.into_iter().map(|x| x as f64).collect(),
        DescriptorValue::IntU16Vec(v) => v.into_iter().map(|x| x as f64).collect(),
        DescriptorValue::FloatVec(v) => v,
        DescriptorValue::Binary(bits) => bits
            .iter()
            .by_vals()
            .map(|b| if b { 1.0 } else { 0.0 })
            .collect(),
    }
}

fn hyb(h: Hybridization) -> &'static str {
    match h {
        Hybridization::S => "S",
        Hybridization::SP => "SP",
        Hybridization::SP2 => "SP2",
        Hybridization::SP3 => "SP3",
        Hybridization::SP3D => "SP3D",
        Hybridization::SP3D2 => "SP3D2",
        Hybridization::Unspecified => "U",
    }
}

fn perception(mol: &Molecule) -> String {
    mol.atoms
        .iter()
        .map(|a| {
            format!(
                "{},{},{},{},{},{},{}",
                a.atomic_number,
                a.formal_charge,
                a.is_aromatic() as u8,
                a.total_h(),
                hyb(a.hybridization),
                a.num_radicals(),
                a.is_in_ring() as u8
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_ref(s: &str) -> f64 {
    match s {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => s
            .parse()
            .unwrap_or_else(|_| panic!("bad reference value {s:?}")),
    }
}

fn close(expected: f64, actual: f64) -> bool {
    if expected.is_nan() || actual.is_nan() {
        return expected.is_nan() && actual.is_nan();
    }
    if expected.is_infinite() || actual.is_infinite() {
        return expected == actual;
    }
    (expected - actual).abs() <= ABS_TOL + REL_TOL * expected.abs()
}

/// How a reference column is checked.
enum Check {
    /// Scalar feature: (descriptor index, offset within its output).
    Value(usize, usize),
    /// Fingerprint stored as on-bit list: (descriptor index, bit offset).
    Bits(usize, usize),
    Perception,
    /// RDKit column without a FerrumChem counterpart (reported, not checked).
    Missing,
}

#[derive(Default)]
struct ColumnStats {
    mismatches: usize,
    examples: Vec<String>,
}

fn load_baseline(file: &str) -> BTreeMap<String, usize> {
    let path = data_path(file);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return BTreeMap::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (k, v) = l.split_once('\t').expect("baseline line: column<TAB>count");
            (k.to_string(), v.trim().parse().expect("baseline count"))
        })
        .collect()
}

fn write_baseline(file: &str, label: &str, stats: &BTreeMap<String, ColumnStats>, total: usize) {
    let mut out = format!(
        "# Allowed number of molecules per column that differ from {label} (tests/common/parity.rs).\n\
         # Regenerate with FERRUM_UPDATE_PARITY_BASELINE=1; only ever lower these numbers.\n",
    );
    out.push_str(&format!("# molecules: {total}\n"));
    for (col, s) in stats {
        if s.mismatches > 0 {
            out.push_str(&format!("{col}\t{}\n", s.mismatches));
        }
    }
    std::fs::write(data_path(file), out).unwrap();
}

/// Compares FerrumChem against a reference CSV (one molecule per row, columns
/// named like FerrumChem features). Empty reference cells are not compared.
pub fn run_parity(label: &str, reference_file: &str, baseline_file: &str) {
    run(label, None, reference_file, baseline_file, &|c| c.to_string());
}

/// Like [`run_parity`], but each reference column name is mapped to the
/// FerrumChem column through `column` (used for Mordred's names).
pub fn run_parity_mapped(label: &str, reference_file: &str, baseline_file: &str, column: &dyn Fn(&str) -> String) {
    run(label, None, reference_file, baseline_file, column);
}

/// Like [`run_parity`], but the molecules are read from an SDF file with
/// FerrumChem's SDF reader. The reference's first column `name` holds the
/// record titles, in file order.
pub fn run_parity_sdf(label: &str, sdf_file: &str, reference_file: &str, baseline_file: &str) {
    run(label, Some(sdf_file), reference_file, baseline_file, &|c| c.to_string());
}

fn run(
    label: &str,
    sdf_file: Option<&str>,
    reference_file: &str,
    baseline_file: &str,
    column: &dyn Fn(&str) -> String,
) {
    let descriptors = BatchDescriptor::all();
    let mut by_name: HashMap<String, (usize, usize)> = HashMap::new();
    for (di, d) in descriptors.iter().enumerate() {
        for (k, name) in d.feature_names().into_iter().enumerate() {
            by_name.insert(name, (di, k));
        }
    }
    let fp_descriptor = |col: &str| -> (usize, usize) {
        let (desc, offset) = match col {
            "fp:MACCS" => (BatchDescriptor::MaccsKeys, 1), // MACCS key i -> output index i-1
            "fp:ECFP4" => (BatchDescriptor::Ecfp4, 0),
            "fp:FCFP4" => (BatchDescriptor::Fcfp4, 0),
            "fp:AtomPair" => (BatchDescriptor::AtomPairFingerprint, 0),
            "fp:TopoTorsion" => (BatchDescriptor::TopologicalTorsionFingerprint, 0),
            "fp:PubChem" => (BatchDescriptor::PubChem881, 0),
            other => panic!("unknown fingerprint column {other}"),
        };
        (descriptors.iter().position(|&d| d == desc).unwrap(), offset)
    };

    let mut rdr = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .from_path(data_path(reference_file))
        .unwrap_or_else(|e| panic!("{label} reference {reference_file}: {e}"));
    let headers = rdr.headers().unwrap().clone();
    assert_eq!(
        &headers[0],
        if sdf_file.is_some() { "name" } else { "smiles" }
    );
    let mut sdf_records = sdf_file.map(|f| {
        ferrum_core::io::sdf::read_sdf(&data_path(f))
            .unwrap_or_else(|e| panic!("{f}: {e}"))
            .into_iter()
    });
    let checks: Vec<(String, Check)> = headers
        .iter()
        .skip(1)
        .map(|col| {
            let check = if col == "perception" {
                Check::Perception
            } else if col.starts_with("fp:") {
                let (di, off) = fp_descriptor(col);
                Check::Bits(di, off)
            } else {
                match by_name.get(&column(col)) {
                    Some(&(di, k)) => Check::Value(di, k),
                    None => Check::Missing,
                }
            };
            (col.to_string(), check)
        })
        .collect();
    let needed: Vec<bool> = (0..descriptors.len())
        .map(|di| {
            checks
                .iter()
                .any(|(_, c)| matches!(c, Check::Value(d, _) | Check::Bits(d, _) if *d == di))
        })
        .collect();

    let missing: Vec<&str> = checks
        .iter()
        .filter(|(_, c)| matches!(c, Check::Missing))
        .map(|(n, _)| n.as_str())
        .collect();
    let mut stats: BTreeMap<String, ColumnStats> = checks
        .iter()
        .filter(|(_, c)| !matches!(c, Check::Missing))
        .map(|(c, _)| (c.clone(), ColumnStats::default()))
        .collect();
    let mut ctx = CalculationContext::new();
    let mut total = 0usize;
    let mut record = csv::StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        total += 1;
        let smiles = &record[0]; // SMILES, or the SDF record title
        let mut note = |col: &str, msg: String| {
            let s = stats.get_mut(col).unwrap();
            s.mismatches += 1;
            if s.examples.len() < MAX_EXAMPLES {
                s.examples.push(format!("{smiles}: {msg}"));
            }
        };
        let rdkit_failed = record.get(1) == Some("ERR");
        let parsed = match sdf_records.as_mut() {
            Some(records) => {
                let rec = records
                    .next()
                    .expect("SDF has fewer records than the reference");
                assert_eq!(
                    rec.name, smiles,
                    "SDF record order differs from the reference"
                );
                rec.molecule
            }
            None => parse_smiles(smiles),
        };
        let mol = match parsed {
            Ok(m) if !rdkit_failed => m,
            Ok(_) => {
                note("perception", "RDKit rejects, FerrumChem accepts".into());
                continue;
            }
            Err(e) if !rdkit_failed => {
                note("perception", format!("FerrumChem rejects: {e}"));
                continue;
            }
            Err(_) => continue,
        };
        ctx.reset_for_new_molecule();
        let values: Vec<Option<Vec<f64>>> = descriptors
            .iter()
            .enumerate()
            .map(|(di, d)| needed[di].then(|| to_f64s(d.calculate(&mol, &mut ctx))))
            .collect();

        for (ci, (col, check)) in checks.iter().enumerate() {
            let expected = &record[ci + 1];
            if expected.is_empty() && !matches!(check, Check::Perception) {
                continue; // reference could not compute this value
            }
            match check {
                Check::Missing => {}
                Check::Perception => {
                    let got = perception(&mol);
                    if got != expected {
                        note(col, "atom typing differs".into());
                    }
                }
                Check::Value(di, k) => {
                    let exp = parse_ref(expected);
                    // a descriptor whose output is shorter than its feature names is a mismatch, not a panic
                    let got = values[*di]
                        .as_ref()
                        .unwrap()
                        .get(*k)
                        .copied()
                        .unwrap_or(f64::NAN);
                    if !close(exp, got) {
                        note(col, format!("rdkit={exp} ferrum={got}"));
                    }
                }
                Check::Bits(di, offset) => {
                    let bits = values[*di].as_ref().unwrap();
                    let mut exp: Vec<usize> = expected
                        .split_whitespace()
                        .map(|b| b.parse::<usize>().unwrap() - offset)
                        .collect();
                    exp.sort_unstable();
                    let got: Vec<usize> = (0..bits.len()).filter(|&i| bits[i] != 0.0).collect();
                    if exp != got {
                        let missing = exp.iter().filter(|b| !got.contains(b)).count();
                        let extra = got.iter().filter(|b| !exp.contains(b)).count();
                        note(col, format!("{missing} bits missing, {extra} extra"));
                    }
                }
            }
        }
    }

    if let Some(mut records) = sdf_records {
        assert!(
            records.next().is_none(),
            "SDF has more records than the reference"
        );
    }

    if std::env::var("FERRUM_UPDATE_PARITY_BASELINE").is_ok() {
        write_baseline(baseline_file, label, &stats, total);
        println!("baseline written");
        return;
    }

    let baseline = load_baseline(baseline_file);
    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    println!("\n{label} parity on {total} molecules (column: mismatching molecules / allowed)");
    for (col, s) in &stats {
        let allowed = baseline.get(col).copied().unwrap_or(0);
        if s.mismatches > 0 || allowed > 0 {
            println!("  {col:28} {:5} / {allowed}", s.mismatches);
            if std::env::var("FERRUM_PARITY_VERBOSE").is_ok() {
                for e in &s.examples {
                    println!("      {e}");
                }
            }
        }
        if s.mismatches > allowed {
            regressions.push(format!(
                "{col}: {} mismatches (allowed {allowed})\n      {}",
                s.mismatches,
                s.examples.join("\n      ")
            ));
        } else if s.mismatches < allowed {
            improvements.push(format!("{col}: {} -> {}", allowed, s.mismatches));
        }
    }
    let exact = stats.values().filter(|s| s.mismatches == 0).count();
    println!(
        "  {exact} of {} columns match {label} on every molecule",
        stats.len()
    );
    if !missing.is_empty() {
        const SHOWN: usize = 40;
        let verbose = std::env::var("FERRUM_PARITY_VERBOSE").is_ok();
        let shown = if verbose { missing.len() } else { missing.len().min(SHOWN) };
        println!(
            "  {} of {} {label} columns implemented; {} not implemented in FerrumChem: {}{}",
            stats.len() - stats.contains_key("perception") as usize,
            checks.len() - stats.contains_key("perception") as usize,
            missing.len(),
            missing[..shown].join(" "),
            if shown < missing.len() { " ... (FERRUM_PARITY_VERBOSE=1 lists all)" } else { "" }
        );
    }
    if !improvements.is_empty() {
        println!(
            "\nImproved columns (tighten with FERRUM_UPDATE_PARITY_BASELINE=1):\n  {}",
            improvements.join("\n  ")
        );
    }
    assert!(
        regressions.is_empty(),
        "{label} parity regressions:\n  {}",
        regressions.join("\n  ")
    );
}
