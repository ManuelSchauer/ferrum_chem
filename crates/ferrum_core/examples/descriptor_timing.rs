//! Single-threaded time per batch descriptor, streaming a SMILES file.
//!
//! ```text
//! cargo run --release -p ferrum_core --example descriptor_timing -- molecules.smi
//! FERRUM_ONLY=WeightedPath,MaccsKeys cargo run --release ...   # only these descriptors
//! FERRUM_TSV=1 cargo run --release ...    # machine-readable: `name<TAB>us/mol` for all
//! FERRUM_LIST=1 cargo run --release ... - # list `name<TAB>feature,feature,...` and exit
//! ```
//!
//! Used by `scripts/benchmark.py`.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use ferrum_core::desc::batch::BatchDescriptor;
use ferrum_core::desc::context::CalculationContext;
use ferrum_core::io::smiles::parse_smiles;

fn main() {
    if std::env::var_os("FERRUM_LIST").is_some() {
        for d in BatchDescriptor::all() {
            println!("{}\t{}", d.name(), d.feature_names().join(","));
        }
        return;
    }
    let path = std::env::args().nth(1).expect("usage: descriptor_timing <file.smi>");
    let reader = BufReader::new(std::fs::File::open(&path).expect("cannot open input"));
    let only = std::env::var("FERRUM_ONLY").ok();
    let descriptors: Vec<BatchDescriptor> = BatchDescriptor::all()
        .iter()
        .copied()
        .filter(|d| only.as_deref().is_none_or(|o| o.split(',').any(|n| n == d.name())))
        .collect();
    let mut times = vec![Duration::ZERO; descriptors.len()];
    let mut parse = Duration::ZERO;
    let mut ctx = CalculationContext::new();
    let mut n = 0usize;
    let mut slowest: Vec<(Duration, &'static str, String)> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read error");
        let t = Instant::now();
        let Ok(mol) = parse_smiles(&line) else { continue };
        parse += t.elapsed();
        n += 1;
        // as `DescriptorCalculator` does between molecules
        ctx.trim();
        ctx.reset_for_new_molecule();
        for (i, d) in descriptors.iter().enumerate() {
            let t = Instant::now();
            std::hint::black_box(d.calculate(&mol, &mut ctx));
            let dt = t.elapsed();
            times[i] += dt;
            if slowest.len() < 10 || dt > slowest[slowest.len() - 1].0 {
                slowest.push((dt, d.name(), line.clone()));
                slowest.sort_by_key(|s| std::cmp::Reverse(s.0));
                slowest.truncate(10);
            }
        }
    }
    let total: Duration = times.iter().sum();
    let us = |d: Duration| d.as_secs_f64() * 1e6 / n.max(1) as f64;
    if std::env::var_os("FERRUM_TSV").is_some() {
        println!("molecules\t{n}");
        println!("parse\t{}", us(parse));
        for (d, &t) in descriptors.iter().zip(&times) {
            println!("{}\t{}", d.name(), us(t));
        }
        return;
    }
    println!("{n} molecules; parse {:.1} us/mol; descriptors {:.1} us/mol", us(parse), us(total));
    let mut order: Vec<usize> = (0..descriptors.len()).collect();
    order.sort_by(|&a, &b| times[b].cmp(&times[a]));
    for i in order.into_iter().take(25) {
        println!("{:>8.1} us  {:>5.1}%  {}", us(times[i]), 100.0 * times[i].as_secs_f64() / total.as_secs_f64(), descriptors[i].name());
    }
    println!("slowest single calculations:");
    for (dt, name, smiles) in &slowest {
        println!("{:>10.1} ms  {name}  {}", dt.as_secs_f64() * 1e3, smiles.chars().take(80).collect::<String>());
    }
}
