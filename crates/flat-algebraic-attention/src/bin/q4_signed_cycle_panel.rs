//! Reproducible CSV export of the frozen Q4-TOPO-4 signed-cycle panel.
//!
//! Usage:
//! cargo run -p flat-algebraic-attention --bin q4_signed_cycle_panel
//! cargo run -p flat-algebraic-attention --bin q4_signed_cycle_panel -- --summary

use flat_algebraic_attention::ising_matched_cycle_panel::run_matched_signed_cycle_panel;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "--cases".to_string());
    if args.next().is_some() || (mode != "--summary" && mode != "--cases") {
        return Err("usage: q4_signed_cycle_panel [--cases|--summary]".into());
    }
    let panel = run_matched_signed_cycle_panel()?;
    if mode == "--summary" {
        println!(
            "schema,family,activation_cost,graphs,perturbation_trials,exact_bit_recalls,effective_spin_recalls,fixed_point_trials,q4_forward_barrier_sum,q4_backward_barrier_sum,binary_comparable_graphs,binary_forward_barrier_sum"
        );
        for row in panel.aggregates {
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                panel.schema,
                row.family.as_str(),
                row.activation_cost,
                row.graph_count,
                row.perturbation_trials,
                row.exact_bit_recalls,
                row.effective_spin_recalls,
                row.fixed_point_trials,
                row.exact_q4_forward_barrier_sum,
                row.exact_q4_backward_barrier_sum,
                row.binary_comparable_graphs,
                row.exact_binary_forward_barrier_sum,
            );
        }
    } else {
        println!(
            "schema,negative_edge_mask,family,activation_cost,cycle_rank,negative_fundamental_cycles,positive_root_attractor,negative_root_attractor,positive_settle_sweeps,negative_settle_sweeps,q4_energy_positive,q4_energy_negative,q4_communication_height,q4_barrier_positive,q4_barrier_negative,binary_barrier_positive,positive_exact_bit_recalls,positive_effective_spin_recalls,positive_fixed_point_trials,negative_exact_bit_recalls,negative_effective_spin_recalls,negative_fixed_point_trials"
        );
        for row in panel.cases {
            let states = |spins: &[flat_algebraic_attention::ising_quaternary::Spin4]| {
                spins
                    .iter()
                    .map(|state| char::from(b'0' + *state as u8))
                    .collect::<String>()
            };
            let binary = row
                .exact_binary
                .as_ref()
                .map(|b| b.barrier_above_initial.to_string())
                .unwrap_or_default();
            println!(
                "{},{:05b},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                panel.schema,
                row.negative_edge_mask,
                row.family.as_str(),
                row.activation_cost,
                row.cycle_rank,
                row.negative_fundamental_cycles,
                states(&row.positive_root_attractor),
                states(&row.negative_root_attractor),
                row.positive_root_settle_sweeps,
                row.negative_root_settle_sweeps,
                row.exact_q4.initial_energy,
                row.exact_q4.target_energy,
                row.exact_q4.communication_height,
                row.exact_q4.barrier_above_initial,
                row.exact_q4.communication_height - row.exact_q4.target_energy,
                binary,
                row.positive_root_recall.exact_recall_count,
                row.positive_root_recall.effective_state_recall_count,
                row.positive_root_recall.fixed_point_count,
                row.negative_root_recall.exact_recall_count,
                row.negative_root_recall.effective_state_recall_count,
                row.negative_root_recall.fixed_point_count,
            );
        }
    }
    Ok(())
}
