//! RAE-3: run paired policy selections against one explicit oracle trace.
//! Diagnostic only: these counters are not measured latency, quality or KV use.

use flat_algebraic_attention::rae_policy_audit::{
    audit_paired_page_policies, RaePolicy, RaeStep, RAE_POLICIES,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let trace = [
        RaeStep {
            tick: 1,
            feature_state: &[0b01],
            required_pages: &[0b01],
            context_invalidated: false,
        },
        RaeStep {
            tick: 2,
            feature_state: &[0b01],
            required_pages: &[0b01],
            context_invalidated: false,
        },
        // Counterexample: a new query needs page 1 even though the same
        // source feature remains active. Neither a front nor an external
        // invalidation of the currently active page can discover page 1.
        RaeStep {
            tick: 3,
            feature_state: &[0b01],
            required_pages: &[0b10],
            context_invalidated: true,
        },
        RaeStep {
            tick: 4,
            feature_state: &[0],
            required_pages: &[0],
            context_invalidated: false,
        },
    ];
    let audit = audit_paired_page_policies(2, 2, &[0, 1], 0, &[0], &trace)?;
    println!("policy,selected_pages,required_pages,missed_required_pages,extra_selected_pages");
    for policy in RAE_POLICIES {
        let counts = audit.policy_counts(policy);
        println!(
            "{policy:?},{},{},{},{}",
            counts.selected_pages,
            counts.required_pages,
            counts.missed_required_pages,
            counts.extra_selected_pages,
        );
    }
    if !audit.policy_counts(RaePolicy::RisingOnly).has_omissions() {
        return Err("expected rising-only counterexample was not detected".into());
    }
    if audit.policy_counts(RaePolicy::AllPages).has_omissions() {
        return Err("all-pages reference must never omit any original page".into());
    }
    Ok(())
}
