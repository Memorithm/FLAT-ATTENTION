//! RAE-2 executable demonstration of logical page wakes.
//! Does not execute numeric attention or claim physical memory savings.

use flat_algebraic_attention::edge_pages::RisingEdgePageRouter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Four Boolean features -> three logical pages. Features 0 and 1
    // intentionally address the same page, so simultaneous rising edges
    // must coalesce to one page-wake event.
    let mut router = RisingEdgePageRouter::new(4, 0, &[0], 3, &[0, 0, 1, 2])?;
    let mut wakes = [0_u64];
    let mut new_latches = [0_u64];

    let stream = [
        (1, 0b0011_u64), // two rising features; one page wake
        (2, 0b0011_u64), // sustained high; zero front, zero page wake
        (3, 0b0001_u64), // falling edge; no positive wake
        (4, 0b0011_u64), // rising again; existing logical latch is reused
        (5, 0b0111_u64), // new page is activated
    ];

    println!("tick active rising falling wakes new_latches persistent");
    for (tick, state) in stream {
        let record = router.sample_into(tick, &[state], &mut wakes, &mut new_latches)?;
        println!(
            "{} {} {} {} {} {} {:#05b}",
            record.tick,
            record.active_features,
            record.rising_features,
            record.falling_features,
            record.woken_pages,
            record.newly_latched_pages,
            router.latched_pages()[0],
        );
    }
    assert_eq!(router.latched_pages(), &[0b011]);
    Ok(())
}
