use flat_attention::paged_kv::{
    KvResidencyEvent, KvResidencyEventKind, PagedKvConfig, PagedKvObservation, PagedKvTable,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceError {
    NonMonotoneSequence,
    PhysicalPageOutOfRange,
    GenerationMismatch,
}

fn validate_trace(
    observation: &PagedKvObservation,
    events: &[KvResidencyEvent],
) -> Result<(), TraceError> {
    let mut previous_sequence = None;
    for event in events {
        if previous_sequence.is_some_and(|previous| event.sequence() <= previous) {
            return Err(TraceError::NonMonotoneSequence);
        }
        if event.physical_page() >= observation.config().physical_pages {
            return Err(TraceError::PhysicalPageOutOfRange);
        }
        if event.generation() != observation.telemetry().generation {
            return Err(TraceError::GenerationMismatch);
        }
        previous_sequence = Some(event.sequence());
    }
    Ok(())
}

#[test]
fn observation_scoped_trace_accepts_ordered_policy_neutral_events() {
    let config = PagedKvConfig {
        page_size: 4,
        physical_pages: 3,
    };
    let mut table = PagedKvTable::new(config).unwrap();
    table.append(6).unwrap();
    let observation = PagedKvObservation::capture(&table).unwrap();
    let generation = observation.telemetry().generation;
    let events = [
        KvResidencyEvent::new(10, 0, generation, KvResidencyEventKind::Retain),
        KvResidencyEvent::new(11, 1, generation, KvResidencyEventKind::Demote),
        KvResidencyEvent::new(12, 1, generation, KvResidencyEventKind::Promote),
    ];

    assert_eq!(validate_trace(&observation, &events), Ok(()));
}

#[test]
fn trace_rejects_non_monotone_sequences() {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 2,
        physical_pages: 2,
    })
    .unwrap();
    table.append(1).unwrap();
    let observation = PagedKvObservation::capture(&table).unwrap();
    let generation = observation.telemetry().generation;
    let events = [
        KvResidencyEvent::new(7, 0, generation, KvResidencyEventKind::Retain),
        KvResidencyEvent::new(7, 0, generation, KvResidencyEventKind::Evict),
    ];

    assert_eq!(
        validate_trace(&observation, &events),
        Err(TraceError::NonMonotoneSequence)
    );
}

#[test]
fn trace_rejects_pages_outside_declared_capacity() {
    let table = PagedKvTable::new(PagedKvConfig {
        page_size: 2,
        physical_pages: 2,
    })
    .unwrap();
    let observation = PagedKvObservation::capture(&table).unwrap();
    let event = KvResidencyEvent::new(
        1,
        observation.config().physical_pages,
        observation.telemetry().generation,
        KvResidencyEventKind::Evict,
    );

    assert_eq!(
        validate_trace(&observation, &[event]),
        Err(TraceError::PhysicalPageOutOfRange)
    );
}

#[test]
fn trace_rejects_events_from_a_stale_generation() {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 2,
        physical_pages: 2,
    })
    .unwrap();
    table.append(1).unwrap();
    let stale_generation = table.generation();
    table.reset().unwrap();
    table.append(1).unwrap();
    let observation = PagedKvObservation::capture(&table).unwrap();
    let event = KvResidencyEvent::new(1, 0, stale_generation, KvResidencyEventKind::Promote);

    assert_eq!(
        validate_trace(&observation, &[event]),
        Err(TraceError::GenerationMismatch)
    );
}
