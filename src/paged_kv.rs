use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvConfig {
    pub page_size: usize,
    pub physical_pages: usize,
}

impl PagedKvConfig {
    pub fn capacity_tokens(self) -> Result<usize, PagedKvError> {
        if self.page_size == 0 || self.physical_pages == 0 {
            return Err(PagedKvError::ZeroDimension);
        }
        self.page_size
            .checked_mul(self.physical_pages)
            .ok_or(PagedKvError::CapacityOverflow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvAddress {
    pub physical_page: usize,
    pub offset_in_page: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvTelemetry {
    pub live_tokens: usize,
    pub capacity_tokens: usize,
    pub mapped_pages: usize,
    pub free_pages: usize,
    pub internal_fragmentation_tokens: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedKvError {
    ZeroDimension,
    CapacityOverflow,
    CapacityExceeded { requested: usize, capacity: usize },
    GenerationOverflow,
}

impl fmt::Display for PagedKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "paged KV dimensions must be non-zero"),
            Self::CapacityOverflow => write!(f, "paged KV capacity overflows usize"),
            Self::CapacityExceeded { requested, capacity } => write!(
                f,
                "paged KV append requires {requested} tokens, capacity is {capacity}"
            ),
            Self::GenerationOverflow => write!(f, "paged KV generation counter overflowed"),
        }
    }
}

impl std::error::Error for PagedKvError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageEntry {
    physical_page: usize,
    generation: u64,
}

/// Vendor-independent logical-to-physical page table for resident KV storage.
///
/// This type owns metadata only. It does not allocate device buffers or perform
/// copies/submissions. Physical pages are assigned deterministically from the
/// lowest available page index and remain stable until [`Self::reset`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvTable {
    config: PagedKvConfig,
    live_tokens: usize,
    generation: u64,
    logical_pages: Vec<PageEntry>,
    free_pages: Vec<usize>,
}

impl PagedKvTable {
    pub fn new(config: PagedKvConfig) -> Result<Self, PagedKvError> {
        config.capacity_tokens()?;
        let free_pages = (0..config.physical_pages).rev().collect();
        Ok(Self {
            config,
            live_tokens: 0,
            generation: 0,
            logical_pages: Vec::new(),
            free_pages,
        })
    }

    pub fn config(&self) -> PagedKvConfig {
        self.config
    }

    pub fn len(&self) -> usize {
        self.live_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.live_tokens == 0
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn append(&mut self, tokens: usize) -> Result<(), PagedKvError> {
        if tokens == 0 {
            return Ok(());
        }
        let new_len = self
            .live_tokens
            .checked_add(tokens)
            .ok_or(PagedKvError::CapacityOverflow)?;
        let capacity = self.config.capacity_tokens()?;
        if new_len > capacity {
            return Err(PagedKvError::CapacityExceeded {
                requested: new_len,
                capacity,
            });
        }
        let required_pages = new_len.div_ceil(self.config.page_size);
        while self.logical_pages.len() < required_pages {
            let physical_page = self
                .free_pages
                .pop()
                .expect("validated capacity guarantees a free page");
            self.logical_pages.push(PageEntry {
                physical_page,
                generation: self.generation,
            });
        }
        self.live_tokens = new_len;
        Ok(())
    }

    pub fn address(&self, logical_token: usize) -> Option<PagedKvAddress> {
        if logical_token >= self.live_tokens {
            return None;
        }
        let logical_page = logical_token / self.config.page_size;
        let offset_in_page = logical_token % self.config.page_size;
        let entry = self.logical_pages.get(logical_page)?;
        if entry.generation != self.generation {
            return None;
        }
        Some(PagedKvAddress {
            physical_page: entry.physical_page,
            offset_in_page,
            generation: entry.generation,
        })
    }

    pub fn telemetry(&self) -> Result<PagedKvTelemetry, PagedKvError> {
        let capacity_tokens = self.config.capacity_tokens()?;
        let mapped_pages = self.logical_pages.len();
        let allocated_tokens = mapped_pages
            .checked_mul(self.config.page_size)
            .ok_or(PagedKvError::CapacityOverflow)?;
        Ok(PagedKvTelemetry {
            live_tokens: self.live_tokens,
            capacity_tokens,
            mapped_pages,
            free_pages: self.free_pages.len(),
            internal_fragmentation_tokens: allocated_tokens - self.live_tokens,
            generation: self.generation,
        })
    }

    pub fn reset(&mut self) -> Result<(), PagedKvError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(PagedKvError::GenerationOverflow)?;
        self.live_tokens = 0;
        self.logical_pages.clear();
        self.free_pages.clear();
        self.free_pages.extend((0..self.config.physical_pages).rev());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_is_stable_across_page_boundaries() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 3,
        })
        .unwrap();
        table.append(9).unwrap();
        assert_eq!(table.address(0).unwrap().physical_page, 0);
        assert_eq!(table.address(3).unwrap().offset_in_page, 3);
        assert_eq!(table.address(4).unwrap().physical_page, 1);
        assert_eq!(table.address(8).unwrap().physical_page, 2);
        assert_eq!(table.address(9), None);
    }

    #[test]
    fn telemetry_reports_exact_internal_fragmentation() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 8,
            physical_pages: 4,
        })
        .unwrap();
        table.append(9).unwrap();
        assert_eq!(
            table.telemetry().unwrap(),
            PagedKvTelemetry {
                live_tokens: 9,
                capacity_tokens: 32,
                mapped_pages: 2,
                free_pages: 2,
                internal_fragmentation_tokens: 7,
                generation: 0,
            }
        );
    }

    #[test]
    fn reset_invalidates_old_generation_before_page_reuse() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 2,
        })
        .unwrap();
        table.append(1).unwrap();
        let old = table.address(0).unwrap();
        table.reset().unwrap();
        assert_eq!(table.address(0), None);
        table.append(1).unwrap();
        let new = table.address(0).unwrap();
        assert_eq!(old.physical_page, new.physical_page);
        assert_ne!(old.generation, new.generation);
    }

    #[test]
    fn capacity_overflow_is_explicit_and_non_mutating() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 2,
        })
        .unwrap();
        table.append(7).unwrap();
        let before = table.clone();
        assert_eq!(
            table.append(2),
            Err(PagedKvError::CapacityExceeded {
                requested: 9,
                capacity: 8,
            })
        );
        assert_eq!(table, before);
    }
}
