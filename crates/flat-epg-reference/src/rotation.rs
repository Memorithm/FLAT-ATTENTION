use crate::{EpgEmbeddingConfig, EpgError, So4Geometry};

#[inline]
pub(crate) fn frequency(theta: f32, pair: usize, head_dim: usize) -> f32 {
    theta.powf(-2.0 * pair as f32 / head_dim as f32)
}

#[inline]
pub(crate) fn rotate_pair(even: f32, odd: f32, angle: f32) -> (f32, f32) {
    let (sin, cos) = angle.sin_cos();
    (even * cos - odd * sin, even * sin + odd * cos)
}

#[inline]
pub(crate) fn rotate_so4_block(
    block: [f32; 4],
    first_pair: usize,
    head_dim: usize,
    position: usize,
    epg: EpgEmbeddingConfig,
) -> Result<[f32; 4], EpgError> {
    let geometry = epg.so4_geometry().ok_or(EpgError::Contract(
        epg_core::EpgContractError::InvalidSo4Dims(0),
    ))?;
    let omega_01 = frequency(epg.theta(), first_pair, head_dim);
    let omega_23 = match geometry {
        So4Geometry::Isoclinic => omega_01,
        So4Geometry::Biplanar => frequency(epg.theta(), first_pair + 1, head_dim),
    };
    let (r0, r1) = rotate_pair(block[0], block[1], position as f32 * omega_01);
    let (r2, r3) = rotate_pair(block[2], block[3], position as f32 * omega_23);
    Ok([r0, r1, r2, r3])
}

pub(crate) fn epg_dot(
    q: &[f32],
    k: &[f32],
    head_dim: usize,
    query_position: usize,
    key_position: usize,
    epg: EpgEmbeddingConfig,
) -> Result<f32, EpgError> {
    let so2_dims = epg.so2_dims(head_dim)?;
    let mut dot = 0.0f32;

    for pair in 0..so2_dims / 2 {
        let dim = pair * 2;
        let omega = frequency(epg.theta(), pair, head_dim);
        let (qe, qo) = rotate_pair(q[dim], q[dim + 1], query_position as f32 * omega);
        let (ke, ko) = rotate_pair(k[dim], k[dim + 1], key_position as f32 * omega);
        dot += qe * ke + qo * ko;
    }

    let first_so4_pair = so2_dims / 2;
    for block_index in 0..epg.so4_dims() / 4 {
        let dim = so2_dims + block_index * 4;
        let first_pair = first_so4_pair + block_index * 2;
        let qr = rotate_so4_block(
            [q[dim], q[dim + 1], q[dim + 2], q[dim + 3]],
            first_pair,
            head_dim,
            query_position,
            epg,
        )?;
        let kr = rotate_so4_block(
            [k[dim], k[dim + 1], k[dim + 2], k[dim + 3]],
            first_pair,
            head_dim,
            key_position,
            epg,
        )?;
        dot += qr[0] * kr[0] + qr[1] * kr[1] + qr[2] * kr[2] + qr[3] * kr[3];
    }
    Ok(dot)
}
