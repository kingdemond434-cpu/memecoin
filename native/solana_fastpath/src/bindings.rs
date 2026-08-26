//! Python bindings. Gated behind the `python` feature so the pricing,
//! decoding and construction logic can be tested with no Python linked.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::curve::{BondingCurve, LEGACY_FEE_BPS};
use crate::instruction;
use crate::pumpswap::{Pool, PoolReserves};

#[pyfunction]
fn b58encode(raw: &[u8]) -> String {
    crate::helpers::b58encode(raw)
}

#[pyfunction]
fn b58decode(value: &str) -> PyResult<Vec<u8>> {
    crate::helpers::b58decode(value)
        .map_err(|err| PyValueError::new_err(format!("invalid base58: {err}")))
}

#[pyfunction]
fn anchor_discriminator(name: &str) -> Vec<u8> {
    crate::helpers::anchor_discriminator(name)
}

#[pyfunction]
fn looks_like_pool_creation(logs: Vec<String>) -> bool {
    crate::helpers::looks_like_pool_creation(&logs)
}

/// Decode a bonding-curve account and price a buy, in one call.
///
/// Deliberately one call rather than decode-then-quote across the FFI
/// boundary: the round trip is a meaningful share of the work at this size,
/// and splitting it would give back most of what moving the arithmetic here
/// was worth.
#[pyfunction]
#[pyo3(signature = (account_data, lamports, fee_bps = LEGACY_FEE_BPS))]
fn quote_buy_from_account(
    account_data: &[u8],
    lamports: u64,
    fee_bps: u64,
) -> PyResult<(u64, u64, u64)> {
    let curve = BondingCurve::decode(account_data)
        .ok_or_else(|| PyValueError::new_err("not a bonding curve account"))?;
    let quote = curve
        .quote_buy(lamports, fee_bps)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok((quote.output_amount, quote.fee_amount, quote.price_impact_bps))
}

#[pyfunction]
#[pyo3(signature = (account_data, tokens, fee_bps = LEGACY_FEE_BPS))]
fn quote_sell_from_account(
    account_data: &[u8],
    tokens: u64,
    fee_bps: u64,
) -> PyResult<(u64, u64, u64)> {
    let curve = BondingCurve::decode(account_data)
        .ok_or_else(|| PyValueError::new_err("not a bonding curve account"))?;
    let quote = curve
        .quote_sell(tokens, fee_bps)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok((quote.output_amount, quote.fee_amount, quote.price_impact_bps))
}

/// Executable exit size at each impact bound, from one decode.
#[pyfunction]
#[pyo3(signature = (account_data, bounds_bps, fee_bps = LEGACY_FEE_BPS))]
fn exit_frontier(
    account_data: &[u8],
    bounds_bps: Vec<u64>,
    fee_bps: u64,
) -> PyResult<Vec<(u64, u64)>> {
    let curve = BondingCurve::decode(account_data)
        .ok_or_else(|| PyValueError::new_err("not a bonding curve account"))?;
    Ok(curve.exit_frontier(&bounds_bps, fee_bps))
}

/// Decoded reserves, or None when the account is not a bonding curve.
#[pyfunction]
fn decode_bonding_curve(account_data: &[u8]) -> Option<(u64, u64, u64, u64, u64, bool)> {
    BondingCurve::decode(account_data).map(|curve| {
        (
            curve.virtual_token_reserves,
            curve.virtual_sol_reserves,
            curve.real_token_reserves,
            curve.real_sol_reserves,
            curve.token_total_supply,
            curve.complete,
        )
    })
}

/// Serialised `buy_v2` instruction data: discriminator plus both u64 args.
#[pyfunction]
fn buy_v2_data(amount: u64, max_sol_cost: u64) -> Vec<u8> {
    let mut data = instruction::anchor_instruction_discriminator("buy_v2").to_vec();
    data.extend_from_slice(&amount.to_le_bytes());
    data.extend_from_slice(&max_sol_cost.to_le_bytes());
    data
}

/// Serialised `sell_v2` instruction data.
#[pyfunction]
fn sell_v2_data(amount: u64, min_sol_output: u64) -> Vec<u8> {
    let mut data = instruction::anchor_instruction_discriminator("sell_v2").to_vec();
    data.extend_from_slice(&amount.to_le_bytes());
    data.extend_from_slice(&min_sol_output.to_le_bytes());
    data
}

/// (is_signer, is_writable) per account, in the documented order.
///
/// Exposed so the Python side can assemble the transaction against the same
/// flags the Rust builder uses, rather than maintaining a second copy of the
/// table that can drift from it.
#[pyfunction]
fn account_flags(instruction_name: &str) -> PyResult<Vec<(bool, bool)>> {
    let zero = [0u8; instruction::PUBKEY_LEN];
    match instruction_name {
        "buy_v2" => {
            let accounts = instruction::BuyAccounts {
                global: zero, base_mint: zero, quote_mint: zero, base_token_program: zero,
                quote_token_program: zero, associated_token_program: zero, fee_recipient: zero,
                associated_quote_fee_recipient: zero, buyback_fee_recipient: zero,
                associated_quote_buyback_fee_recipient: zero, bonding_curve: zero,
                associated_base_bonding_curve: zero, associated_quote_bonding_curve: zero,
                user: zero, associated_base_user: zero, associated_quote_user: zero,
                creator_vault: zero, associated_creator_vault: zero, sharing_config: zero,
                global_volume_accumulator: zero, user_volume_accumulator: zero,
                associated_user_volume_accumulator: zero, fee_config: zero, fee_program: zero,
                system_program: zero, event_authority: zero, program: zero,
            };
            Ok(instruction::build_buy_v2(&accounts, 0, 0)
                .accounts
                .iter()
                .map(|meta| (meta.is_signer, meta.is_writable))
                .collect())
        }
        "sell_v2" => {
            let accounts = instruction::SellAccounts {
                global: zero, base_mint: zero, quote_mint: zero, base_token_program: zero,
                quote_token_program: zero, associated_token_program: zero, fee_recipient: zero,
                associated_quote_fee_recipient: zero, buyback_fee_recipient: zero,
                associated_quote_buyback_fee_recipient: zero, bonding_curve: zero,
                associated_base_bonding_curve: zero, associated_quote_bonding_curve: zero,
                user: zero, associated_base_user: zero, associated_quote_user: zero,
                creator_vault: zero, associated_creator_vault: zero, sharing_config: zero,
                user_volume_accumulator: zero, associated_user_volume_accumulator: zero,
                fee_config: zero, fee_program: zero, system_program: zero,
                event_authority: zero, program: zero,
            };
            Ok(instruction::build_sell_v2(&accounts, 0, 0)
                .accounts
                .iter()
                .map(|meta| (meta.is_signer, meta.is_writable))
                .collect())
        }
        other => Err(PyValueError::new_err(format!("unknown instruction: {other}"))),
    }
}

/// Decoded PumpSwap pool fields, or None when the account is not a pool.
#[pyfunction]
fn decode_pumpswap_pool(account_data: &[u8]) -> Option<(u8, u16, Vec<u8>, Vec<u8>, u64, bool, bool, i128)> {
    Pool::decode(account_data).map(|pool| {
        (
            pool.pool_bump,
            pool.index,
            pool.base_mint.to_vec(),
            pool.quote_mint.to_vec(),
            pool.lp_supply,
            pool.is_mayhem_mode,
            pool.is_cashback_coin,
            pool.virtual_quote_reserves,
        )
    })
}

/// Quote a PumpSwap buy from observed vault reserves.
///
/// Reserves are passed in rather than read from the pool account: the vault
/// balances are not in that account, and a quote computed from whichever half
/// happened to be available would be silently wrong.
#[pyfunction]
#[pyo3(signature = (base_reserves, quote_reserves, quote_in, fee_bps = LEGACY_FEE_BPS))]
fn pumpswap_quote_buy(
    base_reserves: u64,
    quote_reserves: u64,
    quote_in: u64,
    fee_bps: u64,
) -> PyResult<(u64, u64, u64)> {
    let reserves = PoolReserves { base: base_reserves, quote: quote_reserves };
    let quote = reserves
        .quote_buy(quote_in, fee_bps)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok((quote.output_amount, quote.fee_amount, quote.price_impact_bps))
}

#[pyfunction]
#[pyo3(signature = (base_reserves, quote_reserves, base_in, fee_bps = LEGACY_FEE_BPS))]
fn pumpswap_quote_sell(
    base_reserves: u64,
    quote_reserves: u64,
    base_in: u64,
    fee_bps: u64,
) -> PyResult<(u64, u64, u64)> {
    let reserves = PoolReserves { base: base_reserves, quote: quote_reserves };
    let quote = reserves
        .quote_sell(base_in, fee_bps)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok((quote.output_amount, quote.fee_amount, quote.price_impact_bps))
}

#[pyfunction]
#[pyo3(signature = (base_reserves, quote_reserves, max_impact_bps, fee_bps = LEGACY_FEE_BPS))]
fn pumpswap_sell_capacity(
    base_reserves: u64,
    quote_reserves: u64,
    max_impact_bps: u64,
    fee_bps: u64,
) -> u64 {
    PoolReserves { base: base_reserves, quote: quote_reserves }
        .sell_capacity(max_impact_bps, fee_bps)
}

/// PumpSwap `buy` / `sell` instruction DATA. Accounts are deliberately absent.
#[pyfunction]
fn pumpswap_buy_data(base_out: u64, max_quote_in: u64) -> Vec<u8> {
    crate::pumpswap::buy_data(base_out, max_quote_in)
}

#[pyfunction]
fn pumpswap_sell_data(base_in: u64, min_quote_out: u64) -> Vec<u8> {
    crate::pumpswap::sell_data(base_in, min_quote_out)
}

/// Why PumpSwap transaction construction stops at instruction data.
#[pyfunction]
fn pumpswap_account_list_status() -> &'static str {
    crate::pumpswap::ACCOUNT_LIST_STATUS
}

/// Score every action on one forward view, entirely in Rust.
///
/// Returns `(action, q, age_band, allowed, blocked, refused, commit_fraction)`
/// plus the per-action scores, so a caller gets the whole decision from one
/// call rather than reconstructing it from parts. The probabilities are
/// handed in because inference belongs to the trained artifact the promotion
/// gate validated -- reimplementing it here would buy microseconds and
/// introduce a second model that can disagree with the one under test.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
// Explicit signature because pyo3 cannot infer argument order once an
// Option precedes a required parameter, and every one of these is
// deliberately Option where "unmeasured" is a state the policy must see.
#[pyo3(signature = (
    age_seconds, virtual_sol, virtual_token, levels, p_rug_30s, p_rug_5m,
    expected_feasible_multiple, held_fraction, current_multiple, exit_cost,
    entry_cost, exit_capacity_ratio, escape_probability,
    alternative_growth_per_second, expected_remaining_seconds, add_fraction,
    add_capacity_fraction, probe_fraction, min_edge, max_add_fraction, live,
    max_position_fraction,
    max_single_commit_fraction, min_commit_fraction, min_exit_capacity,
    live_unlocked
))]
fn t0_decide(
    age_seconds: f64,
    virtual_sol: u64,
    virtual_token: u64,
    levels: Vec<f64>,
    p_rug_30s: f64,
    p_rug_5m: f64,
    expected_feasible_multiple: f64,
    held_fraction: f64,
    current_multiple: f64,
    exit_cost: f64,
    entry_cost: f64,
    exit_capacity_ratio: Option<f64>,
    escape_probability: Option<f64>,
    alternative_growth_per_second: Option<f64>,
    expected_remaining_seconds: Option<f64>,
    add_fraction: Option<f64>,
    add_capacity_fraction: Option<f64>,
    probe_fraction: Option<f64>,
    min_edge: f64,
    max_add_fraction: f64,
    live: bool,
    max_position_fraction: f64,
    max_single_commit_fraction: f64,
    min_commit_fraction: f64,
    min_exit_capacity: f64,
    live_unlocked: bool,
) -> PyResult<(String, f64, String, bool, Option<String>, Option<String>, f64, Vec<(String, f64, bool)>)>
{
    if levels.len() != crate::policy::SURVIVAL_MULTIPLES.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {} survival levels, got {}",
            crate::policy::SURVIVAL_MULTIPLES.len(),
            levels.len()
        )));
    }
    let mut fixed = [0.0f64; 8];
    fixed.copy_from_slice(&levels);

    let mut state = crate::state::TokenState::new(0.0);
    state.update_reserves(
        crate::state::Reserves {
            virtual_sol,
            virtual_token,
            real_sol: 0,
            real_token: 0,
            measured: false,
        },
        0.0,
    );

    let inputs = crate::decide::Inputs {
        position: crate::policy::Position {
            held_fraction,
            current_multiple,
            exit_cost,
            entry_cost,
            exit_capacity_ratio,
            escape_probability,
            alternative_growth_per_second,
            expected_remaining_seconds,
            add_fraction,
            add_capacity_fraction,
            probe_fraction,
        },
        survival: crate::policy::Survival {
            levels: fixed,
            p_rug_30s,
            p_rug_5m,
            expected_feasible_multiple,
        },
        min_edge,
        max_add_fraction,
        live,
    };
    let limits = crate::safety::Limits {
        max_position_fraction,
        max_single_commit_fraction,
        min_commit_fraction,
        min_exit_capacity,
        live_unlocked,
    };
    let decision = crate::decide::decide(&state, age_seconds, &inputs, &limits);
    Ok((
        decision.action.as_str().to_string(),
        decision.q,
        decision.age_band.to_string(),
        decision.allowed,
        decision.blocked.map(str::to_string),
        decision.refused.map(str::to_string),
        decision.commit_fraction,
        decision
            .scores
            .iter()
            .map(|score| (score.action.as_str().to_string(), score.q, score.feasible))
            .collect(),
    ))
}

/// Disjoint executable bins from the cumulative survival curve.
#[pyfunction]
fn survival_bins(
    levels: Vec<f64>,
    p_rug_30s: f64,
    p_rug_5m: f64,
    expected_feasible_multiple: f64,
) -> PyResult<Vec<(f64, f64)>> {
    if levels.len() != crate::policy::SURVIVAL_MULTIPLES.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "wrong number of survival levels",
        ));
    }
    let mut fixed = [0.0f64; 8];
    fixed.copy_from_slice(&levels);
    Ok(crate::policy::probability_bins(&crate::policy::Survival {
        levels: fixed,
        p_rug_30s,
        p_rug_5m,
        expected_feasible_multiple,
    })
    .into_iter()
    .map(|bin| (bin.probability, bin.gross))
    .collect())
}

/// Which age brain owns a decision at this launch age.
#[pyfunction]
fn t0_age_band(age_seconds: f64) -> &'static str {
    crate::policy::age_band(age_seconds)
}

#[pymodule]
fn solana_fastpath(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(quote_buy_from_account, module)?)?;
    module.add_function(wrap_pyfunction!(quote_sell_from_account, module)?)?;
    module.add_function(wrap_pyfunction!(exit_frontier, module)?)?;
    module.add_function(wrap_pyfunction!(decode_bonding_curve, module)?)?;
    module.add_function(wrap_pyfunction!(buy_v2_data, module)?)?;
    module.add_function(wrap_pyfunction!(sell_v2_data, module)?)?;
    module.add_function(wrap_pyfunction!(account_flags, module)?)?;
    module.add_function(wrap_pyfunction!(decode_pumpswap_pool, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_quote_buy, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_quote_sell, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_sell_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_buy_data, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_sell_data, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_account_list_status, module)?)?;
    module.add_function(wrap_pyfunction!(b58encode, module)?)?;
    module.add_function(wrap_pyfunction!(b58decode, module)?)?;
    module.add_function(wrap_pyfunction!(anchor_discriminator, module)?)?;
    module.add_function(wrap_pyfunction!(looks_like_pool_creation, module)?)?;
    module.add_function(wrap_pyfunction!(t0_decide, module)?)?;
    module.add_function(wrap_pyfunction!(survival_bins, module)?)?;
    module.add_function(wrap_pyfunction!(t0_age_band, module)?)?;
    module.add("IMPLEMENTATION", "rust-pyo3-abi3")?;
    Ok(())
}

