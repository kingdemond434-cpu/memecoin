//! Python bindings. Gated behind the `python` feature so the pricing,
//! decoding and construction logic can be tested with no Python linked.

use crate::curve::{BondingCurve, LEGACY_FEE_BPS};
use crate::instruction;
use crate::pumpswap::{Pool, PoolReserves};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn b58encode(raw: &[u8]) -> String {
    crate::helpers::b58encode(raw)
}

#[pyfunction]
fn b58decode(value: &str) -> PyResult<Vec<u8>> {
    crate::helpers::b58decode(value)
        .map_err(|err| PyValueError::new_err(format!("invalid base58: {err}")))
}

/// Evaluate one promoted hazard head natively.
///
/// Exposed so the parity gate can compare THIS code against the Python
/// artifact through the same interface production calls, rather than
/// against a second harness that might diverge from both. Native inference
/// may only be trusted on the hot path once that comparison holds.
#[pyfunction]
#[pyo3(signature = (features, intercept, coef, calibrator_x=None, calibrator_y=None))]
fn hazard_predict(
    features: Vec<f64>,
    intercept: f64,
    coef: Vec<f64>,
    calibrator_x: Option<Vec<f64>>,
    calibrator_y: Option<Vec<f64>>,
) -> PyResult<f64> {
    let head = crate::inference::Head {
        intercept,
        coef,
        calibrator_x: calibrator_x.unwrap_or_default(),
        calibrator_y: calibrator_y.unwrap_or_default(),
    };
    head.predict(&features)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))
}

/// Decode one Pump CPI event payload natively.
///
/// Returns a dict shaped like the Python decoder's output so the parity
/// gate can compare them field for field. A decoder that disagrees with
/// the one the desk was built on is worse than a slow decoder, so the
/// comparison is the point of exposing this at all.
#[pyfunction]
fn decode_pump_event(py: Python<'_>, data: &[u8]) -> PyResult<Option<PyObject>> {
    use crate::event::{decode, DecodeError, PumpEvent};
    let out = PyDict::new(py);
    match decode(data) {
        Ok(PumpEvent::Trade { mint, user, is_buy, sol_amount, token_amount,
                              timestamp, virtual_sol_reserves, virtual_token_reserves }) => {
            out.set_item("type", "token_trade")?;
            out.set_item("token", mint)?;
            out.set_item("wallet", user)?;
            out.set_item("side", if is_buy { "buy" } else { "sell" })?;
            out.set_item("sol_amount", sol_amount)?;
            out.set_item("token_amount", token_amount)?;
            out.set_item("timestamp", timestamp)?;
            out.set_item("virtual_sol_reserves", virtual_sol_reserves)?;
            out.set_item("virtual_token_reserves", virtual_token_reserves)?;
        }
        Ok(PumpEvent::Create { mint, bonding_curve, user, creator, name, symbol, uri, timestamp }) => {
            out.set_item("type", "token_created")?;
            out.set_item("token", mint)?;
            out.set_item("bonding_curve", bonding_curve)?;
            out.set_item("wallet", user)?;
            out.set_item("creator", creator)?;
            out.set_item("name", name)?;
            out.set_item("symbol", symbol)?;
            out.set_item("uri", uri)?;
            out.set_item("timestamp", timestamp)?;
        }
        Ok(PumpEvent::Complete { mint, user, bonding_curve, timestamp }) => {
            out.set_item("type", "token_migrated")?;
            out.set_item("token", mint)?;
            out.set_item("wallet", user)?;
            out.set_item("bonding_curve", bonding_curve)?;
            out.set_item("timestamp", timestamp)?;
        }
        // Not one of ours: the stream carries plenty of other events and
        // skipping them is normal, so this is None rather than an error.
        Err(DecodeError::UnknownDiscriminator) => return Ok(None),
        Err(err) => return Err(PyValueError::new_err(format!("{err:?}"))),
    }
    Ok(Some(out.into()))
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
    Ok((
        quote.output_amount,
        quote.fee_amount,
        quote.price_impact_bps,
    ))
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
    Ok((
        quote.output_amount,
        quote.fee_amount,
        quote.price_impact_bps,
    ))
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
                global: zero,
                base_mint: zero,
                quote_mint: zero,
                base_token_program: zero,
                quote_token_program: zero,
                associated_token_program: zero,
                fee_recipient: zero,
                associated_quote_fee_recipient: zero,
                buyback_fee_recipient: zero,
                associated_quote_buyback_fee_recipient: zero,
                bonding_curve: zero,
                associated_base_bonding_curve: zero,
                associated_quote_bonding_curve: zero,
                user: zero,
                associated_base_user: zero,
                associated_quote_user: zero,
                creator_vault: zero,
                associated_creator_vault: zero,
                sharing_config: zero,
                global_volume_accumulator: zero,
                user_volume_accumulator: zero,
                associated_user_volume_accumulator: zero,
                fee_config: zero,
                fee_program: zero,
                system_program: zero,
                event_authority: zero,
                program: zero,
            };
            Ok(instruction::build_buy_v2(&accounts, 0, 0)
                .accounts
                .iter()
                .map(|meta| (meta.is_signer, meta.is_writable))
                .collect())
        }
        "sell_v2" => {
            let accounts = instruction::SellAccounts {
                global: zero,
                base_mint: zero,
                quote_mint: zero,
                base_token_program: zero,
                quote_token_program: zero,
                associated_token_program: zero,
                fee_recipient: zero,
                associated_quote_fee_recipient: zero,
                buyback_fee_recipient: zero,
                associated_quote_buyback_fee_recipient: zero,
                bonding_curve: zero,
                associated_base_bonding_curve: zero,
                associated_quote_bonding_curve: zero,
                user: zero,
                associated_base_user: zero,
                associated_quote_user: zero,
                creator_vault: zero,
                associated_creator_vault: zero,
                sharing_config: zero,
                user_volume_accumulator: zero,
                associated_user_volume_accumulator: zero,
                fee_config: zero,
                fee_program: zero,
                system_program: zero,
                event_authority: zero,
                program: zero,
            };
            Ok(instruction::build_sell_v2(&accounts, 0, 0)
                .accounts
                .iter()
                .map(|meta| (meta.is_signer, meta.is_writable))
                .collect())
        }
        other => Err(PyValueError::new_err(format!(
            "unknown instruction: {other}"
        ))),
    }
}

/// Decoded PumpSwap pool fields, or None when the account is not a pool.
#[pyfunction]
fn decode_pumpswap_pool(
    account_data: &[u8],
) -> Option<(u8, u16, Vec<u8>, Vec<u8>, u64, bool, bool, i128)> {
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
    let reserves = PoolReserves {
        base: base_reserves,
        quote: quote_reserves,
    };
    let quote = reserves
        .quote_buy(quote_in, fee_bps)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok((
        quote.output_amount,
        quote.fee_amount,
        quote.price_impact_bps,
    ))
}

#[pyfunction]
#[pyo3(signature = (base_reserves, quote_reserves, base_in, fee_bps = LEGACY_FEE_BPS))]
fn pumpswap_quote_sell(
    base_reserves: u64,
    quote_reserves: u64,
    base_in: u64,
    fee_bps: u64,
) -> PyResult<(u64, u64, u64)> {
    let reserves = PoolReserves {
        base: base_reserves,
        quote: quote_reserves,
    };
    let quote = reserves
        .quote_sell(base_in, fee_bps)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok((
        quote.output_amount,
        quote.fee_amount,
        quote.price_impact_bps,
    ))
}

#[pyfunction]
#[pyo3(signature = (base_reserves, quote_reserves, max_impact_bps, fee_bps = LEGACY_FEE_BPS))]
fn pumpswap_sell_capacity(
    base_reserves: u64,
    quote_reserves: u64,
    max_impact_bps: u64,
    fee_bps: u64,
) -> u64 {
    PoolReserves {
        base: base_reserves,
        quote: quote_reserves,
    }
    .sell_capacity(max_impact_bps, fee_bps)
}

/// PumpSwap `buy` / `sell` instruction data.
#[pyfunction]
#[pyo3(signature = (base_out, max_quote_in, track_volume = false))]
fn pumpswap_buy_data(base_out: u64, max_quote_in: u64, track_volume: bool) -> Vec<u8> {
    crate::pumpswap::buy_data(base_out, max_quote_in, track_volume)
}

#[pyfunction]
fn pumpswap_sell_data(base_in: u64, min_quote_out: u64) -> Vec<u8> {
    crate::pumpswap::sell_data(base_in, min_quote_out)
}

type PyInstruction = (Vec<u8>, Vec<(Vec<u8>, bool, bool)>, Vec<u8>);

fn instruction_to_python(value: instruction::Instruction) -> PyInstruction {
    (
        value.program_id.to_vec(),
        value
            .accounts
            .into_iter()
            .map(|meta| (meta.pubkey.to_vec(), meta.is_signer, meta.is_writable))
            .collect(),
        value.data,
    )
}

fn parse_account_keys(raw: Vec<Vec<u8>>) -> PyResult<Vec<instruction::Pubkey>> {
    raw.into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.try_into().map_err(|value: Vec<u8>| {
                PyValueError::new_err(format!(
                    "account {} is {} bytes, expected 32",
                    index + 1,
                    value.len()
                ))
            })
        })
        .collect()
}

/// Build the complete PumpSwap buy instruction in Rust from IDL-ordered keys.
#[pyfunction]
#[pyo3(signature = (accounts, base_out, max_quote_in, track_volume = false))]
fn pumpswap_build_buy(
    accounts: Vec<Vec<u8>>,
    base_out: u64,
    max_quote_in: u64,
    track_volume: bool,
) -> PyResult<PyInstruction> {
    let keys = parse_account_keys(accounts)?;
    crate::pumpswap::build_buy(&keys, base_out, max_quote_in, track_volume)
        .map(instruction_to_python)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "PumpSwap buy requires {} IDL-ordered accounts",
                crate::generated_flags::PUMPSWAP_BUY_ACCOUNT_COUNT
            ))
        })
}

/// Build the complete PumpSwap sell instruction in Rust from IDL-ordered keys.
#[pyfunction]
fn pumpswap_build_sell(
    accounts: Vec<Vec<u8>>,
    base_in: u64,
    min_quote_out: u64,
) -> PyResult<PyInstruction> {
    let keys = parse_account_keys(accounts)?;
    crate::pumpswap::build_sell(&keys, base_in, min_quote_out)
        .map(instruction_to_python)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "PumpSwap sell requires {} IDL-ordered accounts",
                crate::generated_flags::PUMPSWAP_SELL_ACCOUNT_COUNT
            ))
        })
}

/// Provenance for the native PumpSwap account table.
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
    live_unlocked, reentry_bins = None, replacement_bins = None,
    replacement_fraction = None
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
    // (probability, gross) pairs for the two actions whose distribution
    // the position itself does not carry. Absent for most decisions.
    reentry_bins: Option<Vec<(f64, f64)>>,
    replacement_bins: Option<Vec<(f64, f64)>>,
    replacement_fraction: Option<f64>,
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

    // Converted to the policy's own Bin here rather than passed as tuples,
    // so the kernel never sees a shape chosen on the Python side.
    let to_bins = |rows: &Option<Vec<(f64, f64)>>| -> Option<Vec<crate::policy::Bin>> {
        rows.as_ref().map(|rows| {
            rows.iter()
                .map(|(probability, gross)| crate::policy::Bin {
                    probability: *probability,
                    gross: *gross,
                })
                .collect()
        })
    };
    let reentry = to_bins(&reentry_bins);
    let replacement = to_bins(&replacement_bins);

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
        alternatives: crate::policy::Alternatives {
            reentry: reentry.as_deref(),
            replacement: replacement.as_deref(),
            replacement_fraction,
        },
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

// --- addresses, messages and signing -------------------------------------

fn key32(raw: &[u8], what: &str) -> PyResult<instruction::Pubkey> {
    crate::transaction::pubkey_of(raw).ok_or_else(|| {
        PyValueError::new_err(format!("{what} must be exactly 32 bytes, got {}", raw.len()))
    })
}

/// `find_program_address`, returning (address, bump).
#[pyfunction]
fn find_program_address(seeds: Vec<Vec<u8>>, program_id: &[u8])
    -> PyResult<(Vec<u8>, u8)>
{
    let program = key32(program_id, "program_id")?;
    let refs: Vec<&[u8]> = seeds.iter().map(|seed| seed.as_slice()).collect();
    crate::pubkey::find_program_address(&refs, &program)
        .map(|(address, bump)| (address.to_vec(), bump))
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))
}

/// The SPL associated token account for (owner, token_program, mint).
#[pyfunction]
fn associated_token_address(owner: &[u8], token_program: &[u8], mint: &[u8],
                            associated_token_program: &[u8]) -> PyResult<Vec<u8>> {
    crate::pubkey::associated_token_address(
        &key32(owner, "owner")?, &key32(token_program, "token_program")?,
        &key32(mint, "mint")?,
        &key32(associated_token_program, "associated_token_program")?)
        .map(|address| address.to_vec())
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))
}

/// Derive many PDAs in one crossing.
///
/// MEASURED RESULT, kept here because it is worth more than the function is:
/// this is SLOWER than solders for the same eleven addresses. 78us for
/// solders, 84us calling this eleven times, 90us calling it once with all
/// eleven. Batching made it worse, not better.
///
/// The reason is marshalling, not hashing. solders holds `Pubkey` as a native
/// Rust object, so its `find_program_address` copies nothing; this takes
/// Python `bytes`, and pyo3 allocates and copies every nested list on the way
/// in. At eleven 32-byte seeds the copying costs more than the SHA-256 saves.
///
/// So the Python entry path should keep using solders for address derivation.
/// This is retained for callers already holding raw bytes on this side of the
/// boundary -- inside `build_signed_transaction`, where no crossing happens --
/// and as the record of a plausible optimisation that measurement refuted.
#[pyfunction]
fn find_program_addresses(batch: Vec<(Vec<Vec<u8>>, Vec<u8>)>)
    -> PyResult<Vec<(Vec<u8>, u8)>>
{
    let mut out = Vec::with_capacity(batch.len());
    for (seeds, program_id) in batch {
        let program = key32(&program_id, "program_id")?;
        let refs: Vec<&[u8]> = seeds.iter().map(|seed| seed.as_slice()).collect();
        let (address, bump) = crate::pubkey::find_program_address(&refs, &program)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
        out.push((address.to_vec(), bump));
    }
    Ok(out)
}

/// Derive many associated token accounts in one crossing.
#[pyfunction]
fn associated_token_addresses(batch: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
                              associated_token_program: &[u8]) -> PyResult<Vec<Vec<u8>>> {
    let ata_program = key32(associated_token_program, "associated_token_program")?;
    let mut out = Vec::with_capacity(batch.len());
    for (owner, token_program, mint) in batch {
        out.push(crate::pubkey::associated_token_address(
            &key32(&owner, "owner")?, &key32(&token_program, "token_program")?,
            &key32(&mint, "mint")?, &ata_program)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))?
            .to_vec());
    }
    Ok(out)
}

/// One instruction as Python passes it: (program_id, [(pubkey, signer, writable)], data).
type RawInstruction = (Vec<u8>, Vec<(Vec<u8>, bool, bool)>, Vec<u8>);

fn to_instructions(raw: Vec<RawInstruction>) -> PyResult<Vec<instruction::Instruction>> {
    let mut out = Vec::with_capacity(raw.len());
    for (program_id, metas, data) in raw {
        let mut accounts = Vec::with_capacity(metas.len());
        for (pubkey, is_signer, is_writable) in metas {
            accounts.push(instruction::AccountMeta {
                pubkey: key32(&pubkey, "account")?, is_signer, is_writable });
        }
        out.push(instruction::Instruction {
            program_id: key32(&program_id, "program_id")?, accounts, data });
    }
    Ok(out)
}

/// Compile a v0 message and return the exact bytes that get signed.
///
/// Exposed separately from signing so the Python path can compare these bytes
/// against solders' before anything is promoted onto the money path. Equal
/// bytes are the only evidence that matters; a passing unit test is not.
#[pyfunction]
fn compile_v0_message(payer: &[u8], instructions: Vec<RawInstruction>,
                      recent_blockhash: &[u8]) -> PyResult<Vec<u8>> {
    let message = crate::message::compile(
        &key32(payer, "payer")?, &to_instructions(instructions)?,
        &key32(recent_blockhash, "recent_blockhash")?)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok(message.serialize())
}

/// Sign an already-serialised message. The bridge for incremental adoption.
#[pyfunction]
fn sign_message(serialized_message: &[u8], secret_key: &[u8]) -> PyResult<Vec<u8>> {
    let signer = crate::transaction::Signer32::from_bytes(secret_key)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok(signer.sign(serialized_message).to_vec())
}

/// The public key for a secret. Lets the caller check which account will sign
/// without the secret ever being formatted or returned.
#[pyfunction]
fn public_key_of(secret_key: &[u8]) -> PyResult<Vec<u8>> {
    crate::transaction::Signer32::from_bytes(secret_key)
        .map(|signer| signer.public.to_vec())
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))
}

/// Assemble a transaction from a message the ISOLATED SIGNER signed.
///
/// The call the canonical path uses. `build_signed_transaction` needs the
/// secret key and is therefore unusable by a desk whose signer lives in
/// another process; this takes only signatures, so the compile and assemble
/// steps move to Rust while the key stays exactly where it was.
#[pyfunction]
fn assemble_transaction(serialized_message: &[u8], signatures: Vec<Vec<u8>>)
    -> PyResult<String>
{
    crate::transaction::assemble(serialized_message, &signatures)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))
}

/// Compile, sign and encode in one call: the whole tail of the entry path.
///
/// One call rather than three because the FFI round trip is a meaningful
/// share of the work at this size, and because a caller that can hold a
/// half-built transaction is a caller that can submit one.
///
/// Returns (base64 transaction, base58 signature).
#[pyfunction]
#[pyo3(signature = (payer, instructions, recent_blockhash, secret_keys, allow_partial = false))]
fn build_signed_transaction(payer: &[u8], instructions: Vec<RawInstruction>,
                            recent_blockhash: &[u8], secret_keys: Vec<Vec<u8>>,
                            allow_partial: bool) -> PyResult<(String, String)> {
    let mut signers = Vec::with_capacity(secret_keys.len());
    for secret in &secret_keys {
        signers.push(crate::transaction::Signer32::from_bytes(secret)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))?);
    }
    let transaction = crate::transaction::build_signed(
        &key32(payer, "payer")?, &to_instructions(instructions)?,
        &key32(recent_blockhash, "recent_blockhash")?, &signers, allow_partial)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok((transaction.to_base64(), transaction.signature_b58()))
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
    module.add_function(wrap_pyfunction!(pumpswap_build_buy, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_build_sell, module)?)?;
    module.add_function(wrap_pyfunction!(pumpswap_account_list_status, module)?)?;
    module.add_function(wrap_pyfunction!(b58encode, module)?)?;
    module.add_function(wrap_pyfunction!(b58decode, module)?)?;
    module.add_function(wrap_pyfunction!(anchor_discriminator, module)?)?;
    module.add_function(wrap_pyfunction!(hazard_predict, module)?)?;
    module.add_function(wrap_pyfunction!(decode_pump_event, module)?)?;
    module.add_function(wrap_pyfunction!(looks_like_pool_creation, module)?)?;
    module.add_function(wrap_pyfunction!(t0_decide, module)?)?;
    module.add_function(wrap_pyfunction!(survival_bins, module)?)?;
    module.add_function(wrap_pyfunction!(t0_age_band, module)?)?;
    module.add_function(wrap_pyfunction!(find_program_address, module)?)?;
    module.add_function(wrap_pyfunction!(find_program_addresses, module)?)?;
    module.add_function(wrap_pyfunction!(associated_token_addresses, module)?)?;
    module.add_function(wrap_pyfunction!(associated_token_address, module)?)?;
    module.add_function(wrap_pyfunction!(compile_v0_message, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_transaction, module)?)?;
    module.add_function(wrap_pyfunction!(sign_message, module)?)?;
    module.add_function(wrap_pyfunction!(public_key_of, module)?)?;
    module.add_function(wrap_pyfunction!(build_signed_transaction, module)?)?;
    module.add("IMPLEMENTATION", "rust-pyo3-abi3")?;
    Ok(())
}
