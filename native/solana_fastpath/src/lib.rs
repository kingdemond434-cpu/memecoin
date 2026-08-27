//! Hot-path primitives for the Solana launch stream.
//!
//! Everything here is pure Rust and testable without Python. The
//! `python` feature adds the pyo3 layer on top; nothing in these modules
//! depends on it.

pub mod curve;
pub mod decide;
pub mod filter;
pub mod helpers;
pub mod generated_flags;
pub mod instruction;
pub mod policy;
pub mod pumpswap;
pub mod safety;
pub mod state;
pub mod telemetry;

#[cfg(feature = "python")]
mod bindings;
