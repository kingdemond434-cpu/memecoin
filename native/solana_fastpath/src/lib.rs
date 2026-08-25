//! Hot-path primitives for the Solana launch stream.
//!
//! Everything here is pure Rust and testable without Python. The
//! `python` feature adds the pyo3 layer on top; nothing in these modules
//! depends on it.

pub mod curve;
pub mod filter;
pub mod helpers;
pub mod instruction;
pub mod telemetry;

#[cfg(feature = "python")]
mod bindings;
