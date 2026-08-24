use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

#[pyfunction]
fn b58encode(raw: &[u8]) -> String {
    bs58::encode(raw).into_string()
}

#[pyfunction]
fn b58decode(value: &str) -> PyResult<Vec<u8>> {
    bs58::decode(value)
        .into_vec()
        .map_err(|err| PyValueError::new_err(format!("invalid base58: {err}")))
}

#[pyfunction]
fn anchor_discriminator(name: &str) -> Vec<u8> {
    let digest = Sha256::digest(format!("global:{name}").as_bytes());
    digest[..8].to_vec()
}

#[pyfunction]
fn looks_like_pool_creation(logs: Vec<String>) -> bool {
    logs.iter().any(|line| {
        let Some((_, instruction)) = line.rsplit_once("Instruction:") else { return false; };
        let normalized = instruction.trim().to_ascii_lowercase().replace('_', "").replace(' ', "");
        matches!(normalized.as_str(),
            "initialize" | "initialize2" | "initializepool" | "initializepoolv2" | "createpool"
            | "initializelbpair" | "initializecustomizablepermissionlesslbpair"
            | "initializepermissionlesspool" | "initializepermissionlesspoolwithfeetier"
            | "initializepermissionlessconstantproductpoolwithconfig"
            | "initializepermissionlessconstantproductpoolwithconfig2"
            | "initializecustomizablepermissionlessconstantproductpool")
    })
}

#[pymodule]
fn solana_fastpath(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(b58encode, module)?)?;
    module.add_function(wrap_pyfunction!(b58decode, module)?)?;
    module.add_function(wrap_pyfunction!(anchor_discriminator, module)?)?;
    module.add_function(wrap_pyfunction!(looks_like_pool_creation, module)?)?;
    module.add("IMPLEMENTATION", "rust-pyo3-abi3")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base58_round_trip_and_leading_zeroes() {
        let raw = [0_u8, 0, 1, 2, 3, 254, 255];
        let encoded = b58encode(&raw);
        assert_eq!(b58decode(&encoded).unwrap(), raw);
        assert!(encoded.starts_with("11"));
    }

    #[test]
    fn discriminator_is_stable() {
        assert_eq!(anchor_discriminator("buy"), vec![102, 6, 61, 18, 1, 218, 235, 234]);
    }

    #[test]
    fn pool_filter_accepts_initializer_and_rejects_position_noise() {
        assert!(looks_like_pool_creation(vec!["Program log: Instruction: InitializePoolV2".into()]));
        assert!(!looks_like_pool_creation(vec!["Program log: Instruction: CreatePosition".into()]));
    }
}
