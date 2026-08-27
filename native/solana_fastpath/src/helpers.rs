//! Small pure helpers shared by the hot path and the Python layer.

use sha2::{Digest, Sha256};

pub fn b58encode(raw: &[u8]) -> String {
    bs58::encode(raw).into_string()
}

pub fn b58decode(value: &str) -> Result<Vec<u8>, bs58::decode::Error> {
    bs58::decode(value).into_vec()
}

/// Anchor's instruction discriminator: `sha256("global:<name>")[..8]`.
pub fn anchor_discriminator(name: &str) -> Vec<u8> {
    Sha256::digest(format!("global:{name}").as_bytes())[..8].to_vec()
}

/// Cheap structural check over program logs for a pool-creation instruction.
///
/// Deliberately structural rather than semantic: it may pass what later
/// stages reject, and must never reject what they would have accepted.
pub fn looks_like_pool_creation(logs: &[String]) -> bool {
    logs.iter().any(|line| {
        let Some((_, instruction)) = line.rsplit_once("Instruction:") else {
            return false;
        };
        let normalized = instruction
            .trim()
            .to_ascii_lowercase();
        let normalized: String = normalized
            .chars()
            .filter(|c| *c != '_' && *c != ' ')
            .collect();
        matches!(
            normalized.as_str(),
            "initialize"
                | "initialize2"
                | "initializepool"
                | "initializepoolv2"
                | "createpool"
                | "initializelbpair"
                | "initializecustomizablepermissionlesslbpair"
                | "initializepermissionlesspool"
                | "initializepermissionlesspoolwithfeetier"
                | "initializepermissionlessconstantproductpoolwithconfig"
                | "initializepermissionlessconstantproductpoolwithconfig2"
                | "initializecustomizablepermissionlessconstantproductpool"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base58_round_trips() {
        let raw = [7u8; 32];
        let encoded = b58encode(&raw);
        assert_eq!(b58decode(&encoded).unwrap(), raw);
    }

    #[test]
    fn invalid_base58_is_an_error_not_a_panic() {
        assert!(b58decode("0OIl").is_err());
    }

    #[test]
    fn the_discriminator_matches_the_deployed_program() {
        assert_eq!(anchor_discriminator("buy"), vec![102, 6, 61, 18, 1, 218, 235, 234]);
    }

    #[test]
    fn pool_creation_logs_are_recognised() {
        assert!(looks_like_pool_creation(&[
            "Program log: Instruction: InitializePoolV2".to_string()
        ]));
        assert!(!looks_like_pool_creation(&[
            "Program log: Instruction: CreatePosition".to_string()
        ]));
        assert!(!looks_like_pool_creation(&["no instruction here".to_string()]));
    }
}
