// Parity harness: emit Rust quotes for a grid of curve states and sizes.
use solana_fastpath::curve::{BondingCurve, LEGACY_FEE_BPS};

fn main() {
    let states = [
        (1_073_000_000_000_000u64, 30_000_000_000u64, 793_100_000_000_000u64, 5_000_000_000u64),
        (500_000_000_000_000, 60_000_000_000, 200_000_000_000_000, 40_000_000_000),
        (100_000_000_000_000, 85_000_000_000, 10_000_000_000_000, 80_000_000_000),
    ];
    for (vt, vs, rt, rs) in states {
        let curve = BondingCurve {
            virtual_token_reserves: vt, virtual_sol_reserves: vs,
            real_token_reserves: rt, real_sol_reserves: rs,
            token_total_supply: 1_000_000_000_000_000, complete: false,
        };
        for lamports in [1_000_000u64, 100_000_000, 1_000_000_000, 10_000_000_000] {
            if let Ok(q) = curve.quote_buy(lamports, LEGACY_FEE_BPS) {
                println!("buy {vt} {vs} {rt} {rs} {lamports} {} {}", q.output_amount, q.fee_amount);
            }
        }
        for tokens in [1_000_000u64, 1_000_000_000, 1_000_000_000_000, 100_000_000_000_000] {
            if let Ok(q) = curve.quote_sell(tokens, LEGACY_FEE_BPS) {
                println!("sell {vt} {vs} {rt} {rs} {tokens} {} {}", q.output_amount, q.fee_amount);
            }
        }
    }
}
