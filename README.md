# Memecoin Quant Desk

A production-grade Solana memecoin prediction and execution system built on quantitative research principles.

## Architecture Overview

```
GLOBAL INTELLIGENCE (Hourly)
    ├─ Wallet Discovery & Reranking
    ├─ X/Social Mining (Multi-language)
    ├─ Sniper/Competitor Reverse Engineering
    ├─ Pre-launch Entity Intent Detection
    └─ Failure Mining & Hypothesis Generation
           ↓
POINT-IN-TIME LAUNCH LAKE (Continuous)
    ├─ Yellowstone gRPC Real-time Streams
    ├─ Pump.fun / Raydium / Meteora Monitoring
    ├─ Wallet/Deployer Genealogy Graph
    ├─ Information Lead Graph
    └─ Counterfactual Execution Lab
           ↓
MULTI-HEAD PREDICTORS
    ├─ P(2x), P(5x), P(10x), P(50x)
    ├─ P(Migration)
    ├─ P(Rug 30s/5m)
    ├─ Expected Slippage / Liquidity
    └─ Champion/Challenger Framework
           ↓
E[log W] KELLY ENGINE
    ├─ Robust Expected Log Growth
    ├─ Capacity-Aware Position Sizing
    ├─ Dynamic Rug Hazard Exit
    └─ Right-Tail Preservation
           ↓
EXECUTION ENGINE
    ├─ Jupiter v6 Aggregator
    ├─ Jito Bundles + MEV Protection
    ├─ Priority Fee Optimization
    └─ Real-time Landing Monitor
```

## Key Features

- **Hourly Research Factory**: Discovers wallets, X accounts, narratives, sniper mechanisms
- **Continuous Trading Brain**: Sub-second launch detection, wallet tracking, rug hazard
- **Point-in-Time Correctness**: No future leakage in backtests or live features
- **Multi-Head Probabilities**: Separate models for each outcome, not a single "gem score"
- **Champion/Challenger**: Frozen production models, hourly research mutations
- **Counterfactual Lab**: Every trade simulated across execution policies
- **Adversarial Detection**: Auto-downweights fakeable signals
- **E[log W] Optimization**: Kelly sizing with tail risk, capacity, uncertainty penalties

## Quick Start

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM (Hetzner CX41 or similar)
- Solana wallet private key (base64 encoded)
- Helius API key (for Yellowstone gRPC + enhanced APIs)
- Twitter/X Bearer Token (Elevated/Enterprise for real-time mentions)

### Deployment

```bash
# 1. Clone and configure
git clone <repo>
cd memecoin-bot
cp .env.example .env
# Edit .env with your keys

# 2. Deploy (run as root)
sudo ./deploy.sh

# 3. Start service
systemctl start memecoin-bot

# 4. Monitor
journalctl -u memecoin-bot -f
curl http://localhost:8080/health
```

### Manual Docker Run

```bash
docker compose build
docker compose up -d
docker compose logs -f
```

## Configuration

All chain-specific config in `config/chains.yaml`:
- RPC endpoints with failover
- Factory/router addresses
- Base tokens per chain
- Risk parameters (min liquidity, max tax, etc.)

Global settings in `config/chains.yaml` under `global:`:
- `max_position_size_usd`, `max_daily_loss_usd`
- `min_profit_target_pct`, `stop_loss_pct`
- `dry_run: true` for testing

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check |
| `GET /metrics` | Portfolio PnL, positions |
| `GET /status` | Full system status |

## Monitoring

Key metrics to watch:
- **Daily PnL vs max_daily_loss**
- **Win rate** (target > 35% for positive E[log W])
- **Open positions** (max 10 concurrent)
- **Rug hazard alerts** (CRITICAL/HIGH)
- **Execution success rate** per route
- **Champion model decay scores**

## Research Workflow

1. **Hourly**: Wallet/X/sniper discovery → hypothesis extraction → cheap falsification
2. **Async**: Historical PIT replay → ML training → chronological OOS
3. **Promotion**: Challenger → Shadow (168h) → Canary (72h) → Champion
4. **Decay Monitoring**: Auto-hibernate decaying champions
5. **Source ROI**: Research budget allocates to high-yield sources

## Safety

- **Dry run mode**: Set `dry_run: true` in config
- **Daily loss limit**: Hard stop at `max_daily_loss_usd`
- **Rug hazard**: Auto-exit on CRITICAL/HIGH
- **Time stop**: Auto-exit after 60 min hold
- **Position caps**: Max 5% portfolio per position, 10% total risk

## Directory Structure

```
memecoin-bot/
├── config/chains.yaml          # Chain & global config
├── src/
│   ├── main.py                 # Orchestrator
│   ├── chains/                 # RPC, Yellowstone gRPC
│   ├── detection/              # Token detection, rug detection
│   ├── strategies/             # Intelligence engines, predictors
│   ├── execution/              # Jupiter, Jito, fee optimizer
│   └── research/               # PIT dataset builder
├── data/launch_episodes/       # Compressed JSONL episodes
├── models/                     # Trained predictors
├── logs/
├── Dockerfile
├── docker-compose.yml
├── deploy.sh
├── memecoin-bot.service        # Systemd unit
└── requirements.txt
```

## License

Proprietary - All rights reserved.