import copy
import hashlib
import json
import struct

import pytest
from solders.pubkey import Pubkey

from src.research import fork_simulation as f
from src.research.paper_exit_facts import market_facts
from src.chains.pump_curve import BONDING_CURVE_DISCRIMINATOR
from tests.test_meteora_dbc_route import fixture as dbc_fixture
from tests.test_launchlab_route import fixture as launch_fixture
from tests.test_fork_simulation import wire


@pytest.mark.parametrize("token2022",[False,True])
def test_dbc_exact_price_floor_and_vault_values_are_facts_not_quotes(token2022):
    state,accounts=dbc_fixture(token2022)
    frame={"target":{"venue":"meteora_dbc","pool":state["pool"]},"accounts":accounts}
    before=copy.deepcopy(frame)
    result=market_facts(frame)
    assert result["status"]=="DECODED_RESEARCH_ONLY"
    assert result["values"]["sqrt_price"]==result["values"]["sqrt_start_price"]==2**64
    assert result["values"]["at_or_below_nonzero_start_price"] is True
    assert result["values"]["base_vault_token_units"]==10**9
    assert result["values"]["quote_vault_token_units"]==10**9
    assert result["program_image_sha256"] is None
    assert not result["quote_verified"] and not result["protocol_cause_verified"]
    assert not result["program_source_version_matched"]
    assert frame==before
    assert state["base_mint"] not in json.dumps(result)


def test_launchlab_retains_numeric_fields_without_identity_export():
    state,accounts=launch_fixture()
    result=market_facts({"target":{"venue":"raydium_launchlab","pool":state["pool"]},"accounts":accounts})
    assert result["status"]=="DECODED_RESEARCH_ONLY"
    assert result["values"]["status"]==0
    assert result["values"]["supply"]==100
    assert result["values"]["real_quote"]==105
    assert state["base_mint"] not in json.dumps(result)


def test_pump_real_quote_zero_does_not_become_cause_or_honeypot_verdict():
    mint=str(Pubkey.new_unique());curve=f._curve_address(mint)
    raw=BONDING_CURVE_DISCRIMINATOR+struct.pack("<5Q?",1000,500,800,0,2000,False)
    frame={"target":{"venue":"pump_curve","token":mint},"accounts":{curve:wire(raw,f.PUMP_PROGRAM)}}
    result=market_facts(frame)
    assert result["status"]=="DECODED_RESEARCH_ONLY"
    assert result["values"]["real_sol_reserves"]==0
    assert result["values"]["real_quote_reserve_zero"] is True
    assert not result["protocol_cause_verified"]


@pytest.mark.parametrize("mutation",["wrong_owner","truncated_pool","missing_config"])
def test_malformed_or_unreviewed_state_is_explicit_and_private(mutation):
    state,accounts=dbc_fixture()
    if mutation=="wrong_owner":accounts[state["pool"]]["owner"]="PRIVATE_SENTINEL"
    if mutation=="truncated_pool":accounts[state["pool"]]=wire(b"PRIVATE_SENTINEL",accounts[state["pool"]]["owner"])
    if mutation=="missing_config":accounts.pop(state["config"])
    result=market_facts({"target":{"venue":"meteora_dbc","pool":state["pool"]},"accounts":accounts})
    assert result["status"]=="UNAVAILABLE"
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
    assert not result["protocol_cause_verified"]


def test_unsupported_venue_not_invented():
    result=market_facts({"target":{"venue":"PRIVATE_SENTINEL"}})
    assert result=={"status":"UNSUPPORTED","protocol_cause_verified":False}


@pytest.mark.parametrize("image",[b"\x7fELFSYNTHETIC_NOT_EXECUTED",b"INVALID_IMAGE_PRIVATE_SENTINEL"])
def test_archived_image_hash_never_becomes_source_or_quote_proof(image):
    mint=str(Pubkey.new_unique());curve=f._curve_address(mint)
    raw=BONDING_CURVE_DISCRIMINATOR+struct.pack("<5Q?",1000,500,800,100,2000,False)
    frame={"target":{"venue":"pump_curve","token":mint},"accounts":{
        curve:wire(raw,f.PUMP_PROGRAM),f.PUMP_PROGRAM:wire(image,f.IMMUTABLE,executable=True)}}
    result=market_facts(frame)
    if image.startswith(b"\x7fELF"):
        assert result["program_image_sha256"]==hashlib.sha256(image).hexdigest()
    else:
        assert result["program_image_sha256"] is None
        assert result["program_image_error_type"]=="ForkUnavailable"
    assert not result["program_source_version_matched"] and not result["quote_verified"]
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
