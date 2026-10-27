"""Numeric archived market facts only: no quotes, RPC, VM or state mutation."""
import hashlib

from src.research import fork_simulation as f
from src.chains import meteora_dbc_route as dbc, launchlab_route as launch
from src.chains.pump_curve import parse_bonding_curve


def market_facts(frame):
    venue=frame.get("target",{}).get("venue")
    programs={"pump_curve":f.PUMP_PROGRAM,"meteora_dbc":dbc.DBC_PROGRAM,
              "raydium_launchlab":launch.LAUNCHLAB_PROGRAM}
    if venue not in programs:
        return {"status":"UNSUPPORTED", "protocol_cause_verified":False}
    result={"status":"DECODED_RESEARCH_ONLY", "protocol_cause_verified":False,
            "program_source_version_matched":False, "quote_verified":False}
    try:
        accounts=frame["accounts"]
        address=f._curve_address(frame["target"]["token"]) if venue=="pump_curve" else frame["target"]["pool"]
        value=accounts[address]
        if not value or value["owner"]!=programs[venue]:
            raise ValueError("market account owner mismatch")
        raw=f.account_data(value)
        if venue=="pump_curve":
            state=parse_bonding_curve(raw)
            if state is None:raise ValueError("unreviewed curve")
            result["values"]={name:getattr(state,name) for name in
                ("virtual_token_reserves","virtual_sol_reserves","real_token_reserves",
                 "real_sol_reserves","token_total_supply","complete")}
            result["values"]["real_quote_reserve_zero"]=state.real_sol_reserves==0
        elif venue=="meteora_dbc":
            state=dbc.parse_pool(raw,address)
            config=dbc.validate_frame(state,accounts,f.account_data)
            price,floor=state["sqrt_price"],config["sqrt_start_price"]
            result["values"]={name:state[name] for name in
                ("base_reserve","quote_reserve","sqrt_price","is_migrated","activation_point")}
            result["values"].update(sqrt_start_price=floor,
                migration_sqrt_price=config["migration_sqrt_price"],
                at_or_below_nonzero_start_price=price<=floor if floor>0 else None)
        else:
            state=launch.parse_pool(raw,address)
            launch.validate_frame(state,accounts,f.account_data)
            result["values"]={name:state[name] for name in
                ("status","supply","total_base_sell","virtual_base","virtual_quote",
                 "real_base","real_quote","base_decimals","quote_decimals")}
        if venue!="pump_curve":
            result["values"].update({side+"_vault_token_units":int.from_bytes(
                f.account_data(accounts[state[side+"_vault"]])[64:72],"little")
                for side in ("base","quote")})
        # Hash actual archived ELF bytes; never infer a source revision from an IDL.
        try:
            images=f._program_images(frame)
            image=next((raw for key,loader,raw in images if key==programs[venue]),None)
            result["program_image_sha256"]=hashlib.sha256(image).hexdigest() if image else None
        except Exception as exc:
            result["program_image_sha256"]=None
            result["program_image_error_type"]=type(exc).__name__
        return result
    except Exception as exc:
        return {"status":"UNAVAILABLE", "error_type":type(exc).__name__,
                "protocol_cause_verified":False, "quote_verified":False}
