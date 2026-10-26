"""Prospective native cost controls, isolated from historical validation/promotion.

This is a measurement baseline, not an alpha strategy. Its program lives in a
separate canonical registry and never enters the framework's hypotheses,
shadow/canary buckets, global ForwardEvidence or live order paths.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from src.research.candidate_replay import CompiledCandidate, ReplayBlocked, _digest
from src.research.gauntlet_feed import Mechanism
from src.research.native_forward import NativeForwardPaper
from src.research.research_validation import ResearchValidationService, _number

CHECKPOINT='native_prospective_validation_v1'
PRODUCER='native_prospective_forward_v1'
CONTROL='native-cost-control-0p001sol-60s-v1'


def control_program(seed, registered_at):
    if not _number(seed,positive=True) or not _number(registered_at,positive=True):
        raise ValueError('explicit finite positive virtual seed and registration clock required')
    return {'schema':'prospective-native-cost-control-v1','candidate_id':CONTROL,
        'registered_at':registered_at,'virtual_seed_usd':seed,
        'account_scope':'independent virtual account per cohort; not a shared portfolio',
        'universe':'new post-registration cohorts actually sampled by the native market observer',
        'entry_rule':'supported fresh native state, exact-size quote and successful unsigned simulation',
        'quote_budget_lamports':1_000_000,'max_decision_latency_ms':60_000,
        'exit_rule':{'op':'timeout','value':60.},
        'emergency_exit':{'op':'stop_loss','value':.3},
        'holding_semantics':'first actually observed executable exit after trigger; never fill at an unseen cutoff',
        'selection_scope':'infrastructure cost control, no claimed predictive edge',
        'cost_scope':'native quote fees/impact plus measured unsigned-simulation network fee',
        'amount_scope':'paper amounts from native quote arithmetic; simulation success is not confirmed balance-delta execution',
        'conversion_scope':'native-denominated pnl valued at observed exit SOL/USD; not a USD cash-ledger return',
        'unmeasured_live_costs':['landing/failure distribution','priority fees and bundle tips','rent lifecycle parity']}


@dataclass(frozen=True)
class PaperProgram:
    hypothesis_id:str
    frozen_hash:str
    frozen_at:float
    exact_candidate:dict


class ProspectiveNativeValidation(ResearchValidationService):
    def __init__(self,framework,ledger,*,virtual_seed_usd,now):
        super().__init__(framework,checkpoint=CHECKPOINT)
        self.configure_evidence(ledger)
        programs=self.state.setdefault('programs',{})
        if CONTROL not in programs:
            program=control_program(virtual_seed_usd,now)
            reference=ledger.record_reference('prospective_native_program_declared',program,timestamp=now)
            if not reference:raise ReplayBlocked('prospective program canonical declaration unavailable')
            programs[CONTROL]={'program':program,'spec_hash':_digest(program),'declaration_ref':reference}
            self._save()
        spec=self._candidate({'candidate_id':CONTROL})
        if spec.exact_candidate['virtual_seed_usd'] != virtual_seed_usd:
            raise ReplayBlocked('virtual seed differs from immutable study; declare a new study version')
        self.virtual_seed_usd=virtual_seed_usd
        result=self.consume('FORWARD_SHADOW',{'action':'register','candidate_id':CONTROL,
            'spec_hash':spec.frozen_hash,'experiment_id':'prospective:'+spec.frozen_hash},
            [programs[CONTROL]['declaration_ref']],now)
        if result['disposition']=='BLOCKED':raise ReplayBlocked('prospective registration blocked')

    def _candidate(self,payload):
        row=self.state.get('programs',{}).get(payload.get('candidate_id'))
        if not row:raise ReplayBlocked('prospective program not registered')
        program=row['program'];seal=_digest(program)
        expected=control_program(program['virtual_seed_usd'],program['registered_at'])
        declaration=self.read_evidence(row['declaration_ref'])
        if (program!=expected or seal!=row['spec_hash'] or payload.get('spec_hash',seal)!=seal
                or declaration.get('event_type')!='prospective_native_program_declared'
                or declaration.get('payload')!=program or declaration.get('timestamp')!=program['registered_at']):
            raise ReplayBlocked('prospective program/declaration binding changed')
        if payload.get('candidate') is not None and payload['candidate']!=program:
            raise ReplayBlocked('prospective program substitution refused')
        return PaperProgram(CONTROL,seal,program['registered_at'],copy.deepcopy(program))

    def consume(self,kind,payload,evidence_refs=(),now=None):
        if kind!='FORWARD_SHADOW':
            return self._blocked('prospective_kind','prospective registry accepts paper observations only',{})
        return super().consume(kind,payload,evidence_refs,now)

    def _forward(self,payload,refs,now):
        if payload.get('action')=='cohort' and (not isinstance(payload.get('token'),str) or not payload['token']):
            raise ReplayBlocked('prospective cohort requires explicit native token identity')
        if payload.get('action')!='register':return super()._forward(payload,refs,now)
        spec=self._candidate(payload);eid=payload.get('experiment_id')
        declaration=self.state['programs'][CONTROL]['declaration_ref']
        if (eid!='prospective:'+spec.frozen_hash or refs!=[declaration] or now<spec.frozen_at
                or payload.get('historical_receipt_hash') is not None):
            raise ReplayBlocked('prospective registration clock/identity/declaration mismatch')
        entry={'candidate_id':CONTROL,'spec_hash':spec.frozen_hash,'registered_at':spec.frozen_at,
            'study_kind':'PROSPECTIVE_UNVALIDATED_COST_CONTROL','cohorts':{},'outcomes':{}}
        prior=self.state['experiments'].setdefault(eid,entry)
        if any(prior.get(k)!=entry[k] for k in ('candidate_id','spec_hash','registered_at','study_kind')):
            raise ReplayBlocked('prospective experiment immutable binding changed')
        return self._result('DEEPENING_TASK',{'status':'REGISTERED_PROSPECTIVE_PAPER',
            'experiment_id':eid,'historical_validation':'UNMEASURED','promotion_eligible':False})

    def _outcome(self,spec,experiment,payload,refs,now):
        cohort=experiment['cohorts'].get(payload.get('cohort_id'),{}).get('payload',{})
        if (payload.get('scope')!='observed_forward_shadow' or payload.get('record_in_forward_evidence') is True
                or payload.get('equity_at_decision_usd')!=spec.exact_candidate['virtual_seed_usd']
                or not cohort.get('token') or payload.get('token')!=cohort['token']):
            raise ReplayBlocked('prospective outcomes require paper scope, declared virtual equity and isolated evidence')
        return super()._outcome(spec,experiment,payload,refs,now)

    def _execution_rows(self,spec,cohort_id,execution,refs,paper,now):
        rows=super()._execution_rows(spec,cohort_id,execution,refs,paper,now)
        key=_digest({'candidate_id':spec.hypothesis_id,'spec_hash':spec.frozen_hash,'cohort_id':cohort_id})
        eid=self.state.get('cohort_owners',{}).get(key)
        cohort=self.state.get('experiments',{}).get(eid,{}).get('cohorts',{}).get(cohort_id,{}).get('payload',{})
        token=cohort.get('token')
        for row in rows.values():
            quote=self.read_evidence(row['quote_ref']).get('payload',{})
            simulation=self.read_evidence(row['simulation_ref']).get('payload',{})
            if not token or any(item.get('token')!=token for item in (row,quote,simulation)):
                raise ReplayBlocked('prospective execution/quote/simulation token identity mismatch')
        return rows

    def _outcome_annotations(self,experiment):
        program=self._candidate(experiment).exact_candidate
        return {'study_kind':'PROSPECTIVE_UNVALIDATED_COST_CONTROL','historical_validation':'UNMEASURED',
            'promotion_eligible':False,'verified_live_profit':None,
            'live_costs_complete':False,
            'account_scope':'independent virtual account per cohort',
            'cost_scope':program['cost_scope'],
            'amount_scope':program['amount_scope'],'conversion_scope':program['conversion_scope'],
            'unmeasured_live_costs':program['unmeasured_live_costs']}

    def reconcile_forward(self):
        # Even if an identically named framework bucket exists, no control
        # outcome may become a historical survivor, alpha sample or promotion.
        return {'transferred':0,'scope':'isolated prospective cost controls only'}

    def report(self):
        return {**super().report(),'status':'PROSPECTIVE_UNVALIDATED_COST_CONTROL',
            'registered_programs':len(self.state.get('programs',{})),
            'virtual_seed_usd':self.virtual_seed_usd,'real_fills':0,
            'promotion_eligible':False,'verified_live_profit':None,
            'account_scope':'independent virtual account per cohort; do not sum as deployable capital'}


class ProspectiveNativeForward(NativeForwardPaper):
    def __init__(self,validation):
        super().__init__(validation,component=PRODUCER)

    def _compiled(self,spec):
        from src.chains.pump_route import PUMP_PROGRAM
        from src.chains.pumpswap_route import PUMPSWAP_PROGRAM
        def select(view):
            return (view.raw.get('chain') in {'solana','Solana','Solana Mainnet'}
                and view.raw.get('factory') in {PUMP_PROGRAM,PUMPSWAP_PROGRAM,'pump','pump.fun','pumpswap'})
        program=spec.exact_candidate
        return CompiledCandidate(program,Mechanism(CONTROL,select,is_control=True),
            program['quote_budget_lamports']/1_000_000_000)

    def tick(self,*args,**kwargs):
        kwargs['equity_usd']=self.validation.virtual_seed_usd
        return super().tick(*args,**kwargs)

    def report(self):
        return {**super().report(),'scope':'PROSPECTIVE_UNVALIDATED_COST_CONTROL',
            'validation':self.validation.report(),'verified_live_profit':None,'promotion_eligible':False}
