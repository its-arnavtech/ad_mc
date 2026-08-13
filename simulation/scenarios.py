"""
Phase 3 macro scenario definitions.

WHAT THESE NUMBERS ARE -- READ THIS FIRST
=========================================
Every multiplier in this module is an ASSUMPTION. None of them is measured,
fitted, or back-tested against anything in this repo. They are illustrative
judgment calls chosen to be internally coherent and directionally defensible,
so that the Phase 3 batch has something non-trivial to sweep over.

They are NOT calibrated from data. A real deployment would replace this file
wholesale by fitting each multiplier from historical periods -- e.g. regress
per-channel daily CVR / CPC / revenue-per-conversion on seasonal dummies and
on a macro indicator (unemployment claims, retail sales, consumer sentiment),
or difference the channel's own history around a known platform change, and
carry the fitted standard errors through as a distribution over the multiplier
rather than a point value. Until that is done, every scenario-level number
downstream of this file (silver, the efficient frontier, any recommendation)
inherits an uncertainty that this project does not quantify.

Treat the RANKING of scenarios as the useful output, not the magnitudes.

HOW THE MULTIPLIERS ARE APPLIED
===============================
`cell.simulate_cell` multiplies the bronze assumption MOMENTS (mean and std
together) before the engine fits its Beta / LogNormal marginals. Scaling both
moments by the same factor preserves each quantity's coefficient of variation,
so a scenario shifts the LEVEL of a channel's CVR / CPC / revenue-per-conversion
without changing its relative dispersion. See `cell.py` for why that is the
right place to apply them.

Because the funnel is revenue = (spend / CPC) * CVR * RPC, the deterministic
first-order effect of a scenario on expected revenue at a FIXED budget is:

    net revenue factor = cvr_multiplier * revenue_multiplier / cpc_multiplier

That identity is exact for the expectation here (the Jensen exp(sigma^2)
correction on E[1/CPC] is invariant under moment scaling, because sigma depends
only on the coefficient of variation). It is used as a validation target.

REASONING FOR EACH NUMBER
=========================

normal -- 1.00 / 1.00 / 1.00
    Exact no-op, by construction. This is the Phase 2 baseline; it must
    reproduce Phase 2 bitwise at the same seed, so all three values are
    literally 1.0 and `cell.py` short-circuits rather than multiplying.

seasonal_peak -- cvr 1.20, cpc 1.25, revenue 1.10
    Q4 / holiday-style demand peak.
    * CVR +20%: traffic in a peak window skews toward in-market, high-intent
      users, so a given click converts more often.
    * CPC +25%: every advertiser bids into the same window at once. Auction
      density rises, and it rises by MORE than the conversion lift. This is the
      economically important part -- an auction competes away most of a demand
      shock, so the advertiser keeps only a thin residual. Setting cpc > cvr
      is the deliberate choice that encodes "rents get bid away".
    * Revenue/conversion +10%: bigger baskets and gift purchases lift average
      order value, partly offset by heavier promotional discounting.
    * Net factor 1.20 * 1.10 / 1.25 = 1.056 -- a peak is only mildly accretive
      at a fixed budget, not a windfall.

platform_algo_change -- cvr 0.85, cpc 1.10, revenue 0.98
    An adverse platform/ad-network change: signal loss, a ranking or bidding
    overhaul, or tightened attribution.
    * CVR -15%: targeting efficiency is the direct casualty. The same spend
      reaches a less well-qualified audience.
    * CPC +10%: with noisier relevance signals the platform's quality scoring
      degrades and effective CPMs rise for the same audience, so holding click
      volume requires bidding up.
    * Revenue/conversion -2%: nearly neutral. Order value is set mostly by the
      product catalogue and price, not by the ad platform; the small decline
      reflects a mix shift toward marginally lower-value converters.
    * Net factor 0.85 * 0.98 / 1.10 = 0.757 -- the worst of the four, because
      it is the only scenario where CVR falls while CPC rises. Nothing cushions
      it.

recession -- cvr 0.75, cpc 0.82, revenue 0.90
    Broad demand contraction.
    * CVR -25%: discretionary purchases get deferred. This is the largest and
      most direct recession effect.
    * CPC -18%: coherence requirement. Advertisers cut budgets in a downturn
      (ad spend is famously procyclical and falls faster than GDP), so auction
      pressure falls and clicks get CHEAPER. A recession that lowered CVR while
      also raising CPC would be internally incoherent -- the same contraction
      that stops consumers buying also stops advertisers bidding.
    * Revenue/conversion -10%: trade-down to cheaper SKUs plus heavier
      discounting to move volume.
    * Net factor 0.75 * 0.90 / 0.82 = 0.823. Note the cushion: without the
      auction relief the factor would be 0.75 * 0.90 = 0.675. Roughly a third
      of the demand shock is recovered through cheaper media. That cushioning
      effect is the single most interesting thing this scenario set says, and
      it is entirely a consequence of the assumed cpc_multiplier -- if that
      0.82 is wrong, the conclusion is wrong.

Net-factor summary (fixed $500k budget, relative to normal):
    seasonal_peak         1.056
    normal                1.000
    recession             0.823
    platform_algo_change  0.757

WHAT THIS SCENARIO MODEL DELIBERATELY DOES NOT DO
=================================================
* Multipliers are UNIFORM ACROSS CHANNELS. In reality a recession hits
  brand/upper-funnel channels (display, video) much harder than paid search,
  which is demand-capture. Per-channel multipliers are the obvious next
  refinement and are out of scope here.
* The CORRELATION MATRIX IS HELD FIXED across scenarios. This is the biggest
  simplification in the file. Real cross-channel correlation rises in a
  downturn -- shocks become common-mode -- so a recession's true risk is
  understated here, in the tail specifically.
* Multipliers are DETERMINISTIC point values, so a scenario contributes no
  uncertainty of its own; all simulated dispersion still comes from the
  per-channel marginals.
* Budget is held fixed across scenarios. A real advertiser reallocates budget
  in response to a scenario (spends more into a peak, less into a recession).
  Holding budget fixed is what makes the scenarios comparable, but it means
  these are "what if I spent the same $500k in this world", not a forecast of
  what a business would actually do.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


NORMAL_SCENARIO_ID = "normal"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    scenario_name: str
    description: str
    cvr_multiplier: float
    cpc_multiplier: float
    revenue_multiplier: float
    rationale: str

    @property
    def net_revenue_factor(self) -> float:
        """Deterministic first-order effect on expected revenue at fixed budget.

        revenue = (spend / CPC) * CVR * RPC, so scaling the three inputs scales
        the expectation by cvr * revenue / cpc. Exact for the mean here.
        """
        return self.cvr_multiplier * self.revenue_multiplier / self.cpc_multiplier

    @property
    def is_noop(self) -> bool:
        return (self.cvr_multiplier == 1.0
                and self.cpc_multiplier == 1.0
                and self.revenue_multiplier == 1.0)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="normal",
        scenario_name="Normal",
        description=(
            "Baseline conditions. Bronze assumptions are used exactly as measured, "
            "with no macro overlay of any kind."
        ),
        cvr_multiplier=1.0,
        cpc_multiplier=1.0,
        revenue_multiplier=1.0,
        rationale=(
            "ASSUMPTION-FREE BY CONSTRUCTION. All three multipliers are exactly 1.0 so "
            "this scenario is a bitwise no-op on the Phase 2 engine: it is the control "
            "arm that every other scenario is measured against, and it is the cell used "
            "to prove the distributed Phase 3 engine still reproduces the validated "
            "Phase 2 result ($968,027 / $169,364 / $715,966 at seed 20260808)."
        ),
    ),
    Scenario(
        scenario_id="seasonal_peak",
        scenario_name="Seasonal Peak",
        description=(
            "Q4-style demand peak: high purchase intent, heavy promotional calendar, "
            "and every competitor bidding into the same narrow window."
        ),
        cvr_multiplier=1.20,
        cpc_multiplier=1.25,
        revenue_multiplier=1.10,
        rationale=(
            "ASSUMPTION, not measured. CVR +20%: peak-window traffic skews toward "
            "in-market, high-intent users. CPC +25%: auction density rises when every "
            "advertiser concentrates spend into the same weeks, and it is set ABOVE the "
            "CVR lift on purpose -- an auction competes away most of a demand shock, so "
            "the advertiser retains only a thin residual. Revenue/conversion +10%: "
            "larger baskets and gifting lift average order value, partly given back "
            "through discounting. Net effect on expected revenue at fixed budget is "
            "1.20 * 1.10 / 1.25 = 1.056, i.e. mildly accretive rather than a windfall. "
            "Weak point: the 1.25 CPC figure carries the whole conclusion -- at CPC "
            "1.32 or above the scenario flips to revenue-negative."
        ),
    ),
    Scenario(
        scenario_id="platform_algo_change",
        scenario_name="Platform Algorithm Change",
        description=(
            "An adverse change at the ad platform: signal loss / attribution tightening "
            "or a bidding and ranking overhaul that degrades targeting quality."
        ),
        cvr_multiplier=0.85,
        cpc_multiplier=1.10,
        revenue_multiplier=0.98,
        rationale=(
            "ASSUMPTION, not measured. CVR -15%: degraded targeting is the direct "
            "casualty -- the same spend now reaches a less well-qualified audience. "
            "CPC +10%: noisier relevance signals worsen quality scoring, so effective "
            "CPMs rise and holding click volume requires bidding up. "
            "Revenue/conversion -2%: near-neutral, since order value is set by the "
            "catalogue and price rather than the ad platform; the small decline is a "
            "mix shift toward marginally lower-value converters. Net factor "
            "0.85 * 0.98 / 1.10 = 0.757 -- the worst of the four scenarios, because it "
            "is the only one where CVR falls while CPC rises, so nothing cushions it."
        ),
    ),
    Scenario(
        scenario_id="recession",
        scenario_name="Recession",
        description=(
            "Broad demand contraction: consumers defer discretionary purchases and "
            "advertisers cut budgets, so auctions soften at the same time."
        ),
        cvr_multiplier=0.75,
        cpc_multiplier=0.82,
        revenue_multiplier=0.90,
        rationale=(
            "ASSUMPTION, not measured. CVR -25%: deferred discretionary purchases, the "
            "largest and most direct recession effect. CPC -18%: this is the internal "
            "coherence requirement -- ad spend is procyclical and contracts faster than "
            "the wider economy, so auction pressure falls and clicks get CHEAPER. A "
            "recession that lowered CVR while also raising CPC would be incoherent: the "
            "same contraction that stops consumers buying also stops advertisers "
            "bidding. Revenue/conversion -10%: trade-down to cheaper SKUs plus "
            "discounting. Net factor 0.75 * 0.90 / 0.82 = 0.823 -- versus 0.675 with no "
            "auction relief, so roughly a third of the demand shock is recovered "
            "through cheaper media. That cushioning result rests entirely on the "
            "assumed 0.82; if the real auction response is weaker, the recession is "
            "materially worse than this model says."
        ),
    ),
)

SCENARIO_IDS: tuple[str, ...] = tuple(s.scenario_id for s in SCENARIOS)
SCENARIOS_BY_ID: dict[str, Scenario] = {s.scenario_id: s for s in SCENARIOS}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS_BY_ID[scenario_id]
    except KeyError:
        raise ValueError(
            f"unknown scenario_id {scenario_id!r}; expected one of {list(SCENARIO_IDS)}"
        ) from None


def scenario_rows() -> list[dict]:
    """Rows for `bronze.scenario_definitions`, in declaration order.

    Columns: scenario_id, scenario_name, description, cvr_multiplier,
    cpc_multiplier, revenue_multiplier, rationale, net_revenue_factor.

    `net_revenue_factor` is derived (cvr * revenue / cpc) and is carried into
    the table on purpose: it is the single number that says what a scenario
    does to expected revenue at a fixed budget, and having it stored alongside
    the raw multipliers lets any downstream check confirm the simulation
    against it without re-deriving the funnel algebra.
    """
    rows = []
    for s in SCENARIOS:
        row = asdict(s)
        row["net_revenue_factor"] = s.net_revenue_factor
        rows.append(row)
    return rows


# Fail loudly at import time if the baseline ever stops being a true no-op --
# the whole Phase 2 -> Phase 3 equivalence argument depends on it.
_normal = SCENARIOS_BY_ID[NORMAL_SCENARIO_ID]
assert _normal.is_noop, "the 'normal' scenario must have all multipliers exactly 1.0"
del _normal
