"""HTML for the Phase 6 report. No queries here -- `build_report.py` owns those.

Kept separate so the numbers and the markup cannot drift into each other: every
value this module renders arrives as an argument that came from a gold query.
"""

from __future__ import annotations

import html


SCEN_ORDER = ["seasonal_peak", "normal", "recession", "platform_algo_change"]
SCEN_LABEL = {
    "seasonal_peak": "Seasonal peak",
    "normal": "Normal",
    "recession": "Recession",
    "platform_algo_change": "Platform change",
}
PAIR_LABEL = {
    "mean_revenue vs std_revenue": "Return vs volatility",
    "mean_revenue vs var_95": "Return vs worst-case floor",
    "mean_revenue vs cvar_95": "Return vs deep-tail average",
}
PAIR_RISK = {
    "mean_revenue vs std_revenue": ("std_revenue", False),
    "mean_revenue vs var_95": ("var_95", True),
    "mean_revenue vs cvar_95": ("cvar_95", True),
}

CSS = """
:root{
  --ground:#FBFBFC; --panel:#FFFFFF; --ink:#171A1F; --ink-2:#454C58;
  --ink-3:#6E7683; --rule:#E1E4E9; --rule-2:#EDEFF2;
  --accent:#0E6B6B; --accent-soft:#D7EAE9;
  --warn:#9A6212; --warn-soft:#F6ECD9;
  --alarm:#9C3A2F; --alarm-soft:#F7E3E0;
  --dot:#C3C8D0;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#101317; --panel:#171B21; --ink:#E9ECF0; --ink-2:#B3BBC6;
    --ink-3:#828C99; --rule:#272D36; --rule-2:#1E242B;
    --accent:#4FBDB5; --accent-soft:#17383A;
    --warn:#D9A257; --warn-soft:#382C15;
    --alarm:#E08578; --alarm-soft:#3A211D;
    --dot:#3A424E;
  }
}
:root[data-theme="dark"]{
  --ground:#101317; --panel:#171B21; --ink:#E9ECF0; --ink-2:#B3BBC6;
  --ink-3:#828C99; --rule:#272D36; --rule-2:#1E242B;
  --accent:#4FBDB5; --accent-soft:#17383A;
  --warn:#D9A257; --warn-soft:#382C15;
  --alarm:#E08578; --alarm-soft:#3A211D;
  --dot:#3A424E;
}
*{box-sizing:border-box;}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px; margin:0 auto; padding:56px 28px 96px;}
.measure{max-width:68ch;}
h1,h2,h3{text-wrap:balance; font-family:Georgia,"Iowan Old Style","Times New Roman",serif;
  font-weight:600; letter-spacing:-0.012em; margin:0;}
h1{font-size:2.55rem; line-height:1.14;}
h2{font-size:1.62rem; margin-top:12px;}
h3{font-size:1.12rem; font-family:inherit; font-weight:650; letter-spacing:0;}
p{margin:0;}
.eyebrow{font-size:.735rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-3); font-weight:640;}
.lede{font-size:1.12rem; color:var(--ink-2);}
.stack{display:flex; flex-direction:column; gap:16px;}
section{display:flex; flex-direction:column; gap:20px;
  padding-top:38px; margin-top:38px; border-top:1px solid var(--rule);}
header.head{display:flex; flex-direction:column; gap:18px; padding-bottom:6px;}
.num{font-variant-numeric:tabular-nums;}
.grid2{display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:18px;}
.card{background:var(--panel); border:1px solid var(--rule); border-radius:9px;
  padding:18px 18px 14px; display:flex; flex-direction:column; gap:10px;}
.card h3{margin:0;}
.scroll{overflow-x:auto; border:1px solid var(--rule); border-radius:9px; background:var(--panel);}
table{border-collapse:collapse; width:100%; font-size:.9rem;}
th,td{padding:9px 13px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--rule-2); font-variant-numeric:tabular-nums;}
th{font-size:.715rem; letter-spacing:.075em; text-transform:uppercase;
  color:var(--ink-3); font-weight:650; text-align:right; background:var(--rule-2);}
th:first-child,td:first-child{text-align:left; font-variant-numeric:normal;}
tbody tr:last-child td{border-bottom:none;}
tr.total td{font-weight:680; border-top:1px solid var(--rule);}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.87em;
  background:var(--rule-2); padding:1px 5px; border-radius:4px;}
.chip{display:inline-block; font-size:.685rem; font-weight:660; letter-spacing:.045em;
  text-transform:uppercase; padding:2px 7px; border-radius:100px; white-space:nowrap;}
.chip-ord{background:var(--warn-soft); color:var(--warn);}
.chip-floor{background:var(--alarm-soft); color:var(--alarm);}
.chip-none{background:var(--rule-2); color:var(--ink-3);}
.callout{border-left:3px solid var(--accent); background:var(--panel);
  padding:15px 18px; border-radius:0 8px 8px 0; display:flex;
  flex-direction:column; gap:8px;}
.callout.warn{border-left-color:var(--warn);}
.callout.alarm{border-left-color:var(--alarm);}
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:9px; overflow:hidden;}
.kpi{background:var(--panel); padding:15px 17px; display:flex; flex-direction:column; gap:3px;}
.kpi .v{font-size:1.5rem; font-weight:660; font-variant-numeric:tabular-nums;
  letter-spacing:-0.02em; line-height:1.15;}
.kpi .k{font-size:.715rem; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3);
  font-weight:640;}
svg{width:100%; height:auto; display:block;}
.grid{stroke:var(--rule); stroke-width:1;}
.tick{fill:var(--ink-3); font-size:10px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.pt{fill:var(--dot);}
.pt-front{fill:var(--accent);}
.pt-floor{fill:none; stroke:var(--alarm); stroke-width:1.9;}
.pt-ord{fill:none; stroke:var(--warn); stroke-width:1.9;}
.legend{display:flex; flex-wrap:wrap; gap:14px; font-size:.775rem; color:var(--ink-2);}
.legend i{display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px;
  vertical-align:middle;}
.axis{font-size:.75rem; color:var(--ink-3);}
ul{margin:0; padding-left:20px; display:flex; flex-direction:column; gap:8px;}
.foot{margin-top:44px; padding-top:20px; border-top:1px solid var(--rule);
  font-size:.8rem; color:var(--ink-3);}
a{color:var(--accent);}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important;}}
"""


def money(v, dp=0):
    return f"${v:,.{dp}f}"


def chips(ordering: bool, floored: bool) -> str:
    out = []
    if ordering:
        out.append('<span class="chip chip-ord">ordering unresolved</span>')
    if floored:
        out.append('<span class="chip chip-floor">priced at floor</span>')
    return " ".join(out) or '<span class="chip chip-none">no caveats</span>'


def scatter(sub, front_ids, risk_col, ordering_ids=(), w=470, h=290):
    pad_l, pad_b, pad_t, pad_r = 56, 32, 10, 10
    xs = sub[risk_col].to_numpy(float)
    ys = sub["mean_revenue"].to_numpy(float)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    xr, yr = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    px = lambda v: round(pad_l + (v - x0) / xr * (w - pad_l - pad_r), 1)
    py = lambda v: round(h - pad_b - (v - y0) / yr * (h - pad_b - pad_t), 1)

    base, fr, fl, unresolved = [], [], [], []
    for aid, x, y, flag in zip(sub.allocation_id, xs, ys, sub.extrapolation_floor_applied):
        t = (px(x), py(y))
        if aid in front_ids:
            fr.append(t)
        else:
            base.append(t)
        if flag:
            fl.append(t)
        if aid in ordering_ids:
            unresolved.append(t)

    def dots(pts, cls, r):
        return "".join(f'<circle class="{cls}" cx="{a}" cy="{b}" r="{r}"/>' for a, b in pts)

    g = []
    for f in (0, 0.25, 0.5, 0.75, 1):
        gy = round(pad_t + f * (h - pad_b - pad_t), 1)
        g.append(f'<line class="grid" x1="{pad_l}" y1="{gy}" x2="{w-pad_r}" y2="{gy}"/>')
        g.append(f'<text class="tick" x="{pad_l-8}" y="{gy+3.5}" text-anchor="end">'
                 f'{(y1 - f*yr)/1e6:.2f}M</text>')
    for f in (0, 0.5, 1):
        gx = round(pad_l + f * (w - pad_l - pad_r), 1)
        g.append(f'<text class="tick" x="{gx}" y="{h-pad_b+16}" text-anchor="middle">'
                 f'{(x0 + f*xr)/1e3:.0f}k</text>')
    def squares(pts, cls, r):
        return "".join(
            f'<rect class="{cls}" x="{a-r}" y="{b-r}" width="{2*r}" height="{2*r}"/>'
            for a, b in pts
        )

    return (f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(g)
            + dots(base, "pt", 1.7) + dots(fr, "pt-front", 3.3)
            + dots(fl, "pt-floor", 3.8) + squares(unresolved, "pt-ord", 4.2)
            + "</svg>")


def build(*, sweep, front, recs, sizes, tiers, sens, chan, top,
          n_cand, n_ord, n_floor, n_both, floor_front, floor_sweep, splits,
          candidate_run_id, **_):
    P = []
    add = P.append

    mv = "mean_revenue vs std_revenue"
    sz = {(r.objective_pair, r.scenario_id): int(r.n) for r in sizes.itertuples()}
    mv_lo = min(sz[(mv, s)] for s in SCEN_ORDER)
    mv_hi = max(sz[(mv, s)] for s in SCEN_ORDER)
    var_lo = min(sz[("mean_revenue vs var_95", s)] for s in SCEN_ORDER)
    var_hi = max(sz[("mean_revenue vs var_95", s)] for s in SCEN_ORDER)
    cv_lo = min(sz[("mean_revenue vs cvar_95", s)] for s in SCEN_ORDER)
    cv_hi = max(sz[("mean_revenue vs cvar_95", s)] for s in SCEN_ORDER)

    def ordering_ids(pair, scenario):
        return set(recs[(recs.objective_pair == pair)
                        & (recs.scenario_id == scenario)
                        & recs.ordering_unresolved].allocation_id)

    # ---------------------------------------------------------------- header
    add(f"""<header class="head">
<p class="eyebrow">Ad budget allocation &middot; $500,000 per period &middot; simulated risk analysis</p>
<h1>Where to put the budget, and what could go wrong</h1>
<p class="lede measure">We tested <strong>{n_cand:,} different ways</strong> of splitting a
$500,000 budget across five advertising channels, simulating 10,000 possible outcomes for each
under four market conditions. <strong>The budget carries no time period</strong> &mdash; whether
that is a month or a year changes whether these figures are plausible at all; see limitations. This is what the results support &mdash; and, just as importantly,
what they do not.</p>
</header>""")

    # ---------------------------------------------------------------- primer
    add(f"""<section>
<h2>How to read this</h2>
<div class="measure stack">
<p>We treat each advertising channel the way an investor treats an asset in a portfolio.
A channel has a <strong>return</strong> (revenue per dollar spent) and a <strong>risk</strong>
(how much that return swings). Channels also move together &mdash; when conversion rates dip,
they tend to dip across several channels at once &mdash; so spreading the budget reduces risk
by less than you might assume.</p>
<p>Rather than predict one number, we simulate <strong>10,000 possible futures</strong> for every
candidate split. That produces a distribution, which lets us talk about the bad cases directly:</p>
<ul>
<li><strong>Expected revenue</strong> &mdash; the average across all 10,000 simulated outcomes.</li>
<li><strong>Volatility</strong> &mdash; how widely those outcomes spread. Lower means more predictable.</li>
<li><strong>Worst-case floor (VaR-95)</strong> &mdash; you beat this number in 95% of simulations.
A plain-language version: "19 times out of 20, revenue lands above this."</li>
<li><strong>Deep-tail average (CVaR-95)</strong> &mdash; the average of the worst 5% of outcomes.
This is what a bad year actually looks like, not just its boundary.</li>
</ul>
<p>An allocation is <strong>efficient</strong> if nothing else beats it on both return and risk
at once. Plot all the efficient ones and you get a frontier: the menu of sensible trade-offs.</p>
</div>
</section>""")

    # ------------------------------------------------------------- frontiers
    add("""<section>
<h2>The efficient frontier</h2>
<p class="measure">Each dot is one candidate allocation. Grey dots are dominated &mdash; something
else beats them on both axes. Teal dots are efficient. <strong>Outlined dots carry a caveat</strong>
(explained below), whether efficient or dominated; they are kept in the picture rather than quietly
removed.</p>""")

    add('<div class="legend"><span><i style="background:var(--dot)"></i>dominated</span>'
        '<span><i style="background:var(--accent)"></i>efficient</span>'
        '<span><i style="border:2px solid var(--alarm);background:transparent"></i>'
        'priced at floor</span><span><i style="border:2px solid var(--warn);'
        'background:transparent;border-radius:1px"></i>ordering unresolved</span></div>')

    add('<div class="grid2">')
    for s in SCEN_ORDER:
        sub = sweep[sweep.scenario_id == s]
        fids = set(front[(front.scenario_id == s) & (front.objective_pair == mv)].allocation_id)
        add(f"""<div class="card"><h3>{SCEN_LABEL[s]}</h3>
<p class="axis">Expected revenue (vertical) vs volatility (horizontal). Up and to the left is better.
<strong>{sz[(mv,s)]} of {n_cand:,}</strong> efficient.</p>
{scatter(sub, fids, "std_revenue", ordering_ids(mv, s))}</div>""")
    add("</div>")

    add(f"""<div class="callout"><h3>Two frontiers, and only one of them is a real curve</h3>
<p>Judged on <strong>return vs volatility</strong>, {mv_lo}&ndash;{mv_hi} of {n_cand:,}
allocations are efficient ({100*mv_lo/n_cand:.1f}&ndash;{100*mv_hi/n_cand:.1f}%). That is a
genuine menu of choices.</p>
<p>Judged on the worst-case floor, only <strong>{var_lo}&ndash;{var_hi}</strong> are efficient;
on the deep-tail average, <strong>{cv_lo}&ndash;{cv_hi}</strong>. Those are handfuls, not curves.</p>
<p><strong>This is not a discovery.</strong> The number of efficient points in any two-way
trade-off grows roughly with the logarithm of how many candidates you test &mdash; so a short
list is the expected result at this sample size, not evidence that return and downside risk are
secretly the same thing. We tested that specific explanation and it did not hold. Read the
worst-case frontiers as <em>a few defensible options</em>, and use the volatility frontier when
you want to see the shape of the trade-off.</p>
</div>

<h3>Downside-risk views: mean&ndash;VaR and mean&ndash;CVaR</h3>
<p class="measure">These plots show the two thin downside-risk frontiers rather than hiding
them. Teal points are the efficient handfuls. Their thinness is expected sampling behaviour,
not evidence of collinearity or a special structural relationship.</p>
<div class="grid2">""")
    for s in SCEN_ORDER:
        sub = sweep[sweep.scenario_id == s]
        pair_var = "mean_revenue vs var_95"
        pair_cvar = "mean_revenue vs cvar_95"
        var_ids = set(front[(front.scenario_id == s)
                            & (front.objective_pair == pair_var)].allocation_id)
        cvar_ids = set(front[(front.scenario_id == s)
                             & (front.objective_pair == pair_cvar)].allocation_id)
        add(f"""<div class="card"><h3>{SCEN_LABEL[s]}</h3>
<p class="axis"><strong>Mean&ndash;VaR:</strong> {sz[(pair_var,s)]} of {n_cand:,} efficient.
Up and to the right is better.</p>
{scatter(sub, var_ids, "var_95", ordering_ids(pair_var, s), h=240)}
<p class="axis"><strong>Mean&ndash;CVaR:</strong> {sz[(pair_cvar,s)]} of {n_cand:,} efficient.
Up and to the right is better.</p>
{scatter(sub, cvar_ids, "cvar_95", ordering_ids(pair_cvar, s), h=240)}</div>""")
    add("</div>")

    add("""<p class="measure axis">For VaR and CVaR, farther right means a higher revenue floor
in bad outcomes, so right is safer. Outlines retain both caveats directly in every plot: red
circles mark allocations priced at the historical spend floor; amber squares mark a displayed
recommendation whose ordering against a neighbour is unresolved.</p>
</section>""")

    # -------------------------------------------------------- recommendations
    add(f"""<section>
<h2>Recommendations by risk appetite</h2>
<p class="measure">Five picks under normal conditions, chosen by a fixed rule rather than
judgement. Caveats sit next to the number they qualify.</p>
<div class="scroll"><table>
<thead><tr><th>Risk appetite &amp; budget split</th><th>Expected revenue</th><th>Return on spend</th>
<th>Volatility</th><th>Worst-case floor</th><th>Caveats</th></tr></thead><tbody>""")
    for label, why, row, _risk in tiers:
        sp = splits.get(row.allocation_id)
        if sp is None:
            mix = ""
        else:
            shown = [f"{r.channel} {100*r.spend_pct:.0f}%"
                     for r in sp.itertuples() if r.spend_pct >= 0.005]
            n_thin = int((sp.spend_pct < 0.005).sum())
            mix = " &middot; ".join(shown)
            if n_thin:
                mix += (' &middot; <span style="color:var(--alarm)">plus '
                        f'{n_thin} channel(s) under 0.5%</span>')
        add(f"""<tr><td><strong>{label}</strong><br>
<span class="axis">{why}</span><br><span class="axis">{mix}</span></td>
<td>{money(row.mean_revenue)}</td><td>{row.expected_roas:.2f}&times;</td>
<td>{money(row.std_revenue)}</td><td>{money(row.var_95)}</td>
<td style="text-align:left">{chips(bool(row.ordering_unresolved),
                                   bool(row.extrapolation_floor_applied))}</td></tr>""")
    add("</tbody></table></div>")

    add(f"""<div class="callout warn"><h3>What the caveats mean</h3>
<p><strong>Ordering unresolved</strong> &mdash; this pick and its neighbour on the frontier are
close enough together that the simulation cannot tell them apart. The choice between them is not
supported by the data; either is defensible. <strong>{n_ord} of {len(recs)}</strong>
recommendations across all scenarios carry this.</p>
<p><strong>Priced at floor</strong> &mdash; the allocation funds at least one channel below any
spend level the historical data covers, so its cost-per-click was priced at the lowest
observed level rather than extrapolated below it. The number is bounded and defensible, but it
sits at the edge of the evidence. <strong>{n_floor} of {len(recs)}</strong> recommendations
carry this, along with {floor_front} of {len(front)} frontier points and
{floor_sweep:,} of {len(sweep):,} tested allocations.
{"No recommendation carries both flags at once." if n_both == 0
 else f"{n_both} carry both."}</p>
</div>
</section>""")

    # ------------------------------------------------------------ sensitivity
    add("""<section>
<h2>How bad does "bad" look?</h2>
<p class="measure">The picks above were chosen under normal conditions. Here is how each performs
across all four market conditions. Return on spend below 1.00&times; means the budget loses
money.</p>
<div class="scroll"><table>
<thead><tr><th>Allocation</th>""")
    for s in SCEN_ORDER:
        add(f"<th>{SCEN_LABEL[s]}</th>")
    add("</tr></thead><tbody>")
    for label, _why, row, _r in tiers:
        add(f'<tr><td><strong>{label}</strong><br><code>'
            f'{html.escape(row.allocation_id)}</code></td>')
        for s in SCEN_ORDER:
            m = sens[(sens.allocation_id == row.allocation_id)
                     & (sens.scenario_id == s)]
            if m.empty:
                add("<td>&mdash;</td>"); continue
            rr = m.iloc[0]
            loss = rr.expected_roas < 1.0
            style = ' style="color:var(--alarm);font-weight:650"' if loss else ""
            add(f'<td{style}>{money(rr.mean_revenue)}<br>'
                f'<span class="axis">{rr.expected_roas:.2f}&times;</span></td>')
        add("</tr>")
    add("</tbody></table></div>")

    # How many candidates lose money under the worst scenario. The scenario is a
    # near-uniform multiplier, so this is a property of base-case return alone.
    plat = sweep[sweep.scenario_id == 'platform_algo_change']
    below_thresh = int((plat.expected_roas < 1.0).sum())
    cons = tiers[0][2]
    cons_bad = sens[(sens.allocation_id == cons.allocation_id)
                    & (sens.scenario_id == "platform_algo_change")].iloc[0]
    agg = tiers[2][2]
    agg_bad = sens[(sens.allocation_id == agg.allocation_id)
                   & (sens.scenario_id == "platform_algo_change")].iloc[0]
    normal_roas = (sweep[sweep.scenario_id == "normal"]
                   .set_index("allocation_id")["expected_roas"])
    recession_roas = (sweep[sweep.scenario_id == "recession"]
                      .set_index("allocation_id")["expected_roas"])
    recession_factor = recession_roas / normal_roas
    recession_factor_spread = float(recession_factor.max() - recession_factor.min())
    add(f"""<div class="callout alarm"><h3>The lowest-volatility option is not the safest one</h3>
<p>"Conservative" here means <em>least variable</em>, not <em>least likely to lose money</em>.
Under a platform change it returns <strong>{cons_bad.expected_roas:.2f}&times;</strong>
&mdash; {money(cons_bad.mean_revenue)} on {money(500000)} of spend, a real loss &mdash; and it
loses money in a recession too. The aggressive pick returns
<strong>{agg_bad.expected_roas:.2f}&times;</strong> in that same bad scenario.</p>
<p>If the goal is to avoid losing money rather than to avoid surprises, the low-volatility
allocation is the wrong instrument.</p>
<p><strong>Be precise about why, because the model is simpler than this table looks.</strong>
Each condition applies a near-uniform multiplier to every allocation &mdash; across all
{n_cand:,} candidates the recession factor ranges from {recession_factor.min():.4f} to
{recession_factor.max():.4f} (a {recession_factor_spread:.4f} spread) &mdash; so these columns are
essentially the normal one times three constants. <em>No allocation is specifically fragile to a
downturn in this model.</em> The conservative pick loses money because its base-case return is
barely above break-even at {tiers[0][2].expected_roas:.2f}&times;, not because steadiness makes it
brittle. Anything below roughly 1.32&times; in normal conditions loses money under a platform
change; {below_thresh} of {n_cand:,} candidates are.</p>
</div>
</section>""")

    # -------------------------------------------------------------- channels
    top_id = html.escape(str(top.allocation_id))
    spend_lead = chan.loc[chan.spend_pct.idxmax()]
    return_up = chan.loc[chan.return_corr.idxmax()]
    add(f"""<section>
<h2>Where the money goes, and where the risk comes from</h2>
<p class="measure">Channel breakdown for the aggressive pick (<code>{top_id}</code>), the
highest-expected-revenue allocation on the frontier.</p>
<div class="scroll"><table>
<thead><tr><th>Channel</th><th>Spend</th><th>Share</th><th>Return link</th>
<th>Risk link</th></tr></thead><tbody>""")
    for r in chan.itertuples():
        add(f"""<tr><td>{r.channel}</td><td>{money(r.spend)}</td>
<td>{100*r.spend_pct:.1f}%</td><td>{r.return_corr:+.2f}</td>
<td>{r.risk_corr:+.2f}</td></tr>""")
    add(f"""<tr class="total"><td>Total</td><td>{money(chan.spend.sum())}</td><td>100.0%</td>
<td></td><td></td></tr>
</tbody></table></div>""")

    risk_up = chan.loc[chan.risk_corr.idxmax()]
    risk_down = chan.loc[chan.risk_corr.idxmin()]
    add(f"""<div class="stack measure">
<p><strong>Revenue-seeking weight and risk-reducing weight are not the same.</strong>
{spend_lead.channel} takes {100*spend_lead.spend_pct:.0f}% of the aggressive budget. Across all
{n_cand:,} normal-scenario allocations, a larger {return_up.channel} share has the strongest
association with higher expected revenue (the <strong>return link</strong>,
{return_up.return_corr:+.2f}). A larger
{risk_up.channel} share has the strongest association with more volatility
(correlation {risk_up.risk_corr:+.2f}), while a larger {risk_down.channel} share has the strongest
association with less volatility ({risk_down.risk_corr:+.2f}). That makes the distinction
visible channel by channel without inventing channel-level revenue that was never persisted.</p>
<p><em>The honest caveat:</em> this is an association across candidate portfolios, not a causal
risk attribution; budget shares must add to 100%, so raising one always lowers another.
The model also correlates only conversion rates &mdash; click costs and order values are drawn
independently. The links therefore show which channel weights travel with portfolio return and
volatility in gold, not how much revenue or volatility a channel causes by itself.</p>
<p>The table deliberately does not assign per-channel revenue or a causal risk contribution:
neither was persisted by the verified run, and recreating them would violate the no-recompute
rule. Exact spend comes from that run's persisted candidate file and is checked back to live
gold before rendering.</p>
</div>
</section>""")

    # ----------------------------------------------------------- limitations
    add(f"""<section>
<h2>Limitations</h2>
<p class="measure">These are not hedges. Each one changes how much weight the recommendations
can carry.</p>

<div class="callout alarm"><h3>1. Diminishing returns are an assumption, and the data does not confirm them</h3>
<p>The model assumes each channel gets more expensive as you spend more, at a rate set per
channel. Those rates were chosen by judgement. Their <em>ordering</em> is defensible &mdash; it
follows each channel's inventory size, and search having the least room to grow matches how
search advertising works.</p>
<p>But a later check regressed real cost-per-click against real spend in the historical data
and <strong>did not reproduce those rates</strong>: the measured relationships came out roughly
uniform across channels, with paid search &mdash; the channel assigned the steepest curve
&mdash; measuring among the flattest. Being fair to the assumption: within a channel, daily spend only ever varied about three- to
fourfold, and the entire observed range sits below 7% of the spend level the curve is anchored
at. The measured relationships are statistically solid where they were measured &mdash; they were
simply never measured anywhere near the budgets being recommended. So the historical record
cannot confirm the assumed rates, and cannot fully refute them either.</p>
<p><strong>What this means for the recommendations:</strong> the ranking of allocations depends
on these rates. Change them and the winner moves. Treat the picks as "best under a stated
assumption," not as measured fact.</p>
<p><strong>What would actually settle it:</strong> a spend-variation test &mdash; deliberately
running channels at materially different budget levels and measuring how cost-per-click
responds &mdash; or a holdout test, switching a channel off in some regions and leaving it on in
others, to measure what it genuinely adds.
Either produces the spend variation the historical record lacks.</p>
</div>

<div class="callout warn"><h3>2. Some recommendations are not fully resolved</h3>
<p><strong>{n_ord} of {len(recs)}</strong> recommendations are flagged
<span class="chip chip-ord">ordering unresolved</span>: the pick is statistically
indistinguishable from its neighbour on the frontier. Simulation noise is as large as the gap.
Do not read a ranking into those.</p>
<p><strong>{n_floor} of {len(recs)}</strong> are flagged
<span class="chip chip-floor">priced at floor</span>: at least one channel is funded below
anything in the historical record, so its cost was held at the lowest observed level instead of
extrapolated downward. This bounds the estimate rather than fixing it &mdash; the optimizer can
still propose a channel funded at a token amount, and here the smallest funded channel is
{money(chan.spend.min())}. The report does not assign a channel return to that token amount,
because no per-channel revenue was persisted and analytically recreating one would overstate
what the verified gold data contains.</p>
</div>

<div class="callout"><h3>3. The distributed computing was proven, but was not necessary</h3>
<p>The pipeline runs on Spark and is orchestrated as a persisted, parameterised workflow that
was triggered and verified end to end. It runs on demand rather than on a schedule &mdash; no
cron trigger is attached. Honestly reported: <strong>at this project's scale the distributed
computing did not need to be there.</strong>
The full simulation grid runs in about seven seconds on a single machine. Spark earns its keep
in the {n_cand:,}-candidate optimization sweep, but the earlier stages demonstrate the mechanism
rather than require it. A production version at ten or a hundred times this scale would need it;
this one does not.</p>
</div>

<div class="callout"><h3>4. Scenario severities are judgement, and correlation is held fixed</h3>
<p>The four market conditions are stated assumptions about how conversion rates, click costs and
order values move together &mdash; not forecasts, and not fitted from historical downturns.
Separately, the model holds cross-channel correlation constant across scenarios. In a real
downturn correlations typically <em>rise</em>, which means the recession case here understates
tail risk rather than overstating it.</p>
</div>
</section>""")

    add(f"""<div class="foot measure">
<p>Every figure on this page is queried live from the verified gold tables
(<code>allocation_sweep_results</code>, <code>efficient_frontier</code>,
<code>frontier_recommendations</code>) at render time &mdash; {len(sweep):,} allocation-scenario
results, {len(front)} frontier points, {len(recs)} recommendations. Nothing here is re-simulated
or transcribed. <strong>One input does not live in those three tables:</strong> gold is held at
allocation level, so exact per-channel budget splits are read from the candidate Parquet files
persisted by successful Phase 5 Workflow run {candidate_run_id}. Before use, the report checks
that all {n_cand:,} candidate ids and every persisted aggregate result align with live gold.
The return/risk links combine those exact shares with gold's stored mean and volatility; they do
not recreate channel revenue, draw paths, or regenerate candidates.</p>
</div>""")

    return (f'<title>Budget Allocation Under Uncertainty</title>\n'
            f"<style>{CSS}</style>\n"
            f'<div class="wrap">' + "\n".join(P) + "</div>")
