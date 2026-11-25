// The replication on Criteo's ad experiments (Criteo Uplift Prediction Dataset v2.1,
// Diemert et al. 2018, CC BY-NC-SA 4.0). Every number comes from data/criteo.json, which
// `python -m uplift_targeting.demo` writes from results/criteo/. Treatment there is slightly
// predictable from the features, so every test-set estimate comes twice, as randomized and
// reweighted; one control switches everything in this section. No model runs in the browser.
//
// render() draws into its own <article> only, and returns this dataset's bullets for the
// short answer plus a function that redraws its charts. Its URL keys start with "criteo-".

import {
  d3, num, numPlus, pct, int, span, cents, clamp, cap, word, elem, readNumber, readParam, writeParams,
  showTip, hideTip, svgFor, yAxis, xAxis, baseline, marker, dot, valueLabel, crosshair, barPath,
  fillTable, bindSlider,
} from "./lib.js";

const PER = "per 100,000 users";
const KEYS = { budget: "criteo-budget", estimator: "criteo-estimator", measure: "criteo-measure" };
const ESTIMATORS = ["as_randomized", "reweighted"];
const MEASURES = { area: "all budgets", "10pct": "a 10% budget", "30pct": "a 30% budget" };
const SERIES = [
  { key: "chosen", name: "Uplift model", cls: "blue" },
  { key: "random", name: "Random", cls: "gray" },
];
const n0 = (v) => num(v, 0);
const n0p = (v) => numPlus(v, 0);
const n1 = (v) => num(v, 1);
const n1p = (v) => numPlus(v, 1);
const includesZero = ([lo, hi]) => lo <= 0 && hi >= 0;

export function render(root, D) {
  const $ = (name) => root.querySelector(`[data-el="${name}"]`);
  const setText = (name, text) => { $(name).textContent = text; };

  // --- the data -----------------------------------------------------------------------------
  const F = D.fractions;
  const BUDGET_KEYS = Object.keys(D.estimators.as_randomized.budgets); // budgets with paired intervals
  const BUDGETS = BUDGET_KEYS.map((k) => D.estimators.as_randomized.budgets[k].budget);
  const [LEARNER, TARGET] = D.models.chosen.split(" | ");
  const millions = (v) => `${num(v / 1e6, 0)} million`;
  const pairedList = BUDGETS.map((b) => pct(b)).join(", ").replace(/, ([^,]*)$/, " and $1");
  const estName = () => (state.estimator === "reweighted" ? "reweighted" : "as randomized");
  const E = () => D.estimators[state.estimator];

  /** Value and 95% interval of the chosen model and of random targeting at budget index i. */
  function valuesAt(i, e = E()) {
    const f = F[i];
    return {
      chosen: [e.curve.gain[i], e.curve.lo[i], e.curve.hi[i]],
      random: [f * e.full_gain, f * e.full_gain_ci[0], f * e.full_gain_ci[1]],
    };
  }
  /** Paired differences in gain over random, for one measure: area, 10pct or 30pct. */
  function differences(measure, e = E()) {
    const vc = e.differences["best_visits minus best_conversion"][measure];
    const cr = e.differences["best_conversion minus best_response"][measure];
    return [
      { name: "Trained on visits minus trained on conversions", est: vc.estimate, ci: vc.ci },
      // The file holds chosen minus response; the page shows response minus chosen, like the README.
      { name: "Response model minus chosen model", est: -cr.estimate, ci: [-cr.ci[1], -cr.ci[0]] },
    ];
  }

  // Fixed axes across both estimates, so switching moves the marks, not the scale.
  const both = ESTIMATORS.map((k) => D.estimators[k]);
  const CURVE_Y = d3.scaleLinear(
    [d3.min([0, ...both.flatMap((e) => e.curve.lo)]), d3.max(both.flatMap((e) => [...e.curve.hi, e.full_gain_ci[1]]))],
    [0, 1]).nice().domain();
  // Round only the top: the few intervals below zero are small, and rounding the bottom too
  // would leave an empty strip under the bars.
  const DECILE_Y = [d3.min([0, ...both.flatMap((e) => e.deciles.lo)]),
    d3.nice(0, d3.max(both.flatMap((e) => [...e.deciles.hi, e.full_gain])), 5)[1]];
  const DIFF_X = Object.fromEntries(Object.keys(MEASURES).map((k) => {
    const ends = both.flatMap((e) => differences(k, e).flatMap((r) => r.ci));
    const m = d3.max(ends.map(Math.abs));
    return [k, d3.scaleLinear([-m, m], [0, 1]).nice().domain()];
  }));

  // --- state, kept in the page's URL hash under the criteo- keys ----------------------------
  const state = { budget: BUDGETS[0], estimator: "as_randomized", measure: "area" };
  if (readNumber(KEYS.budget) != null) state.budget = clamp(Math.round(readNumber(KEYS.budget)), 1, 100) / 100;
  if (ESTIMATORS.includes(readParam(KEYS.estimator))) state.estimator = readParam(KEYS.estimator);
  if (readParam(KEYS.measure) in MEASURES) state.measure = readParam(KEYS.measure);
  const save = () => writeParams({
    [KEYS.budget]: cents(state.budget), [KEYS.estimator]: state.estimator, [KEYS.measure]: state.measure,
  });

  // --- controls -------------------------------------------------------------------------------
  const budgetSlider = bindSlider($("budget-slider"), {
    min: 1, max: 100, step: 1, format: (v) => `${v}%`, onInput: (v) => setBudget(v / 100),
  });
  function setBudget(b) { state.budget = clamp(cents(b), 1, 100) / 100; renderBudget(); save(); }
  for (const b of BUDGETS) {
    const btn = elem("button", "", `Top ${pct(b)}`);
    btn.type = "button";
    btn.dataset.budget = b;
    btn.addEventListener("click", () => setBudget(b));
    $("budget-chips").append(btn);
  }
  const radios = (name) => root.querySelectorAll(`[data-el="${name}"] input`);
  for (const input of radios("estimator")) {
    input.addEventListener("change", () => { state.estimator = input.value; renderAll(); save(); });
  }
  for (const input of radios("measure")) {
    input.addEventListener("change", () => { state.measure = input.value; renderDifferences(); save(); });
  }

  // --- the short answer and the fixed text ----------------------------------------------------
  const R = D.estimators.as_randomized;
  const W = D.estimators.reweighted;
  const first = BUDGET_KEYS[0];
  const holds = (e) => e.budgets[first].ci[0] > 0
    && includesZero(differences("area", e)[0].ci) && includesZero(differences("area", e)[1].ci);

  function summary() {
    const g = R.budgets[first];
    const [vc, rc] = differences("area", R);
    const items = [
      (g.ci[0] > 0 ? "<strong>Here, targeting clearly beats random.</strong> "
        : "<strong>Here too, targeting isn't confirmed to beat random.</strong> ") +
        `The chosen model's top ${pct(g.budget)} got ${n0(g.vs_random)} more extra conversions ${PER} than a random ` +
        `${pct(g.budget)} (95% CI ${span(...g.ci, n0)}), and this test could detect a gain of ${n0(R.power[first].detectable)}.`,
      (includesZero(vc.ci) ? "<strong>Training on visits didn't help.</strong> " : "<strong>The training outcome mattered.</strong> ") +
        `Models trained on visits and on conversions ranked users ${includesZero(vc.ci) ? "equally well" : "differently"} ` +
        `(difference in area over random ${n1p(vc.est)}, 95% CI ${span(...vc.ci, n1)}), and a response model ` +
        `${includesZero(rc.ci) ? "did as well as" : "differed from"} the chosen uplift model.`,
      "<strong>Treatment is slightly predictable from the features.</strong> Reweighting for it lowers the effect of " +
        `ads from ${n0(R.full_gain)} to ${n0(W.full_gain)} extra conversions ${PER}` +
        `${holds(R) && holds(W) ? ", and every answer above holds either way." : "."}`,
    ];
    return {
      title: `Replication on ${millions(D.users)} users (Criteo)`,
      items: items.map((html) => {
        const li = elem("li");
        li.innerHTML = html; // our own wording and formatted numbers only
        return li;
      }),
    };
  }

  function buildText() {
    setText("users", int(D.users));
    setText("control-share", pct(1 - D.treated_share));
    const c = D.effects.conversion;
    setText("intro",
      `The email experiment's test set was too small to say whether targeting pays. Criteo's is ${int(D.split.test)} ` +
      `held-out users. The data record whether each user visited the advertiser's site (${pct(D.visit_rate, 1)}) and ` +
      `whether they converted (${pct(D.conversion_rate, 2)}), but not revenue, so conversions stand in for spend. ` +
      `Across all users, ads raised conversions by ${pct(c.relative)} (${n0p(c.per_100k)} ${PER}, ` +
      `95% CI ${span(...c.ci_per_100k, n0)}).`);
    const p = D.propensity;
    setText("caveat-text",
      `A classifier recovers treatment slightly: its AUC is ${num(p.auc, 4)}, where independence would give ` +
      `${num(0.5, 4)} ± ${num(p.auc_null_se, 4)}. Most users' estimated chance of treatment lies between ` +
      `${pct(p.p01)} and ${pct(p.p99)}, but it ranges from ${pct(p.min)} to ${pct(p.max)}. Pooling all users ` +
      `overstates the effect of ads: reweighting each user by their chance of treatment lowers it from ` +
      `${n0(R.full_gain)} to ${n0(W.full_gain)} extra conversions ${PER}, and the effect on visits from ` +
      `${n0(R.visit_full_gain)} to ${n0(W.visit_full_gain)}. ` +
      (holds(R) && holds(W)
        ? "The comparisons below barely move, because they pit models against each other on the same users. Switch the estimate to check."
        : "Switch the estimate to compare."));
    setText("budget-lede",
      `The chosen model (${LEARNER}, trained on ${TARGET === "conversion" ? "conversions" : TARGET}) was picked by the ` +
      `email experiment's rule from ${D.candidates} candidates, cross-validated on ${int(D.split.cross_validation)} ` +
      `training users, and saved before the ${int(D.split.test)} test users were scored.`);
    const dc = D.deciles;
    const others = dc.control_conversion_rate.slice(1);
    setText("who-lede",
      `Ranked by the chosen model, the users who respond to ads sit almost entirely in the top tenth. They are also ` +
      `the ones most likely to convert anyway: ${pct(dc.control_conversion_rate[0], 2)} of the top tenth converted ` +
      `without ads, against at most ${pct(d3.max(others), 2)} in any other tenth.`);
    setText("outcome-lede",
      "On the email experiment, models trained on the frequent outcome, visits, ranked customers better, because " +
      `purchases were too rare to learn from. Criteo's training set holds ${int(D.train_conversions)} conversions. ` +
      "These are paired differences between the best model of each kind, on the same held-out users.");
  }

  function buildNotes() {
    const notes = [
      ["The experiment.", `${int(D.users)} users in Criteo's randomized advertising experiments; a random ` +
        `${pct(1 - D.treated_share)} were kept from the advertiser's ads. Criteo altered the rates on purpose to hide ` +
        "the true size of the effects, so absolute effects describe this sample, not real campaigns. Whether a treated " +
        "user actually saw an ad is measured after assignment, so this is the effect of being assigned to ads."],
      ["The protocol.", `The email experiment's: ${int(D.split.train)} training and ${int(D.split.test)} test users, ` +
        `${D.candidates} candidates compared by cross-validation on training users only, and a selection rule fixed ` +
        "and saved before the test set was scored. The test-set stage was later rerun from the saved selection, only " +
        "to add each tenth's treated share and chance of treatment, and every test-set estimate reproduced exactly."],
      ["Intervals.", `All intervals are 95%, from ${int(D.bootstrap_resamples)} Poisson bootstrap resamples shared ` +
        "across models, so each difference between models gets its own paired interval. Reweighted intervals treat " +
        "each user's estimated chance of treatment as fixed."],
      ["Two estimates.", "As randomized compares treated and untreated users directly, as the experiment's design " +
        "intends; the selection rule used it. Reweighted weights each user by their estimated chance of treatment."],
    ];
    $("notes").replaceChildren(...notes.map(([lead, text]) => {
      const li = elem("li");
      li.append(elem("strong", "", lead), ` ${text}`);
      return li;
    }));
  }

  // --- 1. targeting versus random -----------------------------------------------------------------
  const budgetIndex = () => Math.round(state.budget / (F[1] - F[0]));
  const pairedAt = (b) => BUDGET_KEYS.map((k) => E().budgets[k]).find((v) => Math.abs(v.budget - b) < 1e-9);

  function renderBudget() {
    const b = state.budget;
    budgetSlider.set(cents(b), `Top ${pct(b)}`, `Top ${pct(b)} of users`);
    for (const btn of $("budget-chips").querySelectorAll("button")) {
      btn.setAttribute("aria-pressed", String(Math.abs(Number(btn.dataset.budget) - b) < 1e-9));
    }
    for (const tr of $("gain-table").querySelectorAll("tbody tr")) {
      tr.classList.toggle("current", Math.abs(Number(tr.dataset.budget) - b) < 1e-9);
    }
    setText("curve-title", `Showing ads to the top ${pct(b)}, ${estName()}`);
    const v = valuesAt(budgetIndex());
    const paired = pairedAt(b);
    $("readout").replaceChildren(
      ...SERIES.flatMap((s) => [
        elem("span", `key line ${s.cls}`), elem("span", "r-name", s.name),
        elem("span", "r-val", n0(v[s.key][0])), elem("span", "r-ci", `95% CI ${span(v[s.key][1], v[s.key][2], n0)}`),
      ]),
      elem("span"), elem("span", "r-name", "Gain over random"),
      elem("span", "r-val", paired ? n0p(paired.vs_random) : "–"),
      elem("span", "r-ci", paired ? `95% CI ${span(...paired.ci, n0)}, paired` : `paired only at ${pairedList}`),
    );
    drawCurve();
  }

  function drawCurve() {
    const e = E();
    const el = $("curve-chart");
    const { svg, W, H } = svgFor(el, (w) => (w < 440 ? 240 : 270));
    const m = { top: 14, right: 18, bottom: 52, left: 44 };
    const x = d3.scaleLinear([0, 1], [m.left, W - m.right]);
    const y = d3.scaleLinear(CURVE_Y, [H - m.bottom, m.top]);
    yAxis(svg, y, m, W, n0, 5);
    xAxis(svg, x, m, W, H, [0, 0.25, 0.5, 0.75, 1], (v) => pct(v), "Share of users shown ads");
    svg.append("path").attr("class", "band").attr("d", d3.area()
      .x((_, i) => x(F[i])).y0((_, i) => y(e.curve.lo[i])).y1((_, i) => y(e.curve.hi[i]))(F));
    baseline(svg, y, m, W);
    svg.append("path").attr("class", "line gray").attr("d", d3.line()([[x(0), y(0)], [x(1), y(e.full_gain)]]));
    svg.append("path").attr("class", "line blue").attr("d", d3.line().x((_, i) => x(F[i])).y((d) => y(d))(e.curve.gain));
    const i = budgetIndex();
    const v = valuesAt(i);
    marker(svg, x(F[i]), m, H);
    dot(svg, x(F[i]), y(v.random[0]), "gray");
    dot(svg, x(F[i]), y(v.chosen[0]), "blue");
    valueLabel(svg, x(F[i]), y(v.chosen[0]), n0(v.chosen[0]), W, m);
    crosshair(svg, {
      x, xs: F, m, W, H,
      tipFor: (j) => {
        const w = valuesAt(j);
        return [`Top ${pct(F[j])} shown ads`, SERIES.map((s) => ({
          key: s.cls, value: n0(w[s.key][0]), label: `${s.name}, 95% CI ${span(w[s.key][1], w[s.key][2], n0)}`,
        }))];
      },
      pick: (j) => setBudget(Math.max(F[j], F[1])),
    });
    el.replaceChildren(svg.node());
    el.setAttribute("aria-label", `Line chart of extra conversions ${PER} against the share of users shown ads, ` +
      `${estName()}. At the top ${pct(F[i])}: uplift model ` +
      `${n0(v.chosen[0])}, random ${n0(v.random[0])}.`);
  }

  function renderGains() {
    const e = E();
    const t = $("gain-table");
    t.replaceChildren();
    const head = t.createTHead().insertRow();
    ["Budget", "Gain over random", "Share of extra conversions found"].forEach((h) => head.append(elem("th", "", h)));
    const body = t.createTBody();
    for (const key of BUDGET_KEYS) {
      const g = e.budgets[key];
      const tr = body.insertRow();
      tr.dataset.budget = g.budget;
      tr.insertCell().textContent = `Top ${pct(g.budget)}`;
      tr.insertCell().append(elem("b", "", n0p(g.vs_random)), elem("span", "ci-line", span(...g.ci, n0)));
      tr.insertCell().append(elem("b", "", pct(g.share)), elem("span", "ci-line", span(...g.share_ci, (v) => pct(v))));
    }
    setText("gain-sub", `The chosen model minus random targeting at the same budget, ${PER}, with paired 95% ` +
      `intervals, ${estName()}. The share is the part of all the ads' extra conversions found in that top share.`);
    const above = BUDGET_KEYS.filter((k) => e.budgets[k].ci[0] > 0).length;
    const p = e.power[first];
    setText("gain-verdict",
      (above === BUDGET_KEYS.length
        ? `All ${word(above)} intervals are above zero: targeting clearly beats random. `
        : `${cap(word(above))} of the ${word(BUDGET_KEYS.length)} intervals are above zero. `) +
      `At a ${pct(e.budgets[first].budget)} budget, this test could detect a gain of ${n0(p.detectable)} ${PER}; ` +
      `cross-validation had predicted ${n0(p.cv_suggested)}, and the test found ${n0(e.budgets[first].vs_random)}.`);
    for (const tr of t.querySelectorAll("tbody tr")) {
      tr.classList.toggle("current", Math.abs(Number(tr.dataset.budget) - state.budget) < 1e-9);
    }
  }

  function buildCurveTable() {
    const rows = F.slice(1).map((f, j) => {
      const i = j + 1;
      const cells = [`Top ${pct(f)}`];
      for (const k of ESTIMATORS) {
        const v = valuesAt(i, D.estimators[k]);
        cells.push(n0(v.chosen[0]), span(v.chosen[1], v.chosen[2], n0), n0(v.random[0]));
      }
      return cells;
    });
    fillTable($("curve-table"),
      ["Shown ads", "Uplift model", "95% CI", "Random", "Reweighted: uplift model", "95% CI", "Reweighted: random"],
      rows, `Extra conversions ${PER}, held-out users`);
  }

  // --- 2. where the ads work ----------------------------------------------------------------------
  const tenth = (d) => (d === 1 ? "Top 10%" : d === 10 ? "Bottom 10%" : `Tenth ${d}`);

  function renderDeciles() {
    const e = E();
    const u = e.deciles.uplift;
    const rest = d3.max(u.slice(2));
    setText("decile-title", u[0] > d3.sum(u.slice(1))
      ? "The ads work almost entirely through the top tenth"
      : "The effect of the ads is spread across tenths");
    setText("decile-sub",
      `Extra conversions ${PER} from the ads, by tenth of the chosen model's predicted uplift, with 95% intervals, ` +
      `${estName()}. The top tenth gains ${n0(u[0])} ` +
      `(${span(e.deciles.lo[0], e.deciles.hi[0], n0)}), the second ${n0(u[1])} ` +
      `(${span(e.deciles.lo[1], e.deciles.hi[1], n0)}), and no other tenth more than ${n0(rest)}.`);
    drawDeciles();
  }

  function drawDeciles() {
    const e = E();
    const dc = D.deciles;
    const { uplift: vals, lo: los, hi: his } = e.deciles;
    const avg = e.full_gain;
    const el = $("decile-chart");
    const { svg, W, H } = svgFor(el, (w) => (w < 520 ? 260 : 300));
    const m = { top: 26, right: 68, bottom: 64, left: 44 };
    const wide = W - m.left - m.right >= 290;
    if (!wide) m.bottom = 50;
    const x = d3.scaleBand(dc.decile, [m.left, W - m.right]).paddingInner(0.22).paddingOuter(0.1);
    const y = d3.scaleLinear(DECILE_Y, [H - m.bottom, m.top]);
    yAxis(svg, y, m, W, n0, 5);
    const labels = svg.append("g");
    dc.decile.forEach((d) => {
      const t = labels.append("text").attr("class", "axis-text").attr("y", H - m.bottom + 18);
      if (wide) {
        t.attr("text-anchor", "middle");
        (d === 1 ? ["Top", "10%"] : d === 10 ? ["Bottom", "10%"] : [String(d)]).forEach((s, j) => {
          t.append("tspan").attr("x", x(d) + x.bandwidth() / 2).attr("dy", j ? "1.2em" : 0).text(s);
        });
      } else if (d === 1 || d === 10) {
        t.attr("text-anchor", d === 1 ? "start" : "end").attr("x", d === 1 ? x(d) : x(d) + x.bandwidth()).text(tenth(d));
      }
    });
    svg.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
      .attr("x", (m.left + W - m.right) / 2).attr("y", H - 8).text("Users ranked by predicted uplift, in tenths");
    const bw = Math.min(24, x.bandwidth() * 0.72);
    dc.decile.forEach((d, i) => {
      const cx = x(d) + x.bandwidth() / 2;
      svg.append("path").attr("class", `bar ${i === 0 ? "top" : "rest"}`).attr("d", barPath(cx - bw / 2, bw, y(0), y(vals[i])));
    });
    baseline(svg, y, m, W);
    dc.decile.forEach((d, i) => {
      const cx = x(d) + x.bandwidth() / 2;
      svg.append("line").attr("class", "whisker").attr("x1", cx).attr("x2", cx).attr("y1", y(los[i])).attr("y2", y(his[i]));
    });
    svg.append("line").attr("class", "ref-line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(avg)).attr("y2", y(avg));
    const ref = svg.append("text").attr("class", "chart-note").attr("y", y(avg) - 7);
    ref.append("tspan").attr("x", W - m.right + 8).text("Everyone");
    ref.append("tspan").attr("x", W - m.right + 8).attr("dy", "1.25em").text(n0p(avg));
    const topText = n0p(vals[0]);
    const topX = x(1) + x.bandwidth() / 2;
    const fits = topX - topText.length * 3.8 > m.left;
    svg.append("text").attr("class", "value-label").attr("text-anchor", fits ? "middle" : "start")
      .attr("x", fits ? topX : x(1)).attr("y", y(Math.max(his[0], 0)) - 8).text(topText);

    const step = x.step();
    const cols = svg.append("g").selectAll("rect").data(dc.decile).join("rect").attr("class", "col-hit")
      .attr("x", (d) => x(d) - (step - x.bandwidth()) / 2).attr("width", step)
      .attr("y", m.top).attr("height", H - m.top - m.bottom);
    cols.on("pointerenter pointermove", (event, d) => {
      const i = d - 1;
      cols.classed("on", (k) => k === d);
      showTip(event, `${tenth(d)} · ${int(dc.users[i])} users`, [
        { value: n0p(vals[i]), label: `extra conversions ${PER}` },
        { value: span(los[i], his[i], n0), label: "95% interval" },
        { value: pct(dc.treated_share[i], 1), label: "shown ads" },
        { value: pct(dc.control_conversion_rate[i], 2), label: "converted without ads" },
      ]);
    }).on("pointerleave", () => { cols.classed("on", false); hideTip(); });
    el.replaceChildren(svg.node());
    el.setAttribute("aria-label", `Bar chart of extra conversions ${PER} from the ads in each tenth of users ranked ` +
      `by predicted uplift. Top 10%: ${n0p(vals[0])}; everyone: ${n0p(avg)}.`);
  }

  function buildDecileTable() {
    const dc = D.deciles;
    fillTable($("decile-table"),
      ["Tenth", "Users", "Shown ads", "Converted without ads", "Extra conversions", "95% CI", "Reweighted", "95% CI"],
      dc.decile.map((d, i) => [tenth(d), int(dc.users[i]), pct(dc.treated_share[i], 1), pct(dc.control_conversion_rate[i], 2),
        ...ESTIMATORS.flatMap((k) => {
          const q = D.estimators[k].deciles;
          return [n0p(q.uplift[i]), span(q.lo[i], q.hi[i], n0)];
        })]),
      `Extra conversions ${PER}, held-out users`);
  }

  // --- 3. the training outcome ----------------------------------------------------------------------
  function renderDifferences() {
    for (const input of radios("measure")) input.checked = input.value === state.measure;
    const e = E();
    const [vc, rc] = differences(state.measure);
    const where = `${state.measure === "area" ? "Averaged over all budgets" : `At ${MEASURES[state.measure]}`}, ${estName()}`;
    setText("diff-verdict",
      `${where}, ${includesZero(vc.ci) ? "training on visits made no clear difference" : `the model trained on ${vc.est > 0 ? "visits" : "conversions"} did better`}: ` +
      `${n1p(vc.est)} ${PER} (95% CI ${span(...vc.ci, n1)}), where the test could have detected ` +
      `${n1(e.detectable_difference[state.measure])}. ` +
      `${includesZero(rc.ci) ? "The response model matched the chosen model" : `The ${rc.est > 0 ? "response" : "chosen"} model did better`}: ` +
      `${n1p(rc.est)} (95% CI ${span(...rc.ci, n1)}).`);
    drawDifferences();
  }

  function drawDifferences() {
    const rows = differences(state.measure);
    const el = $("diff-chart");
    const { svg, W, H } = svgFor(el, () => 176);
    const m = { top: 8, right: 22, bottom: 52, left: 22 };
    const x = d3.scaleLinear(DIFF_X[state.measure], [m.left, W - m.right]);
    const rowY = (i) => m.top + 34 + i * 54;
    svg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(6)).join("line")
      .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", m.top + 14).attr("y2", H - m.bottom);
    xAxis(svg, x, m, W, H, x.ticks(6), n1p,
      W < 440 ? `Difference, conversions ${PER}` : `Difference in gain over random, conversions ${PER}`);
    svg.append("line").attr("class", "axis-line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.top + 14).attr("y2", H - m.bottom);
    rows.forEach((r, i) => {
      const yy = rowY(i);
      svg.append("text").attr("class", "row-label").attr("x", m.left).attr("y", yy - 14).text(r.name);
      svg.append("line").attr("class", "range blue").attr("x1", x(r.ci[0])).attr("x2", x(r.ci[1])).attr("y1", yy).attr("y2", yy);
      dot(svg, x(r.est), yy, "blue");
      const right = x(r.ci[1]) + 60 < W - m.right;
      svg.append("text").attr("class", "value-label").attr("y", yy + 4)
        .attr("x", right ? x(r.ci[1]) + 8 : x(r.ci[0]) - 8).attr("text-anchor", right ? "start" : "end").text(n1p(r.est));
      svg.append("rect").attr("class", "col-hit").attr("x", m.left).attr("width", W - m.left - m.right)
        .attr("y", yy - 24).attr("height", 40)
        .on("pointerenter pointermove", (event) => showTip(event, r.name, [
          { key: "blue", value: n1p(r.est), label: PER },
          { value: span(...r.ci, n1), label: "95% interval, paired" },
        ]))
        .on("pointerleave", hideTip);
    });
    el.replaceChildren(svg.node());
    el.setAttribute("aria-label", `Dot plot of paired differences in gain over random, ${MEASURES[state.measure]}: ` +
      rows.map((r) => `${r.name} ${n1p(r.est)}, 95% CI ${span(...r.ci, n1)}`).join("; ") + ".");
  }

  function buildModelTable() {
    const labels = {
      best_visits: "Best trained on visits", best_conversion: "Best trained on conversions (chosen)",
      best_response: "Response model",
    };
    const keys = ["10pct", "30pct", "area"];
    const cell = (v, ci) => `${n1p(v)} (${span(...ci, n1)})`;
    const rows = [];
    for (const k of ESTIMATORS) {
      const e = D.estimators[k];
      for (const [role, name] of Object.entries(labels)) {
        const md = e.models[role];
        rows.push([`${name}${k === "reweighted" ? ", reweighted" : ""}`,
          ...keys.map((key) => (key === "area" ? cell(md.area, md.area_ci) : cell(md[key].vs_random, md[key].ci)))]);
      }
      for (const key of [0, 1]) {
        const r = keys.map((mk) => differences(mk, e)[key]);
        rows.push([`${r[0].name}${k === "reweighted" ? ", reweighted" : ""}`, ...r.map((d) => cell(d.est, d.ci))]);
      }
    }
    fillTable($("model-table"), ["Gain over random", "Top 10%", "Top 30%", "All budgets (area)"], rows,
      `Conversions ${PER}, held-out users`);
  }

  // --- go -------------------------------------------------------------------------------------------
  function renderAll() {
    for (const input of radios("estimator")) input.checked = input.value === state.estimator;
    renderGains();
    renderBudget();
    renderDeciles();
    renderDifferences();
  }
  buildText();
  buildNotes();
  buildCurveTable();
  buildDecileTable();
  buildModelTable();
  renderAll();

  return {
    summary: summary(),
    redraw() { drawCurve(); drawDeciles(); drawDifferences(); },
  };
}
