// The email experiment (Kevin Hillstrom's MineThatData challenge, 2008). Every number comes
// from data/hillstrom.json, which `python -m uplift_targeting.demo` writes from results/.
// The only arithmetic here is the profit of emailing everyone at the cost and margin a
// visitor picks, with the formula of analysis.blanket_economics (the export checks it
// against summary.json). No model runs in the browser.
//
// render() draws into its own <article> only, and returns this dataset's bullets for the
// short answer at the top of the page plus a function that redraws its charts.

import {
  d3, usd, usdPlus, num, numPlus, pct, int, about, span, cents, clamp, cap, word, elem,
  readNumber, readParam, writeParams, showTip, hideTip, svgFor, yAxis, xAxis, baseline,
  marker, dot, valueLabel, crosshair, barPath, fillTable, bindSlider,
} from "./lib.js";

const MARGIN_RANGE = [10, 60]; // percent; the analysis assumed one value inside it
const LEAD_ROW = "bought men's and women's"; // the profile row the text names
const SERIES = [
  { key: "chosen", name: "Uplift model", cls: "blue" },
  { key: "best_response", name: "Response model", cls: "orange" },
  { key: "random", name: "Random", cls: "gray" },
];

export function render(root, D) {
  const $ = (name) => root.querySelector(`[data-el="${name}"]`);
  const setText = (name, text) => { $(name).textContent = text; };

  // --- the data -----------------------------------------------------------------------------
  const EXP = D.experiment;
  const MENS = EXP.mens;
  const SWEEP = D.cost_sweep;
  const COSTS = SWEEP.cost_per_email;
  const CURVES = D.curves;
  const FRACTIONS = CURVES.spend.fraction_emailed;
  const BUDGET_KEYS = Object.keys(CURVES.spend.vs_random); // budgets with paired intervals
  const BUDGETS = BUDGET_KEYS.map((k) => CURVES.spend.vs_random[k].budget);
  const [CHOSEN_LEARNER, CHOSEN_TARGET] = D.models.chosen.split(" | ");
  const OUTCOMES = {
    spend: { f: (v) => usd(v), fPlus: (v) => usdPlus(v), axis: (v) => usd(v), title: "Extra revenue", noun: "extra revenue" },
    visits: { f: (v) => num(v, 1), fPlus: (v) => numPlus(v, 1), axis: (v) => num(v, 0), title: "Extra visitors", noun: "extra visitors" },
  };
  const DECILE = {
    visit: {
      scale: 100, v: "observed_visit_uplift", lo: "visit_lo", hi: "visit_hi",
      f: (v) => numPlus(v, 1), fr: (v) => num(v, 1), axis: (v) => num(v, 0), unit: "extra visitors per 100 customers",
    },
    spend: {
      scale: 1, v: "observed_spend_uplift", lo: "spend_lo", hi: "spend_hi",
      f: (v) => usdPlus(v, 2), fr: (v) => usd(v, 2), axis: (v) => usd(v, 0), unit: "extra revenue per customer",
    },
  };

  /** Profit per 1,000 customers of emailing everyone the men's email (analysis.blanket_economics). */
  function everyone(margin, cost) {
    const profit = (effect) => (margin * effect - cost) * 1000;
    return { est: profit(MENS.spend_effect), lo: profit(MENS.spend_ci[0]), hi: profit(MENS.spend_ci[1]) };
  }
  const breakeven = (margin) => [MENS.spend_effect, ...MENS.spend_ci].map((v) => margin * v);

  /** Value and 95% interval of each targeting method when the top share `i` is emailed. */
  function valuesAt(k, i) {
    const c = CURVES[k];
    const f = c.fraction_emailed[i];
    return {
      chosen: [c.chosen[i], c.chosen_lo[i], c.chosen_hi[i]],
      best_response: [c.best_response[i], c.best_response_lo[i], c.best_response_hi[i]],
      random: [f * c.full_gain, f * c.full_gain_ci[0], f * c.full_gain_ci[1]],
    };
  }

  // Fixed axes, so moving a slider moves the marks, not the scale.
  const PROFIT_Y = (() => {
    const ends = [];
    for (const m of MARGIN_RANGE) {
      for (const c of [COSTS[0], COSTS.at(-1)]) {
        const p = everyone(m / 100, c);
        ends.push(p.lo, p.hi);
      }
    }
    return d3.scaleLinear(d3.extent(ends), [0, 1]).nice().domain();
  })();
  const CURVE_Y = Object.fromEntries(Object.keys(OUTCOMES).map((k) => {
    const c = CURVES[k];
    const lo = d3.min([0, ...c.chosen_lo, ...c.best_response]);
    const hi = d3.max([c.full_gain, ...c.chosen_hi, ...c.best_response]);
    return [k, d3.scaleLinear([lo, hi], [0, 1]).nice().domain()];
  }));

  // --- state, kept in the page's URL hash under these keys -----------------------------------
  const state = {
    cost: D.economics.cost_per_email,
    margin: D.economics.margin,
    budget: BUDGETS[0],
    deciles: "visit",
  };
  if (readNumber("cost") != null) state.cost = clamp(cents(readNumber("cost")), cents(COSTS[0]), cents(COSTS.at(-1))) / 100;
  if (readNumber("margin") != null) state.margin = clamp(Math.round(readNumber("margin")), ...MARGIN_RANGE) / 100;
  if (readNumber("budget") != null) state.budget = clamp(Math.round(readNumber("budget")), 1, 100) / 100;
  if (["visit", "spend"].includes(readParam("deciles"))) state.deciles = readParam("deciles");
  const save = () => writeParams({
    cost: state.cost.toFixed(2), margin: cents(state.margin), budget: cents(state.budget), deciles: state.deciles,
  });

  // --- controls -------------------------------------------------------------------------------
  const costSlider = bindSlider($("cost-slider"), {
    min: cents(COSTS[0]), max: cents(COSTS.at(-1)), step: cents(COSTS[1] - COSTS[0]),
    format: (v) => usd(v / 100, 2), onInput: (v) => setCost(v / 100),
  });
  const marginSlider = bindSlider($("margin-slider"), {
    min: MARGIN_RANGE[0], max: MARGIN_RANGE[1], step: 1, format: (v) => `${v}%`, onInput: (v) => setMargin(v / 100),
  });
  const budgetSlider = bindSlider($("budget-slider"), {
    min: 1, max: 100, step: 1, format: (v) => `${v}%`, onInput: (v) => setBudget(v / 100),
  });
  function setCost(c) { state.cost = c; renderEveryone(); save(); }
  function setMargin(m) { state.margin = m; renderEveryone(); save(); }
  function setBudget(b) { state.budget = clamp(cents(b), 1, 100) / 100; renderBudget(); save(); }

  for (const b of BUDGETS) {
    const btn = elem("button", "", `Top ${pct(b)}`);
    btn.type = "button";
    btn.dataset.budget = b;
    btn.addEventListener("click", () => setBudget(b));
    $("budget-chips").append(btn);
  }
  const decileInputs = root.querySelectorAll('[data-el="decile-outcome"] input');
  for (const input of decileInputs) {
    input.addEventListener("change", () => { state.deciles = input.value; renderDeciles(); save(); });
  }

  // --- the short answer and the fixed text ----------------------------------------------------
  function summary() {
    const m0 = D.economics.margin;
    const [be] = breakeven(m0);
    const ratio = D.deciles.observed_visit_uplift[0] / D.everyone.visit.effect;
    const first = BUDGET_KEYS[0];
    const visits1 = CURVES.visits.vs_random[first];
    const spend1 = CURVES.spend.vs_random[first];
    const spendClear = BUDGET_KEYS.filter((k) => CURVES.spend.vs_random[k].ci[0] > 0);
    const dollars2 = (v) => usd(v, 2);
    const items = [
      `<strong>Email everyone the men's email</strong> while an email costs less than about ${usd(be, 2)}, at a ${pct(m0)} margin. ` +
        `It raised spend by ${usd(MENS.spend_effect, 2)} per customer (95% CI ${span(...MENS.spend_ci, dollars2)}).`,
      (visits1.ci[0] > 0
        ? `<strong>In visits, the model finds a persuadable ${pct(visits1.budget)}.</strong> `
        : `<strong>In visits, the model's top ${pct(visits1.budget)} isn't confirmed to beat random targeting.</strong> `) +
        `Among held-out customers in its top tenth, the email's effect on visits was ${num(ratio, 1)} times the average ` +
        `(${numPlus(D.deciles.observed_visit_uplift[0] * 100)} vs ${numPlus(D.everyone.visit.effect * 100)} extra visitors per 100).`,
      (spendClear.length === 0
        ? "<strong>In dollars, targeting isn't confirmed.</strong> "
        : "<strong>In dollars, targeting beat random at some budgets.</strong> ") +
        `At a ${pct(spend1.budget)} budget, the model's gain over random targeting was ${usdPlus(spend1.per_1000)} per 1,000 customers, ` +
        `but the 95% interval runs from ${usd(spend1.ci[0])} to ${usd(spend1.ci[1])}. ` +
        `This test can only reliably detect gains above ${usd(D.power[first].detectable_per_1000)}.`,
    ];
    return {
      title: `The email experiment (${int(EXP.customers)} customers)`,
      items: items.map((html) => {
        const li = elem("li");
        li.innerHTML = html; // our own wording and formatted numbers only
        return li;
      }),
    };
  }

  function buildText() {
    const dollars2 = (v) => usd(v, 2);
    setText("customers", int(EXP.customers));
    setText("everyone-lede",
      `Across all ${int(EXP.customers)} customers, the men's email raised spend by ${usd(MENS.spend_effect, 2)} per customer ` +
      `(95% CI ${span(...MENS.spend_ci, dollars2)}), and the women's email by ${usd(EXP.womens.spend_effect, 2)}. ` +
      "Whether emailing everyone pays depends on two assumptions: what an email really costs, counting unsubscribes and " +
      "fatigue, and how much of each extra dollar of revenue is profit. Set both.");
    setText("budget-lede",
      `The uplift model ranks customers by how much the email should change their behavior. It's the ${CHOSEN_LEARNER} ` +
      `trained on ${CHOSEN_TARGET}, picked by cross-validation on ${int(D.split.train)} customers with a rule written down ` +
      `before testing, then evaluated once on ${int(D.split.test)} held-out customers it never saw. The response model ` +
      "ranks customers the usual way, by how much they're predicted to spend if emailed. Both have to beat random targeting.");
    setText("who-lede",
      "Split the held-out customers into tenths by the model's predicted uplift. If the ranking works, the email's " +
      "effect should be largest at the top.");
  }

  function buildGainTable() {
    const t = $("gain-table");
    const head = t.createTHead().insertRow();
    ["Budget", "Extra revenue", "Extra visitors"].forEach((h) => head.append(elem("th", "", h)));
    const body = t.createTBody();
    for (const key of BUDGET_KEYS) {
      const tr = body.insertRow();
      tr.dataset.budget = CURVES.spend.vs_random[key].budget;
      tr.insertCell().textContent = `Top ${pct(CURVES.spend.vs_random[key].budget)}`;
      for (const k of ["spend", "visits"]) {
        const v = CURVES[k].vs_random[key];
        tr.insertCell().append(elem("b", "", OUTCOMES[k].fPlus(v.per_1000)), elem("span", "ci-line", span(...v.ci, OUTCOMES[k].f)));
      }
    }
    const cells = ["spend", "visits"].flatMap((k) => BUDGET_KEYS.map((key) => ({ k, ...CURVES[k].vs_random[key] })));
    const clear = cells.filter((c) => c.ci[0] > 0 || c.ci[1] < 0);
    let text = clear.length === 0
      ? `None of the ${word(cells.length)} intervals excludes zero.`
      : `${clear.length === 1 ? "Only one" : cap(word(clear.length))} of the ${word(cells.length)} intervals ` +
        `exclude${clear.length === 1 ? "s" : ""} zero: ` +
        clear.map((c) => `${OUTCOMES[c.k].noun} at the top ${pct(c.budget)} (${c.ci[0] > 0 ? "above" : "below"} zero)`).join(", ") + ".";
    if (!clear.some((c) => c.k === "spend")) {
      text += " In revenue, every interval includes zero, so the gain from targeting is not confirmed.";
    }
    setText("gain-verdict", text);
  }

  function buildPower() {
    const [a, b] = Object.entries(D.power).map(([key, v]) => ({ budget: parseInt(key, 10) / 100, ...v }));
    setText("power-text",
      `Purchases are rare (${pct(EXP.purchase_rate, 1)} of customers) and large (${usd(EXP.average_order)} on average), ` +
      "so revenue per customer is almost all zeros with occasional big spikes. " +
      `At a ${pct(a.budget)} budget, this test can only reliably detect a revenue gain over random above ` +
      `${usd(a.detectable_per_1000)} per 1,000 customers; cross-validation suggested ${usd(a.cv_suggested_per_1000)}. ` +
      `Confirming a gain that size would take about ${about(a.customers_needed)} test customers, ` +
      `${num(a.times_test_set)} times the ${int(D.split.test)} here. At a ${pct(b.budget)} budget it would take ` +
      `about ${about(b.customers_needed)} (${num(b.times_test_set)} times).`);
  }

  function buildProfile() {
    const { columns, rows } = D.profile;
    const t = $("profile-table");
    const head = t.createTHead().insertRow();
    head.append(elem("th"));
    columns.forEach((c, j) => {
      const [main, sub] = c.split(" (");
      const th = elem("th");
      if (j === 0) th.append(elem("span", "th-key"));
      th.append(main);
      if (sub) th.append(elem("span", "th-sub", sub.replace(/\)$/, "")));
      head.append(th);
    });
    const body = t.createTBody();
    for (const [label, ...vals] of rows) {
      const tr = body.insertRow();
      if (label === LEAD_ROW) tr.className = "lead";
      const money = label.includes("(median $)");
      const median = label.includes("(median");
      tr.insertCell().textContent = cap(label.replace(" (median $)", " (median)"));
      for (const v of vals) {
        tr.insertCell().textContent = label === "customers" ? int(v)
          : money ? usd(v) : median ? num(v, Number.isInteger(v) ? 0 : 1) : pct(v);
      }
    }
    const lead = rows.find((r) => r[0] === LEAD_ROW);
    setText("profile-sub",
      `Held-out customers in the model's top and bottom tenths, against all of them. ${pct(lead[1])} of the top 10% ` +
      `bought both men's and women's items last year, against ${pct(lead[2])} of everyone.`);
  }

  function buildNotes() {
    const notes = [
      ["The experiment.", `${int(EXP.customers)} recent customers were split at random: ${int(MENS.n)} got the men's email, ` +
        `${int(EXP.womens.n)} the women's email and ${int(EXP.control.n)} no email. Visits, purchases and spend were ` +
        "recorded for the next two weeks. Because the split was random, the difference between groups is the email's effect."],
      ["Held-out test.", `Models were trained and compared by cross-validation on ${int(D.split.train)} customers only. ` +
        `The chosen one was evaluated once on the other ${int(D.split.test)}. The second and third parts above use only ` +
        `those held-out customers; the first uses all ${int(EXP.customers)}, since no model is involved.`],
      ["Intervals.", "All intervals are 95%. An email's effect is a difference in means with a normal-approximation " +
        "interval. The targeting curves use bootstrap resamples drawn within each email group and shared across " +
        "methods, so each gain over random targeting gets its own paired interval."],
      ["Profit.", "Profit per 1,000 customers = 1,000 × (margin × extra spend per customer − cost per email). " +
        `The margin and the cost are assumptions; the analysis used ${pct(D.economics.margin)} and ` +
        `${usd(D.economics.cost_per_email, 2)}.`],
    ];
    $("notes").replaceChildren(...notes.map(([lead, text]) => {
      const li = elem("li");
      li.append(elem("strong", "", lead), ` ${text}`);
      return li;
    }));
  }

  // --- 1. emailing everyone -----------------------------------------------------------------------
  function renderEveryone() {
    const { cost, margin } = state;
    costSlider.set(cents(cost), usd(cost, 2), `${usd(cost, 2)} per email`);
    marginSlider.set(cents(margin), pct(margin), `${pct(margin)} margin`);
    const p = everyone(margin, cost);
    const [be, beLo, beHi] = breakeven(margin);
    setText("profit-hero", usd(p.est));
    setText("profit-ci", `95% CI ${span(p.lo, p.hi, usd)}, at ${usd(cost, 2)} an email and a ${pct(margin)} margin`);
    const verdict = p.lo > 0 ? "Emailing everyone pays: the whole 95% interval is a profit."
      : p.est > 0 ? "The estimate is a profit, but the 95% interval also includes a loss."
      : p.hi > 0 ? "The estimate is a loss, but the 95% interval also includes a profit."
      : "Emailing everyone loses money: the whole 95% interval is a loss.";
    setText("profit-verdict", `${verdict} At this margin, emailing everyone pays while an email costs less than ` +
      `${usd(be, 2)} (95% CI ${span(beLo, beHi, (v) => usd(v, 2))}).`);
    setText("profit-title", `Profit at every cost, at a ${pct(margin)} margin`);
    drawProfit();
    fillTable($("profit-table"), ["Cost per email", "Profit per 1,000", "95% CI"],
      COSTS.map((c) => {
        const q = everyone(margin, c);
        return [usd(c, 2), usd(q.est), span(q.lo, q.hi, usd)];
      }), `At a ${pct(margin)} margin`);
    renderSweep();
  }

  function drawProfit() {
    const el = $("profit-chart");
    const { svg, W, H } = svgFor(el, (w) => (w < 520 ? 250 : 290));
    const m = { top: 14, right: 22, bottom: 52, left: 56 };
    const x = d3.scaleLinear([COSTS[0], COSTS.at(-1)], [m.left, W - m.right]);
    const y = d3.scaleLinear(PROFIT_Y, [H - m.bottom, m.top]);
    yAxis(svg, y, m, W, (v) => usd(v), H < 270 ? 5 : 6);
    xAxis(svg, x, m, W, H, x.ticks(5), (v) => usd(v, 2), "Cost per email");
    const pts = COSTS.map((c) => ({ c, ...everyone(state.margin, c) }));
    svg.append("path").attr("class", "band")
      .attr("d", d3.area().x((d) => x(d.c)).y0((d) => y(d.lo)).y1((d) => y(d.hi))(pts));
    baseline(svg, y, m, W);
    svg.append("path").attr("class", "line blue")
      .attr("d", d3.line().x((d) => x(d.c)).y((d) => y(d.est))(pts));

    const [be] = breakeven(state.margin);
    const p = everyone(state.margin, state.cost);
    const showBe = be > COSTS[0] && be < COSTS.at(-1);
    if (showBe) {
      dot(svg, x(be), y(0), "ink", 4);
      // Skip the words when the selected point's label would sit on top of them.
      if (Math.abs(x(be) - x(state.cost)) > 64 || Math.abs(y(p.est) - y(0)) > 24) {
        svg.append("text").attr("class", "chart-note").attr("x", x(be) + 8).attr("y", y(0) - 9)
          .text(`Break-even ${usd(be, 2)}`);
      }
    }
    marker(svg, x(state.cost), m, H);
    dot(svg, x(state.cost), y(p.est), "blue");
    valueLabel(svg, x(state.cost), y(p.est), usd(p.est), W, m);
    crosshair(svg, {
      x, xs: COSTS, m, W, H,
      tipFor: (i) => [`At ${usd(pts[i].c, 2)} an email`, [
        { key: "blue", value: usd(pts[i].est), label: "profit per 1,000 customers" },
        { value: span(pts[i].lo, pts[i].hi, usd), label: "95% interval" },
      ]],
      pick: (i) => setCost(COSTS[i]),
    });
    el.replaceChildren(svg.node());
    el.setAttribute("aria-label", "Line chart of profit per 1,000 customers from emailing everyone against the cost " +
      `per email, at a ${pct(state.margin)} margin. ${showBe ? `It crosses zero at ${usd(be, 2)}. ` : ""}` +
      `At ${usd(state.cost, 2)}: ${usd(p.est)}, 95% CI ${span(p.lo, p.hi, usd)}.`);
  }

  function renderSweep() {
    const i = COSTS.findIndex((c) => Math.abs(c - state.cost) < 1e-9);
    const m0 = D.economics.margin;
    const share = SWEEP.share_emailed[i];
    const diff = SWEEP.targeted_minus_best_simple_per_1000[i];
    const clear = COSTS.filter((_, k) => SWEEP.diff_lo[k] > 0 || SWEEP.diff_hi[k] < 0);
    const costs = `costs from ${usd(COSTS[0], 2)} to ${usd(COSTS.at(-1), 2)}`;
    let text;
    if (share >= 1) {
      text = `At ${usd(state.cost, 2)} an email, the model would email every customer too, so it can't do better ` +
        "than emailing everyone.";
    } else {
      const change = Math.abs(diff) < 0.5 ? "about the same"
        : `${usd(Math.abs(diff))} ${diff < 0 ? "less" : "more"} per 1,000 customers`;
      text = `At ${usd(state.cost, 2)} an email, the model would skip the ${pct(1 - share)} of customers it ranks lowest, ` +
        `a cutoff chosen on the training set. On the ${int(D.split.test)} held-out customers, that earned ${change} than ` +
        `the better of emailing everyone or no one (95% CI for the difference: ` +
        `${span(SWEEP.diff_lo[i], SWEEP.diff_hi[i], usd)}).`;
    }
    text += clear.length === 0
      ? ` Across ${costs}, letting the model skip its lowest-ranked customers never clearly beats the better of ` +
        "emailing everyone or no one: every 95% interval for the difference includes zero."
      : ` Across ${costs}, the 95% interval for the difference excludes zero at ${word(clear.length)} of ` +
        `${COSTS.length}.`;
    text += ` This comparison uses the analysis's ${pct(m0)} margin` +
      `${Math.abs(state.margin - m0) > 1e-9 ? ", not the one set above" : ""}.`;
    setText("sweep-text", text);
  }

  function buildSweepTable() {
    fillTable($("sweep-table"),
      ["Cost per email", "Model emails", "Difference per 1,000", "95% CI"],
      COSTS.map((c, k) => [usd(c, 2), pct(SWEEP.share_emailed[k]),
        usdPlus(SWEEP.targeted_minus_best_simple_per_1000[k]), span(SWEEP.diff_lo[k], SWEEP.diff_hi[k], usd)]),
      `Targeting minus the better of emailing everyone or no one, held-out customers, ${pct(D.economics.margin)} margin`);
  }

  // --- 2. emailing only some ----------------------------------------------------------------------
  const budgetIndex = () => Math.round(state.budget / (FRACTIONS[1] - FRACTIONS[0]));

  function renderBudget() {
    const b = state.budget;
    budgetSlider.set(cents(b), `Top ${pct(b)}`, `Top ${pct(b)} of customers`);
    for (const btn of $("budget-chips").querySelectorAll("button")) {
      btn.setAttribute("aria-pressed", String(Math.abs(Number(btn.dataset.budget) - b) < 1e-9));
    }
    for (const tr of $("gain-table").querySelectorAll("tbody tr")) {
      tr.classList.toggle("current", Math.abs(Number(tr.dataset.budget) - b) < 1e-9);
    }
    for (const k of Object.keys(OUTCOMES)) {
      setText(`${k}-title`, `${OUTCOMES[k].title} from emailing the top ${pct(b)}`);
      renderReadout(k);
      drawCurve(k);
    }
  }

  function renderReadout(k) {
    const v = valuesAt(k, budgetIndex());
    const f = OUTCOMES[k].f;
    $(`${k}-readout`).replaceChildren(...SERIES.flatMap((s) => [
      elem("span", `key line ${s.cls}`),
      elem("span", "r-name", s.name),
      elem("span", "r-val", f(v[s.key][0])),
      elem("span", "r-ci", `95% CI ${span(v[s.key][1], v[s.key][2], f)}`),
    ]));
  }

  function drawCurve(k) {
    const c = CURVES[k];
    const { f } = OUTCOMES[k];
    const el = $(`${k}-chart`);
    const { svg, W, H } = svgFor(el, (w) => (w < 440 ? 240 : 270));
    const m = { top: 14, right: 18, bottom: 52, left: k === "spend" ? 58 : 40 };
    const x = d3.scaleLinear([0, 1], [m.left, W - m.right]);
    const y = d3.scaleLinear(CURVE_Y[k], [H - m.bottom, m.top]);
    const xs = c.fraction_emailed;
    yAxis(svg, y, m, W, OUTCOMES[k].axis, 5);
    xAxis(svg, x, m, W, H, [0, 0.25, 0.5, 0.75, 1], (v) => pct(v), "Share of customers emailed");
    svg.append("path").attr("class", "band").attr("d", d3.area()
      .x((_, i) => x(xs[i])).y0((_, i) => y(c.chosen_lo[i])).y1((_, i) => y(c.chosen_hi[i]))(xs));
    baseline(svg, y, m, W);
    const line = d3.line().x((_, i) => x(xs[i])).y((d) => y(d));
    svg.append("path").attr("class", "line gray").attr("d", d3.line()([[x(0), y(0)], [x(1), y(c.full_gain)]]));
    svg.append("path").attr("class", "line orange").attr("d", line(c.best_response));
    svg.append("path").attr("class", "line blue").attr("d", line(c.chosen));

    const i = budgetIndex();
    const v = valuesAt(k, i);
    marker(svg, x(xs[i]), m, H);
    for (const s of [...SERIES].reverse()) dot(svg, x(xs[i]), y(v[s.key][0]), s.cls);
    valueLabel(svg, x(xs[i]), y(v.chosen[0]), f(v.chosen[0]), W, m);
    crosshair(svg, {
      x, xs, m, W, H,
      tipFor: (j) => {
        const w = valuesAt(k, j);
        return [`Top ${pct(xs[j])} emailed`, SERIES.map((s) => ({
          key: s.cls, value: f(w[s.key][0]), label: `${s.name}, 95% CI ${span(w[s.key][1], w[s.key][2], f)}`,
        }))];
      },
      pick: (j) => setBudget(Math.max(xs[j], xs[1])),
    });
    el.replaceChildren(svg.node());
    el.setAttribute("aria-label", `Line chart of ${OUTCOMES[k].noun} per 1,000 customers against the share of ` +
      `customers emailed, for the uplift model, the response model and random targeting. At the top ${pct(xs[i])}: ` +
      SERIES.map((s) => `${s.name} ${f(v[s.key][0])}`).join(", ") + ".");
  }

  function buildCurveTables() {
    for (const k of Object.keys(OUTCOMES)) {
      const { f } = OUTCOMES[k];
      fillTable($(`${k}-table`),
        ["Emailed", "Uplift model", "95% CI", "Response model", "95% CI", "Random", "95% CI"],
        CURVES[k].fraction_emailed.slice(1).map((fr, j) => {
          const v = valuesAt(k, j + 1);
          return [`Top ${pct(fr)}`, ...SERIES.flatMap((s) => [f(v[s.key][0]), span(v[s.key][1], v[s.key][2], f)])];
        }), `${OUTCOMES[k].title} per 1,000 held-out customers`);
    }
  }

  // --- 3. who the persuadable customers are --------------------------------------------------------
  const tenth = (d) => (d === 1 ? "Top 10%" : d === 10 ? "Bottom 10%" : `Tenth ${d}`);

  function renderDeciles() {
    const visit = state.deciles === "visit";
    for (const input of decileInputs) input.checked = input.value === state.deciles;
    const dc = D.deciles;
    const ratio = dc.observed_visit_uplift[0] / D.everyone.visit.effect;
    const throughZero = dc.spend_lo.filter((lo, i) => lo <= 0 && dc.spend_hi[i] >= 0).length;
    setText("decile-title", visit
      ? `The model's top 10% respond ${num(ratio, 1)} times as much as average in visits`
      : "In revenue, the same tenths are too noisy to rank");
    setText("decile-sub", visit
      ? "Extra visitors per 100 customers from the men's email, by tenth of the model's predicted uplift, with 95% " +
        "intervals. Held-out customers."
      : "Extra revenue per customer from the men's email, by tenth of predicted uplift, with 95% intervals. " +
        `${cap(word(throughZero))} of the ${dc.decile.length} intervals include zero.`);
    drawDeciles();
  }

  function drawDeciles() {
    const o = DECILE[state.deciles];
    const dc = D.deciles;
    const vals = dc[o.v].map((v) => v * o.scale);
    const los = dc[o.lo].map((v) => v * o.scale);
    const his = dc[o.hi].map((v) => v * o.scale);
    const avg = D.everyone[state.deciles].effect * o.scale;
    const el = $("decile-chart");
    const { svg, W, H } = svgFor(el, (w) => (w < 520 ? 260 : 300));
    const m = { top: 26, right: 68, bottom: 64, left: state.deciles === "visit" ? 36 : 44 };
    // Room for a label under every bar ("Bottom" needs about 29px a tenth)? Otherwise name only the ends.
    const wide = W - m.left - m.right >= 290;
    if (!wide) m.bottom = 50;
    const x = d3.scaleBand(dc.decile, [m.left, W - m.right]).paddingInner(0.22).paddingOuter(0.1);
    const y = d3.scaleLinear([Math.min(0, d3.min(los)), Math.max(avg, d3.max(his))], [H - m.bottom, m.top]).nice();
    yAxis(svg, y, m, W, o.axis, 5);
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
      .attr("x", (m.left + W - m.right) / 2).attr("y", H - 8).text("Customers ranked by predicted uplift, in tenths");

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
    ref.append("tspan").attr("x", W - m.right + 8).attr("dy", "1.25em").text(o.f(avg));
    // The top tenth's value above its interval, kept clear of the y-axis labels.
    const topText = o.f(vals[0]);
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
      cols.classed("on", (e) => e === d);
      showTip(event, tenth(d), [
        { value: o.f(vals[i]), label: o.unit },
        { value: span(los[i], his[i], o.fr), label: "95% interval" },
        { value: int(dc.customers[i]), label: "customers" },
      ]);
    }).on("pointerleave", () => { cols.classed("on", false); hideTip(); });
    el.replaceChildren(svg.node());
    el.setAttribute("aria-label", `Bar chart of ${o.unit} from the men's email in each tenth of customers ranked by ` +
      `predicted uplift. Top 10%: ${o.f(vals[0])}; everyone: ${o.f(avg)}.`);
  }

  function buildDecileTable() {
    const dc = D.deciles;
    const v = DECILE.visit;
    const s = DECILE.spend;
    fillTable($("decile-table"),
      ["Tenth", "Customers", "Extra visitors per 100", "95% CI", "Extra revenue per customer", "95% CI"],
      dc.decile.map((d, i) => [tenth(d), int(dc.customers[i]),
        v.f(dc[v.v][i] * 100), span(dc[v.lo][i] * 100, dc[v.hi][i] * 100, v.fr),
        s.f(dc[s.v][i]), span(dc[s.lo][i], dc[s.hi][i], s.fr)]),
      "Men's email versus no email, held-out customers");
  }

  // --- go -------------------------------------------------------------------------------------------
  buildText();
  buildGainTable();
  buildPower();
  buildProfile();
  buildNotes();
  buildSweepTable();
  buildCurveTables();
  buildDecileTable();
  renderEveryone();
  renderBudget();
  renderDeciles();

  return {
    summary: summary(),
    redraw() { drawProfit(); drawCurve("spend"); drawCurve("visits"); drawDeciles(); },
  };
}
