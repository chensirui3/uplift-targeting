// Entry point. Each <article class="dataset"> in index.html names its data file (data-src)
// and the module in js/ that draws it (data-dataset). A dataset's article reads only its own
// file, so adding a dataset means one more data file (uplift_targeting/demo.py), one more
// module and one more article; nothing else on the page changes. Each module also returns
// its group of bullets for "The short answer", listed in page order.

import { elem, loadJSON, onResize } from "./js/lib.js";
import * as hillstrom from "./js/hillstrom.js";
import * as criteo from "./js/criteo.js";

const MODULES = { hillstrom, criteo };

if (!window.d3) {
  document.getElementById("page-error").hidden = false;
  throw new Error("d3 didn't load");
}

const articles = [...document.querySelectorAll("article.dataset")];
const loads = articles.map((a) => loadJSON(a.dataset.src).then((data) => ({ data }), (error) => ({ error })));
const summary = document.getElementById("summary");
for (const [i, article] of articles.entries()) {
  const { data, error } = await loads[i];
  const module = MODULES[article.dataset.dataset];
  if (error || !module) {
    article.querySelector('[data-el="error"]').hidden = false;
    console.error(error ?? new Error(`No module for ${article.dataset.dataset}`));
    continue;
  }
  const view = module.render(article, data);
  const list = elem("ol");
  list.append(...view.summary.items);
  const group = elem("div", "summary-group");
  group.append(elem("h3", "", view.summary.title), list);
  summary.append(group);
  onResize(view.redraw);
}
