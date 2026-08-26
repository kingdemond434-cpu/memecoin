// Site-wide search over the embedded index. Simple scored substring match.
(function () {
  const index = JSON.parse(document.getElementById("search-data").textContent);
  const box = document.getElementById("search-box");
  const results = document.getElementById("search-results");

  function score(entry, terms) {
    const title = entry.title.toLowerCase();
    const text = (entry.text || "").toLowerCase();
    let s = 0;
    for (const t of terms) {
      if (title === t) s += 100;
      else if (title.includes(t)) s += 40;
      if (text.includes(t)) s += 10;
    }
    return s;
  }

  function render() {
    const terms = box.value.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      results.innerHTML = '<p class="muted">Start typing to search ' + index.length + " entries.</p>";
      return;
    }
    const hits = index
      .map((e) => [score(e, terms), e])
      .filter(([s]) => s > 0)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 30);
    results.innerHTML = hits.length
      ? hits
          .map(
            ([, e]) =>
              '<div class="search-result"><span class="type">' + e.type +
              (e.status ? " · " + e.status : "") + "</span><br>" +
              '<a href="' + e.url + '">' + e.title + "</a>" +
              '<p class="muted">' + (e.text || "").slice(0, 160) + "…</p></div>"
          )
          .join("")
      : '<p class="muted">No matches.</p>';
  }

  box.addEventListener("input", render);
  render();
})();
