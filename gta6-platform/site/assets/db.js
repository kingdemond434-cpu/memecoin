// Vehicle database: client-side filter + sort over the embedded JSON.
(function () {
  const data = JSON.parse(document.getElementById("vehicles-data").textContent);
  const tbody = document.querySelector("#veh-table tbody");
  const search = document.getElementById("veh-search");
  const clsSel = document.getElementById("veh-class");
  const statusSel = document.getElementById("veh-status");
  const sortSel = document.getElementById("veh-sort");

  const badge = (s) => {
    const cls = { confirmed: "ok", rumored: "warn", speculated: "spec" }[s] || "spec";
    return '<span class="badge badge-' + cls + '">' + s + "</span>";
  };
  const money = (p) => (p == null ? "?" : "$" + Number(p).toLocaleString());

  function render() {
    const q = (search.value || "").toLowerCase();
    const cls = clsSel.value;
    const status = statusSel.value;
    const sortKey = sortSel.value;

    let rows = data.filter(
      (v) =>
        (!q || (v.name + " " + v.manufacturer).toLowerCase().includes(q)) &&
        (!cls || v.class === cls) &&
        (!status || v.status === status)
    );
    rows.sort((a, b) =>
      sortKey === "name"
        ? a.name.localeCompare(b.name)
        : (b[sortKey] || 0) - (a[sortKey] || 0)
    );

    tbody.innerHTML = rows
      .map(
        (v) =>
          "<tr>" +
          '<td><a href="vehicles/' + v.slug + '.html">' + v.name + "</a></td>" +
          "<td>" + v.manufacturer + "</td>" +
          "<td>" + v.class + "</td>" +
          "<td>" + (v.top_speed || "?") + " mph</td>" +
          "<td><strong>" + (v.overall ?? "?") + "</strong></td>" +
          "<td>" + (v.seats ?? "?") + "</td>" +
          "<td>" + money(v.price) + "</td>" +
          "<td>" + badge(v.status) + "</td>" +
          "</tr>"
      )
      .join("");
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="muted">No vehicles match.</td></tr>';
    }
  }

  [search, clsSel, statusSel, sortSel].forEach((el) =>
    el.addEventListener("input", render)
  );
  render();
})();
