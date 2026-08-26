// Personal tracker: checklists over every trackable entity, progress bars,
// localStorage persistence, JSON export/import, optional server sync
// (serve.py exposes /api/progress).
(function () {
  const data = JSON.parse(document.getElementById("tracker-data").textContent);
  const root = document.getElementById("tracker-root");
  const msg = document.getElementById("trk-msg");
  const SECTIONS = [
    ["vehicles", "Vehicles spotted"],
    ["missions", "Missions completed"],
    ["locations", "Locations visited / collected"],
  ];

  const key = (type, slug) => "trk:" + type + ":" + slug;
  const get = (type, slug) => {
    try { return localStorage.getItem(key(type, slug)) === "1"; } catch (e) { return false; }
  };
  const set = (type, slug, val) => {
    try { localStorage.setItem(key(type, slug), val ? "1" : "0"); } catch (e) {}
  };

  function render() {
    root.innerHTML = SECTIONS.map(([type, title]) => {
      const items = data[type] || [];
      const done = items.filter((i) => get(type, i.slug)).length;
      const pct = items.length ? Math.round((done / items.length) * 100) : 0;
      const rows = items
        .map(
          (i) =>
            '<li class="' + (get(type, i.slug) ? "done" : "") + '">' +
            '<input type="checkbox" data-type="' + type + '" data-slug="' + i.slug + '"' +
            (get(type, i.slug) ? " checked" : "") + ">" +
            '<a class="trk-name" href="' + i.url + '">' + i.name + "</a>" +
            '<span class="muted">' + i.detail + "</span></li>"
        )
        .join("");
      return (
        '<section class="tracker-section"><h2>' + title + "</h2>" +
        '<p class="muted">' + done + " / " + items.length + " (" + pct + "%)</p>" +
        '<div class="progress-track"><span class="progress-fill" style="width:' + pct + '%"></span></div>' +
        '<ul class="tracker-list">' + rows + "</ul></section>"
      );
    }).join("");

    root.querySelectorAll("input[type=checkbox]").forEach((cb) =>
      cb.addEventListener("change", () => {
        set(cb.dataset.type, cb.dataset.slug, cb.checked);
        render();
      })
    );
  }

  function snapshot() {
    const out = {};
    SECTIONS.forEach(([type]) =>
      (data[type] || []).forEach((i) => {
        if (get(type, i.slug)) out[key(type, i.slug)] = 1;
      })
    );
    return out;
  }

  function restore(snap) {
    SECTIONS.forEach(([type]) =>
      (data[type] || []).forEach((i) => set(type, i.slug, !!snap[key(type, i.slug)]))
    );
    render();
  }

  document.getElementById("trk-export").onclick = () => {
    prompt("Copy your progress token:", JSON.stringify(snapshot()));
  };
  document.getElementById("trk-import").onclick = () => {
    const raw = prompt("Paste a progress token:");
    if (!raw) return;
    try { restore(JSON.parse(raw)); msg.textContent = "Progress imported."; }
    catch (e) { msg.textContent = "That token didn't parse."; }
  };
  document.getElementById("trk-sync-up").onclick = async () => {
    try {
      const res = await fetch("/api/progress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ progress: snapshot() }),
      });
      const body = await res.json();
      msg.textContent = "Synced. Your account token: " + body.token;
    } catch (e) {
      msg.textContent = "Server sync needs serve.py running.";
    }
  };
  document.getElementById("trk-sync-down").onclick = async () => {
    const token = prompt("Enter your account token:");
    if (!token) return;
    try {
      const res = await fetch("/api/progress?token=" + encodeURIComponent(token));
      const body = await res.json();
      restore(body.progress || {});
      msg.textContent = "Progress loaded from server.";
    } catch (e) {
      msg.textContent = "Server sync needs serve.py running.";
    }
  };

  render();
})();
