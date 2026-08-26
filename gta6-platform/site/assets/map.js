// Interactive Leonida map: pan/zoom the SVG viewBox, render pins from the
// embedded JSON, category filters, click-for-details panel, and a
// localStorage "collected" state shared with the tracker (key: trk:locations:<slug>).
(function () {
  const mapData = JSON.parse(document.getElementById("map-data").textContent);
  const svg = document.getElementById("leonida-map");
  const pinLayer = svg.querySelector("#pins");
  const frame = document.getElementById("map-frame");
  const panel = document.getElementById("pin-panel");
  const SIZE = 1000;
  let view = { x: 0, y: 0, w: SIZE, h: SIZE };

  const store = {
    get(slug) {
      try { return localStorage.getItem("trk:locations:" + slug) === "1"; }
      catch (e) { return false; }
    },
    set(slug, val) {
      try { localStorage.setItem("trk:locations:" + slug, val ? "1" : "0"); }
      catch (e) { /* private mode */ }
    },
  };

  function applyView() {
    svg.setAttribute("viewBox", view.x + " " + view.y + " " + view.w + " " + view.h);
  }

  function drawPins() {
    const enabled = {};
    document.querySelectorAll(".legend-item input").forEach((cb) => {
      enabled[cb.dataset.category] = cb.checked;
    });
    pinLayer.innerHTML = "";
    mapData.pins.forEach((pin) => {
      if (enabled[pin.category] === false) return;
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "map-pin" + (store.get(pin.slug) ? " collected" : ""));
      g.dataset.slug = pin.slug;
      const color = mapData.categories[pin.category] || "#fff";
      g.innerHTML =
        '<circle cx="' + pin.x + '" cy="' + pin.y + '" r="9" fill="' + color + '"></circle>' +
        '<text x="' + (pin.x + 13) + '" y="' + (pin.y + 4) + '">' + pin.name + "</text>";
      g.addEventListener("click", (e) => {
        e.stopPropagation();
        openPanel(pin);
      });
      pinLayer.appendChild(g);
    });
  }

  function openPanel(pin) {
    const links = (list, prefix) =>
      list.map((x) => '<a href="' + prefix + x.slug + '.html">' + x.name + "</a>").join(", ");
    panel.innerHTML =
      '<span class="close" id="pin-close">✕</span>' +
      "<h3>" + pin.name + "</h3>" +
      '<p class="muted">' + pin.region + " · " + pin.category.replace("_", " ") +
      " · " + pin.status + "</p>" +
      "<p>" + pin.summary + "</p>" +
      (pin.vehicles.length ? "<p><strong>Vehicles:</strong> " + links(pin.vehicles, "vehicles/") + "</p>" : "") +
      (pin.missions.length ? "<p><strong>Missions:</strong> " + links(pin.missions, "missions/") + "</p>" : "") +
      '<label><input type="checkbox" id="pin-collected"' +
      (store.get(pin.slug) ? " checked" : "") + "> Collected / visited</label>";
    panel.hidden = false;
    document.getElementById("pin-close").onclick = () => (panel.hidden = true);
    document.getElementById("pin-collected").onchange = (e) => {
      store.set(pin.slug, e.target.checked);
      drawPins();
    };
  }

  function focusPin(slug) {
    const pin = mapData.pins.find((p) => p.slug === slug);
    if (!pin) return;
    view = { x: pin.x - 150, y: pin.y - 150, w: 300, h: 300 };
    applyView();
    openPanel(pin);
  }

  // --- pan (pointer drag) ---
  let dragging = null;
  frame.addEventListener("pointerdown", (e) => {
    dragging = { px: e.clientX, py: e.clientY, vx: view.x, vy: view.y };
    frame.setPointerCapture(e.pointerId);
  });
  frame.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = frame.getBoundingClientRect();
    const scale = view.w / rect.width;
    view.x = dragging.vx - (e.clientX - dragging.px) * scale;
    view.y = dragging.vy - (e.clientY - dragging.py) * scale;
    applyView();
  });
  frame.addEventListener("pointerup", () => (dragging = null));

  // --- zoom (wheel, about cursor) ---
  frame.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const rect = frame.getBoundingClientRect();
      const mx = view.x + ((e.clientX - rect.left) / rect.width) * view.w;
      const my = view.y + ((e.clientY - rect.top) / rect.height) * view.h;
      const factor = e.deltaY > 0 ? 1.2 : 1 / 1.2;
      const w = Math.min(SIZE * 1.5, Math.max(80, view.w * factor));
      const h = w;
      view = { x: mx - ((mx - view.x) / view.w) * w, y: my - ((my - view.y) / view.h) * h, w: w, h: h };
      applyView();
    },
    { passive: false }
  );

  document.querySelectorAll(".legend-item input").forEach((cb) =>
    cb.addEventListener("change", drawPins)
  );

  applyView();
  drawPins();
  const focus = new URLSearchParams(location.search).get("focus");
  if (focus) focusPin(focus);
})();
