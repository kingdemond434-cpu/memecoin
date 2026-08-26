// Subscribe form: POSTs to serve.py's /api/subscribe when a backend is
// running; falls back to pointing at the RSS feed on the static build.
(function () {
  const form = document.getElementById("subscribe-form");
  if (!form) return;
  const msg = document.getElementById("sub-msg");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = document.getElementById("sub-email").value;
    const topics = Array.from(
      form.querySelectorAll("input[name=topic]:checked")
    ).map((cb) => cb.value);
    try {
      const res = await fetch("/api/subscribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email, topics: topics }),
      });
      if (!res.ok) throw new Error("bad status");
      msg.textContent = "Subscribed! Alerts will fire on the next pipeline run.";
      form.reset();
    } catch (err) {
      msg.innerHTML =
        'No backend on this static build — subscribe via the <a href="feed.xml">RSS feed</a>, or run serve.py.';
    }
  });
})();
