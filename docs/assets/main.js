// Mobile nav toggle
const toggleBtn = document.querySelector(".nav-toggle");
const nav = document.querySelector(".nav");
if (toggleBtn && nav) {
  toggleBtn.addEventListener("click", () => {
    const isHidden = getComputedStyle(nav).display === "none";
    nav.style.display = isHidden ? "flex" : "none";
    nav.style.flexDirection = "column";
    nav.style.position = "absolute";
    nav.style.top = "58px";
    nav.style.right = "20px";
    nav.style.background = "rgba(11,15,25,.92)";
    nav.style.border = "1px solid rgba(255,255,255,.12)";
    nav.style.borderRadius = "16px";
    nav.style.padding = "10px";
    nav.style.width = "200px";
  });
}

// Tabs switch
const tabs = document.querySelectorAll(".tab");
const panels = {
  qual: document.getElementById("tab-qual"),
  quant: document.getElementById("tab-quant"),
};

tabs.forEach((t) => {
  t.addEventListener("click", () => {
    tabs.forEach((x) => x.classList.remove("active"));
    t.classList.add("active");

    const key = t.dataset.tab;
    Object.values(panels).forEach((p) => p.classList.remove("active"));
    if (panels[key]) panels[key].classList.add("active");
  });
});
