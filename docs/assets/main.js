// tiny selectors
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// --------------------
// Mobile nav toggle
// --------------------
const toggleBtn = $(".nav-toggle");
const nav = $(".nav");

if (toggleBtn && nav) {
  toggleBtn.addEventListener("click", () => {
    if (getComputedStyle(nav).display === "none") nav.style.display = "flex";
    nav.classList.toggle("mobile-open");
  });

  // click a nav item -> close dropdown (mobile)
  $$(".nav .nav-link").forEach((a) => {
    a.addEventListener("click", () => {
      if (nav.classList.contains("mobile-open")) {
        nav.classList.remove("mobile-open");
        nav.style.display = ""; // let css handle
      }
    });
  });
}

// --------------------
// Results Tabs
// --------------------
const tabs = $$(".tab");
const panels = {
  qual: $("#tab-qual"),
  quant: $("#tab-quant"),
};

tabs.forEach((t) => {
  t.addEventListener("click", () => {
    tabs.forEach((x) => {
      x.classList.remove("active");
      x.setAttribute("aria-selected", "false");
    });

    t.classList.add("active");
    t.setAttribute("aria-selected", "true");

    const key = t.dataset.tab;
    Object.values(panels).forEach((p) => p && p.classList.remove("active"));
    if (panels[key]) panels[key].classList.add("active");
  });
});

// --------------------
// Nav underline (滑动横线像图3)
// underline 必须在 nav 内：<span class="nav-underline-bar"></span>
// --------------------
const navLinks = $$(".nav .nav-link");
const underline = $(".nav .nav-underline-bar");
const sections = $$("main section[id]");

function setUnderlineToLink(link) {
  if (!underline || !nav || !link) return;

  const linkRect = link.getBoundingClientRect();
  const navRect = nav.getBoundingClientRect();

  const left = linkRect.left - navRect.left;
  const width = linkRect.width;

  underline.style.transform = `translateX(${left}px)`;
  underline.style.width = `${width}px`;
}

function setActiveById(id) {
  const target = navLinks.find((a) => a.getAttribute("href") === `#${id}`);
  if (!target) return;

  navLinks.forEach((a) => a.classList.remove("active"));
  target.classList.add("active");
  setUnderlineToLink(target);
}

// click nav -> smooth scroll + underline
navLinks.forEach((a) => {
  a.addEventListener("click", (e) => {
    const href = a.getAttribute("href") || "";
    if (!href.startsWith("#")) return;

    const el = document.getElementById(href.slice(1));
    if (!el) return;

    e.preventDefault();
    el.scrollIntoView({ behavior: "smooth", block: "start" });

    navLinks.forEach((x) => x.classList.remove("active"));
    a.classList.add("active");
    setUnderlineToLink(a);
  });
});

// scroll highlight: IntersectionObserver
if (sections.length && navLinks.length) {
  const obs = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((x) => x.isIntersecting)
        .sort((a, b) => (b.intersectionRatio || 0) - (a.intersectionRatio || 0))[0];

      if (visible?.target?.id) setActiveById(visible.target.id);
    },
    {
      root: null,
      rootMargin: "-25% 0px -65% 0px",
      threshold: [0.15, 0.3, 0.45, 0.6],
    }
  );

  sections.forEach((s) => obs.observe(s));
}

// initial underline
window.addEventListener("load", () => {
  const hash = (location.hash || "").replace("#", "");
  if (hash && document.getElementById(hash)) {
    setActiveById(hash);
    return;
  }
  if (sections[0]?.id) {
    setActiveById(sections[0].id);
    return;
  }
  if (navLinks[0]) {
    navLinks[0].classList.add("active");
    setUnderlineToLink(navLinks[0]);
  }
});

// resize -> reposition underline
window.addEventListener("resize", () => {
  const active = $(".nav .nav-link.active");
  if (active) setUnderlineToLink(active);
});
