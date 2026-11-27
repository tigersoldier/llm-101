(() => {
  const MIN_SCALE = 1;
  const MAX_SCALE = 8;

  let overlay;
  let stage;
  let zoomCanvas;
  let closeButton;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let moved = false;
  let dragStart = null;
  let pinchStart = null;
  let openToken = 0;
  const pointers = new Map();

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function applyTransform() {
    zoomCanvas.style.transform =
      `translate3d(${offsetX}px, ${offsetY}px, 0) scale(${scale})`;
  }

  function resetTransform() {
    scale = 1;
    offsetX = 0;
    offsetY = 0;
    applyTransform();
  }

  function zoomAt(nextScale, clientX, clientY) {
    const clamped = clamp(nextScale, MIN_SCALE, MAX_SCALE);
    if (clamped === scale) return;
    const rect = stage.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const pointX = clientX - centerX;
    const pointY = clientY - centerY;
    const ratio = clamped / scale;
    offsetX = pointX - (pointX - offsetX) * ratio;
    offsetY = pointY - (pointY - offsetY) * ratio;
    scale = clamped;
    applyTransform();
  }

  function centerPoint() {
    const rect = stage.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
  }

  function showImageFallback(src, alt) {
    const image = document.createElement("img");
    image.src = src;
    image.alt = alt;
    image.draggable = false;
    zoomCanvas.replaceChildren(image);
  }

  async function openZoom(source, pushHistory = true) {
    const token = ++openToken;
    const src = source.currentSrc || source.src;
    const alt = source.alt || "";
    const wasOpen = overlay.classList.contains("is-open");
    const path = location.pathname + location.search + location.hash;
    if (!wasOpen && pushHistory) {
      const currentState =
        history.state && typeof history.state === "object" ? history.state : {};
      history.pushState(
        { ...currentState, imageZoom: { src, alt, path } },
        "",
        location.href,
      );
    } else if (wasOpen && history.state?.imageZoom) {
      history.replaceState(
        { ...history.state, imageZoom: { src, alt, path } },
        "",
        location.href,
      );
    }

    zoomCanvas.replaceChildren();
    resetTransform();
    overlay.classList.add("is-open");
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("image-zoom-open");
    closeButton.focus({ preventScroll: true });

    if (!src.toLowerCase().endsWith(".svg")) {
      showImageFallback(src, alt);
      return;
    }

    try {
      const response = await fetch(src);
      if (!response.ok) throw new Error(`SVG request failed: ${response.status}`);
      const markup = await response.text();
      if (token !== openToken) return;
      const documentNode = new DOMParser().parseFromString(markup, "image/svg+xml");
      const svg = documentNode.documentElement;
      if (svg.nodeName !== "svg") throw new Error("not an SVG document");
      const viewBox = (svg.getAttribute("viewBox") || "")
        .trim()
        .split(/[\s,]+/)
        .map(Number);
      if (viewBox.length === 4 && viewBox.every(Number.isFinite)) {
        svg.setAttribute("width", String(viewBox[2]));
        svg.setAttribute("height", String(viewBox[3]));
      }
      svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", alt);
      zoomCanvas.replaceChildren(document.importNode(svg, true));
    } catch (error) {
      if (token !== openToken) return;
      showImageFallback(src, alt);
    }
  }

  function closeZoom({ fromHistory = false, syncHistory = true } = {}) {
    if (!overlay || !overlay.classList.contains("is-open")) return;
    overlay.classList.remove("is-open");
    overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("image-zoom-open");
    openToken += 1;
    pointers.clear();
    dragStart = null;
    pinchStart = null;
    if (!fromHistory && syncHistory && history.state?.imageZoom) {
      history.back();
    }
  }

  function bindOverlay() {
    overlay = document.createElement("div");
    overlay.className = "image-zoom";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "放大查看图片");
    overlay.setAttribute("aria-hidden", "true");
    overlay.innerHTML = `
      <div class="image-zoom__stage"></div>
      <div class="image-zoom__toolbar">
        <button type="button" data-action="zoom-out" aria-label="缩小">−</button>
        <button type="button" data-action="zoom-in" aria-label="放大">+</button>
        <button type="button" data-action="reset" aria-label="还原">⟲</button>
        <button type="button" data-action="close" aria-label="关闭">×</button>
      </div>
      <p class="image-zoom__hint">双指缩放 · 拖动移动 · 双击放大</p>
    `;
    document.body.appendChild(overlay);

    stage = overlay.querySelector(".image-zoom__stage");
    closeButton = overlay.querySelector('[data-action="close"]');
    zoomCanvas = document.createElement("div");
    zoomCanvas.className = "image-zoom__canvas";
    stage.appendChild(zoomCanvas);

    overlay.querySelector(".image-zoom__toolbar").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      const action = button.dataset.action;
      const point = centerPoint();
      if (action === "close") closeZoom();
      if (action === "reset") resetTransform();
      if (action === "zoom-in") zoomAt(scale * 1.4, point.x, point.y);
      if (action === "zoom-out") zoomAt(scale / 1.4, point.x, point.y);
    });

    stage.addEventListener("click", (event) => {
      if (event.target === stage && !moved) closeZoom();
    });

    stage.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
      zoomAt(scale * factor, event.clientX, event.clientY);
    }, { passive: false });

    stage.addEventListener("dblclick", (event) => {
      event.preventDefault();
      const next = scale > 1.5 ? 1 : 2.5;
      zoomAt(next, event.clientX, event.clientY);
    });

    stage.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      event.preventDefault();
      stage.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      moved = false;
      if (pointers.size === 1) {
        dragStart = {
          x: event.clientX,
          y: event.clientY,
          offsetX,
          offsetY,
        };
        pinchStart = null;
      } else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchStart = {
          distance: Math.hypot(a.x - b.x, a.y - b.y),
          scale,
          midX: (a.x + b.x) / 2,
          midY: (a.y + b.y) / 2,
        };
        dragStart = null;
      }
    });

    stage.addEventListener("pointermove", (event) => {
      if (!pointers.has(event.pointerId)) return;
      event.preventDefault();
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

      if (pointers.size === 1 && dragStart) {
        const dx = event.clientX - dragStart.x;
        const dy = event.clientY - dragStart.y;
        if (Math.hypot(dx, dy) > 3) moved = true;
        offsetX = dragStart.offsetX + dx;
        offsetY = dragStart.offsetY + dy;
        applyTransform();
      } else if (pointers.size === 2 && pinchStart) {
        const [a, b] = [...pointers.values()];
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        const midX = (a.x + b.x) / 2;
        const midY = (a.y + b.y) / 2;
        const nextScale = pinchStart.scale * (distance / pinchStart.distance);
        zoomAt(nextScale, midX, midY);
        pointerPan(midX - pinchStart.midX, midY - pinchStart.midY);
        moved = true;
      }
    });

    function pointerPan(dx, dy) {
      offsetX += dx;
      offsetY += dy;
      pinchStart.midX += dx;
      pinchStart.midY += dy;
      applyTransform();
    }

    function endPointer(event) {
      pointers.delete(event.pointerId);
      if (pointers.size < 2) pinchStart = null;
      if (pointers.size === 1) {
        const remaining = [...pointers.values()][0];
        dragStart = {
          x: remaining.x,
          y: remaining.y,
          offsetX,
          offsetY,
        };
      }
      if (pointers.size === 0) dragStart = null;
    }

    stage.addEventListener("pointerup", endPointer);
    stage.addEventListener("pointercancel", endPointer);
    stage.addEventListener("dragstart", (event) => event.preventDefault());

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && overlay.classList.contains("is-open")) {
        closeZoom();
      }
    });

    window.addEventListener("popstate", (event) => {
      const saved = event.state?.imageZoom;
      if (overlay.classList.contains("is-open")) {
        if (!saved) closeZoom({ fromHistory: true });
        return;
      }
      if (!saved || typeof saved.src !== "string") return;
      if (saved.path && saved.path !== location.pathname + location.search + location.hash) {
        return;
      }
      openZoom(
        { currentSrc: saved.src, src: saved.src, alt: saved.alt || "" },
        false,
      );
    });
  }

  function bindPageImages() {
    document.querySelectorAll(".md-content img").forEach((image) => {
      if (image.dataset.zoomBound) return;
      image.dataset.zoomBound = "true";
      image.addEventListener("click", (event) => {
        event.preventDefault();
        openZoom(image);
      });
    });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => {
      if (!overlay) bindOverlay();
      closeZoom({ syncHistory: false });
      bindPageImages();
    });
  } else {
    document.addEventListener("DOMContentLoaded", () => {
      bindOverlay();
      bindPageImages();
    });
  }
})();
