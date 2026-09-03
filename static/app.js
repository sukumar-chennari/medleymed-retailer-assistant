const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const textInput = document.getElementById("text-input");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const imagePreviewImg = document.getElementById("image-preview-img");
const imageRemoveBtn = document.getElementById("image-remove");
const chatFab = document.getElementById("chat-fab");
const chatPanel = document.getElementById("chat-panel");
const chatClose = document.getElementById("chat-close");
const ctaOpenChat = document.getElementById("cta-open-chat");

let sessionId = localStorage.getItem("session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("session_id", sessionId);
}

function openChat(prefillText) {
  chatPanel.classList.remove("hidden");
  chatFab.classList.add("hidden");
  if (prefillText) {
    textInput.value = prefillText;
  }
  textInput.focus();
}

function closeChat() {
  chatPanel.classList.add("hidden");
  chatFab.classList.remove("hidden");
}

chatFab.addEventListener("click", () => openChat());
chatClose.addEventListener("click", closeChat);
ctaOpenChat.addEventListener("click", () => openChat());

async function loadDashboard() {
  try {
    const res = await fetch("/api/dashboard");
    const data = await res.json();

    document.getElementById("topbar-user").textContent = `Hi, ${data.name}`;
    document.getElementById("card-address").textContent =
      data.address || "No address on file yet — it'll be saved the first time you place an order.";
    document.getElementById("stat-products").textContent = data.catalog_count;
    document.getElementById("stat-orders").textContent = data.orders.length;

    const ordersEl = document.getElementById("card-orders");
    if (!data.orders.length) {
      ordersEl.textContent = "No orders yet.";
    } else {
      ordersEl.innerHTML = "";
      data.orders.forEach((order) => {
        const item = document.createElement("div");
        item.className = "order-item";
        const cancelledTag = order.status === "cancelled" ? " — Cancelled" : "";
        item.textContent = `${order.product_name} — $${order.price_usd} (${order.order_id})${cancelledTag}`;
        if (order.status === "cancelled") {
          item.classList.add("order-cancelled");
        }
        ordersEl.appendChild(item);
      });
    }
  } catch (err) {
    console.error("Failed to load dashboard", err);
  }
}

function formatGuardrailName(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function timeAgo(sqliteUtcString) {
  const then = new Date(sqliteUtcString.replace(" ", "T") + "Z");
  const seconds = Math.max(0, Math.round((Date.now() - then.getTime()) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function renderGuardrailLog(events) {
  const el = document.getElementById("guardrail-log");
  if (!events.length) {
    el.innerHTML = '<div class="guardrail-log-empty">No guardrail interventions yet.</div>';
    return;
  }
  el.innerHTML = "";
  events.forEach((event) => {
    const item = document.createElement("div");
    item.className = "guardrail-log-item";

    const name = document.createElement("span");
    name.className = "guardrail-log-name";
    name.textContent = formatGuardrailName(event.name);

    const detail = document.createElement("span");
    detail.className = "guardrail-log-detail";
    detail.textContent = event.detail || "";
    detail.title = event.detail || "";

    const time = document.createElement("span");
    time.className = "guardrail-log-time";
    time.textContent = timeAgo(event.created_at);

    item.appendChild(name);
    item.appendChild(detail);
    item.appendChild(time);
    el.appendChild(item);
  });
}

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    const confidenceEl = document.getElementById("stat-retrieval-confidence");
    confidenceEl.textContent =
      data.avg_retrieval_confidence == null ? "–" : `${Math.round(data.avg_retrieval_confidence * 100)}%`;

    document.getElementById("stat-guardrails").textContent = data.guardrail_total;

    const feedbackEl = document.getElementById("stat-feedback");
    feedbackEl.textContent =
      data.feedback_positive_rate == null ? "–" : `${Math.round(data.feedback_positive_rate * 100)}%`;

    renderGuardrailLog(data.recent_guardrail_events || []);
  } catch (err) {
    console.error("Failed to load metrics", err);
  }
}

async function sendFeedback(rating, replyText, chosenBtn, otherBtn) {
  chosenBtn.classList.add("chosen");
  chosenBtn.disabled = true;
  otherBtn.disabled = true;
  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, rating, reply_snippet: replyText.slice(0, 200) }),
    });
    loadMetrics();
  } catch (err) {
    console.error("Failed to send feedback", err);
  }
}

function buildFeedbackControls(replyText) {
  const wrap = document.createElement("div");
  wrap.className = "feedback-controls";

  const upBtn = document.createElement("button");
  upBtn.type = "button";
  upBtn.className = "feedback-btn";
  upBtn.textContent = "\u{1F44D}";
  upBtn.setAttribute("aria-label", "Mark this reply as helpful");

  const downBtn = document.createElement("button");
  downBtn.type = "button";
  downBtn.className = "feedback-btn";
  downBtn.textContent = "\u{1F44E}";
  downBtn.setAttribute("aria-label", "Mark this reply as not helpful");

  upBtn.addEventListener("click", () => sendFeedback("up", replyText, upBtn, downBtn));
  downBtn.addEventListener("click", () => sendFeedback("down", replyText, downBtn, upBtn));

  wrap.appendChild(upBtn);
  wrap.appendChild(downBtn);
  return wrap;
}

let allProducts = [];

function renderCatalog(products) {
  const catalogEl = document.getElementById("catalog-grid");
  catalogEl.innerHTML = "";

  if (!products.length) {
    catalogEl.innerHTML = '<p class="catalog-empty">No products match your search.</p>';
    return;
  }

  products.forEach((product) => {
    const card = document.createElement("div");
    card.className = "product-card";

    const top = document.createElement("div");
    top.className = "product-card-top";

    const name = document.createElement("span");
    name.className = "product-name";
    name.textContent = product.name;

    const pill = document.createElement("span");
    pill.className = `category-pill ${product.category}`;
    pill.textContent = product.category;

    top.appendChild(name);
    top.appendChild(pill);

    const ingredient = document.createElement("div");
    ingredient.className = "product-ingredient";
    ingredient.textContent = product.active_ingredient;

    const price = document.createElement("div");
    price.className = "product-price";
    price.textContent = `$${product.price_usd}`;

    const askButton = document.createElement("button");
    askButton.type = "button";
    askButton.className = "product-ask-button";
    askButton.textContent = "Ask Assistant";
    askButton.addEventListener("click", () => {
      openChat(`I'd like to order ${product.name}`);
    });

    card.appendChild(top);
    card.appendChild(ingredient);
    card.appendChild(price);
    card.appendChild(askButton);
    catalogEl.appendChild(card);
  });
}

function filterCatalog(query) {
  const q = query.trim().toLowerCase();
  if (!q) return allProducts;
  return allProducts.filter(
    (product) =>
      product.name.toLowerCase().includes(q) ||
      product.category.toLowerCase().includes(q) ||
      product.active_ingredient.toLowerCase().includes(q)
  );
}

async function loadCatalog() {
  const catalogEl = document.getElementById("catalog-grid");
  try {
    const res = await fetch("/api/catalog");
    allProducts = await res.json();
    renderCatalog(allProducts);
  } catch (err) {
    catalogEl.textContent = "Couldn't load the product catalog.";
    console.error("Failed to load catalog", err);
  }
}

const catalogSearchInput = document.getElementById("catalog-search");
catalogSearchInput.addEventListener("input", () => {
  renderCatalog(filterCatalog(catalogSearchInput.value));
});

loadDashboard();
loadMetrics();
loadCatalog();

let pendingImage = null; // { b64, mediaType, dataUrl }

function addBubble(role, text, imageDataUrl) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  if (imageDataUrl) {
    const img = document.createElement("img");
    img.src = imageDataUrl;
    bubble.appendChild(img);
  }
  const textNode = document.createElement("span");
  textNode.textContent = text;
  bubble.appendChild(textNode);
  if (role === "assistant") {
    bubble.appendChild(buildFeedbackControls(text));
  }
  chatEl.appendChild(bubble);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    const b64 = dataUrl.split(",")[1];
    pendingImage = { b64, mediaType: file.type, dataUrl };
    imagePreviewImg.src = dataUrl;
    imagePreview.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

imageRemoveBtn.addEventListener("click", () => {
  pendingImage = null;
  imageInput.value = "";
  imagePreview.classList.add("hidden");
});

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  const image = pendingImage;

  if (!text && !image) return;

  addBubble("user", text, image ? image.dataUrl : null);
  textInput.value = "";
  pendingImage = null;
  imageInput.value = "";
  imagePreview.classList.add("hidden");

  const pendingBubble = addBubble("assistant pending", "…");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        text,
        image_b64: image ? image.b64 : null,
        image_media_type: image ? image.mediaType : null,
      }),
    });

    if (!res.ok) {
      throw new Error(`Request failed (${res.status})`);
    }

    const data = await res.json();
    pendingBubble.remove();
    addBubble("assistant", data.reply);
    loadDashboard();
    loadMetrics();
  } catch (err) {
    pendingBubble.remove();
    addBubble("assistant", "Sorry, something went wrong reaching the assistant. Please try again.");
    console.error(err);
  }
});
