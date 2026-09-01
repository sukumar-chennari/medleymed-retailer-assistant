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
        item.textContent = `${order.product_name} — $${order.price_usd} (${order.order_id})`;
        ordersEl.appendChild(item);
      });
    }
  } catch (err) {
    console.error("Failed to load dashboard", err);
  }
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

async function loadCatalog() {
  const catalogEl = document.getElementById("catalog-grid");
  try {
    const res = await fetch("/api/catalog");
    const products = await res.json();

    catalogEl.innerHTML = "";
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
  } catch (err) {
    catalogEl.textContent = "Couldn't load the product catalog.";
    console.error("Failed to load catalog", err);
  }
}

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
