import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const PORT = Number(process.env.PORT || 3000);
const BASE_URL = process.env.COMFY_CLOUD_BASE_URL || "https://cloud.comfy.org";
const API_KEY = process.env.COMFY_CLOUD_API_KEY || "";
const PARTNER_KEY = process.env.COMFY_PARTNER_API_KEY || "";
const PUBLIC_BASE_URL = process.env.PUBLIC_BASE_URL || `http://localhost:${PORT}`;
const COMFY_CKPT_NAME = process.env.COMFY_CKPT_NAME || "your_model.safetensors";
const WORKFLOW_TEMPLATE = process.env.WORKFLOW_TEMPLATE || path.join(__dirname, "../comfyui/workflow_api_img2img.json");
const DEFAULT_NEG = process.env.DEFAULT_NEG_PROMPT || "low quality, extra fingers, duplicate object, distorted product, changed product shape, changed color, bad anatomy, cluttered background, cropped product, extra accessories, watermark, text";

function assertEnv() {
  if (!API_KEY) throw new Error("Missing COMFY_CLOUD_API_KEY");
}
function getHeaders(json = true) {
  return json
    ? { "X-API-Key": API_KEY, "Content-Type": "application/json" }
    : { "X-API-Key": API_KEY };
}
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}
async function loadWorkflowTemplate() {
  const raw = await fs.readFile(WORKFLOW_TEMPLATE, "utf-8");
  return JSON.parse(raw);
}
function setWorkflowInput(workflow, nodeId, inputName, value) {
  if (workflow[nodeId] && workflow[nodeId].inputs) {
    workflow[nodeId].inputs[inputName] = value;
  }
  return workflow;
}
function patchWorkflow(workflow, params) {
  workflow = structuredClone(workflow);
  workflow = setWorkflowInput(workflow, "1", "ckpt_name", params.ckpt_name || COMFY_CKPT_NAME);
  workflow = setWorkflowInput(workflow, "2", "image", params.input_image);
  workflow = setWorkflowInput(workflow, "4", "text", params.prompt_text);
  workflow = setWorkflowInput(workflow, "5", "text", params.neg_prompt || DEFAULT_NEG);
  workflow = setWorkflowInput(workflow, "6", "seed", Number(params.seed || 1));
  workflow = setWorkflowInput(workflow, "6", "steps", Number(params.steps || 24));
  workflow = setWorkflowInput(workflow, "6", "cfg", Number(params.cfg || 6.5));
  workflow = setWorkflowInput(workflow, "6", "denoise", Number(params.denoise || 0.55));
  workflow = setWorkflowInput(workflow, "8", "filename_prefix", params.shot_id || "shot");
  return workflow;
}
async function downloadUrlToBlob(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download input image: HTTP ${res.status}`);
  const buffer = Buffer.from(await res.arrayBuffer());
  const contentType = res.headers.get("content-type") || "image/png";
  const ext = contentType.includes("jpeg") ? "jpg" : contentType.includes("webp") ? "webp" : "png";
  return { buffer, contentType, ext };
}
async function uploadInputImage(imgUrl) {
  const { buffer, contentType, ext } = await downloadUrlToBlob(imgUrl);
  const form = new FormData();
  form.append("image", new Blob([buffer], { type: contentType }), `input.${ext}`);
  form.append("type", "input");
  form.append("overwrite", "true");
  const res = await fetch(`${BASE_URL}/api/upload/image`, {
    method: "POST",
    headers: { "X-API-Key": API_KEY },
    body: form
  });
  if (!res.ok) {
    throw new Error(`Upload failed: HTTP ${res.status} ${await res.text()}`);
  }
  return await res.json();
}
async function submitWorkflow(workflow) {
  const body = { prompt: workflow };
  if (PARTNER_KEY) body.extra_data = { api_key_comfy_org: PARTNER_KEY };
  const res = await fetch(`${BASE_URL}/api/prompt`, {
    method: "POST",
    headers: getHeaders(true),
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`Submit failed: HTTP ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.prompt_id;
}
async function waitForCompletion(promptId, timeoutMs = 300000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const res = await fetch(`${BASE_URL}/api/job/${promptId}/status`, { headers: getHeaders(false) });
    if (!res.ok) throw new Error(`Status failed: HTTP ${res.status}`);
    const data = await res.json();
    const status = data.status;
    if (status === "completed") return true;
    if (status === "failed" || status === "cancelled") throw new Error(`Job ${status}`);
    await sleep(2000);
  }
  throw new Error("Job timeout");
}
async function fetchOutputs(promptId) {
  const res = await fetch(`${BASE_URL}/api/history_v2/${promptId}`, { headers: getHeaders(false) });
  if (!res.ok) throw new Error(`History failed: HTTP ${res.status}`);
  const data = await res.json();
  return data.outputs || {};
}
function firstImageFile(outputs) {
  for (const nodeOutputs of Object.values(outputs)) {
    const images = nodeOutputs?.images || [];
    if (images.length) return images[0];
  }
  return null;
}
function buildProxyUrl(fileInfo) {
  const params = new URLSearchParams({
    filename: fileInfo.filename,
    subfolder: fileInfo.subfolder || "",
    type: fileInfo.type || "output"
  });
  return `${PUBLIC_BASE_URL}/output_proxy?${params.toString()}`;
}
function compilePrompt({ product_profile, style_strategy, global_style_line, shot_task }) {
  const parts = [];
  const mainProd = product_profile?.main_prod || "product";
  const accProd = product_profile?.acc_prod || "";
  parts.push(`commercial etsy product photography of ${mainProd}`);
  if (accProd && shot_task?.need_acc) {
    parts.push(`with accessory ${accProd}`);
  }
  if (shot_task?.shot_goal) parts.push(shot_task.shot_goal);
  if (shot_task?.scene_hint) parts.push(shot_task.scene_hint);
  if (shot_task?.light_hint) parts.push(shot_task.light_hint);
  if (shot_task?.props_hint) parts.push(shot_task.props_hint);
  if (shot_task?.ratio_rule) parts.push(`main to accessory ratio ${shot_task.ratio_rule}`);
  if (style_strategy?.scene_rule) parts.push(style_strategy.scene_rule);
  if (style_strategy?.light_rule) parts.push(style_strategy.light_rule);
  if (style_strategy?.props_rule) parts.push(style_strategy.props_rule);
  if (style_strategy?.comp_rule) parts.push(style_strategy.comp_rule);
  if (global_style_line) parts.push(global_style_line);
  parts.push("clean background, product shape preserved, realistic materials, hasselblad-like commercial realism");
  return parts.filter(Boolean).join(", ");
}
function scoreItem({ shot_task, success }) {
  const groupWeight = {
    hero: 1.0,
    lifestyle: 0.95,
    detail: 0.9,
    usage: 0.88,
    angle: 0.86,
    feature: 0.84,
    advertising: 0.82,
    wildcard: 0.8
  };
  const w = groupWeight[shot_task?.shot_group] || 0.8;
  const priority = Number(shot_task?.priority || 0);
  const successBonus = success ? 0.2 : 0;
  const finalScore = successBonus + w + priority / 1000;
  return {
    qc_score: success ? 0.85 : 0,
    comp_score: w,
    etsy_score: Math.min(1, w),
    final_score: finalScore
  };
}
function pickTop12(items, finalCount = 12) {
  const quotas = { hero:1, lifestyle:3, detail:2, usage:2, angle:2, feature:1, advertising:1 };
  const groups = {};
  for (const item of items) {
    const g = item.shot_group || "wildcard";
    groups[g] ||= [];
    groups[g].push(item);
  }
  for (const g of Object.keys(groups)) {
    groups[g].sort((a,b) => (b.final_score || 0) - (a.final_score || 0));
  }
  const selected = [];
  const used = new Set();
  for (const [g, quota] of Object.entries(quotas)) {
    const arr = groups[g] || [];
    for (const item of arr.slice(0, quota)) {
      selected.push(item);
      used.add(item.shot_id);
    }
  }
  const leftovers = items
    .filter(x => !used.has(x.shot_id))
    .sort((a,b) => (b.final_score || 0) - (a.final_score || 0));
  while (selected.length < finalCount && leftovers.length) {
    selected.push(leftovers.shift());
  }
  const selectedIds = new Set(selected.map(x => x.shot_id));
  const dropped = items.filter(x => !selectedIds.has(x.shot_id));
  return { top12_items: selected.slice(0, finalCount), drop4_items: dropped };
}
function extractImgUrls(body) {
  if (Array.isArray(body.img_urls) && body.img_urls.length) return body.img_urls;
  if (Array.isArray(body.img_inputs) && body.img_inputs.length) {
    return body.img_inputs.map((x) => {
      if (typeof x === "string") return x;
      return x?.url || x?.uri || x?.image_url || x?.src || "";
    }).filter(Boolean);
  }
  return [];
}
async function generateOneInternal(body) {
  assertEnv();
  const imgUrls = extractImgUrls(body);
  if (!imgUrls.length) throw new Error("img_urls is empty");
  const uploaded = await uploadInputImage(imgUrls[0]);
  const inputName = uploaded.name || uploaded.filename || uploaded.image || uploaded.file || uploaded?.filename;
  if (!inputName) throw new Error("Upload response does not contain uploaded filename");
  let workflow = await loadWorkflowTemplate();
  workflow = patchWorkflow(workflow, {
    input_image: inputName,
    prompt_text: body.prompt_text,
    neg_prompt: body.neg_prompt || DEFAULT_NEG,
    shot_id: body.shot_id,
    steps: body.steps,
    cfg: body.cfg,
    denoise: clamp(Number(body.denoise ?? 0.55), 0, 1),
    seed: body.seed,
    ckpt_name: body.ckpt_name
  });
  const promptId = await submitWorkflow(workflow);
  await waitForCompletion(promptId);
  const outputs = await fetchOutputs(promptId);
  const fileInfo = firstImageFile(outputs);
  if (!fileInfo) {
    return {
      image_url: "",
      shot_id: body.shot_id,
      shot_group: body.shot_group,
      ...scoreItem({ shot_task: { shot_group: body.shot_group, priority: 0 }, success: false }),
      fail_reason: "No image output found",
      prompt_text: body.prompt_text
    };
  }
  return {
    image_url: buildProxyUrl(fileInfo),
    shot_id: body.shot_id,
    shot_group: body.shot_group,
    ...scoreItem({ shot_task: { shot_group: body.shot_group, priority: 0 }, success: true }),
    fail_reason: "",
    prompt_text: body.prompt_text,
    file_info: fileInfo
  };
}

app.get("/health", async (_req, res) => {
  res.json({ ok: true, service: "coze-comfy-bridge", base_url: BASE_URL });
});

app.get("/output_proxy", async (req, res) => {
  try {
    assertEnv();
    const params = new URLSearchParams({
      filename: String(req.query.filename || ""),
      subfolder: String(req.query.subfolder || ""),
      type: String(req.query.type || "output")
    });
    const response = await fetch(`${BASE_URL}/api/view?${params.toString()}`, {
      headers: getHeaders(false),
      redirect: "manual"
    });
    if (response.status !== 302) throw new Error(`view failed: HTTP ${response.status}`);
    const signedUrl = response.headers.get("location");
    if (!signedUrl) throw new Error("Missing signed URL");
    const fileRes = await fetch(signedUrl);
    if (!fileRes.ok) throw new Error(`signed fetch failed: HTTP ${fileRes.status}`);
    res.setHeader("Content-Type", fileRes.headers.get("content-type") || "application/octet-stream");
    res.setHeader("Cache-Control", "public, max-age=300");
    const arrayBuffer = await fileRes.arrayBuffer();
    res.send(Buffer.from(arrayBuffer));
  } catch (err) {
    res.status(500).json({ ok:false, error: String(err.message || err) });
  }
});

app.post("/generate_one", async (req, res) => {
  try {
    const data = await generateOneInternal(req.body || {});
    res.json(data);
  } catch (err) {
    res.status(500).json({ ok:false, error: String(err.message || err) });
  }
});

app.post("/run_batch16", async (req, res) => {
  try {
    assertEnv();
    const body = req.body || {};
    const shotTasks = Array.isArray(body.shot_tasks) ? body.shot_tasks : [];
    const imgUrls = extractImgUrls(body);
    if (!imgUrls.length) {
      return res.status(400).json({ ok:false, error:"img_urls or img_inputs is required" });
    }
    if (!shotTasks.length) {
      return res.status(400).json({ ok:false, error:"shot_tasks is required" });
    }
    const items = [];
    const baseSeed = Number(body.seed || 1);
    for (let i = 0; i < shotTasks.length; i++) {
      const shot = shotTasks[i];
      const promptText = compilePrompt({
        product_profile: body.product_profile || {},
        style_strategy: body.style_strategy || {},
        global_style_line: body.global_style_line || "",
        shot_task: shot
      });
      try {
        const item = await generateOneInternal({
          img_urls: imgUrls,
          prompt_text: promptText,
          neg_prompt: body.neg_prompt || DEFAULT_NEG,
          shot_id: shot.shot_id || `shot_${i+1}`,
          shot_group: shot.shot_group || "wildcard",
          steps: body.steps || 24,
          cfg: body.cfg || 6.5,
          denoise: body.denoise || 0.55,
          seed: baseSeed + i,
          ckpt_name: body.ckpt_name
        });
        const score = scoreItem({ shot_task: shot, success: !item.fail_reason });
        items.push({ ...item, ...score, shot_goal: shot.shot_goal || "", priority: shot.priority || 0 });
      } catch (err) {
        const score = scoreItem({ shot_task: shot, success: false });
        items.push({
          image_url: "",
          shot_id: shot.shot_id || `shot_${i+1}`,
          shot_group: shot.shot_group || "wildcard",
          ...score,
          fail_reason: String(err.message || err),
          prompt_text: promptText,
          shot_goal: shot.shot_goal || "",
          priority: shot.priority || 0
        });
      }
    }
    const { top12_items, drop4_items } = pickTop12(items, Number(body.final_count || 12));
    const summary_text = `已生成 ${items.length} 张候选图，筛选出 ${top12_items.length} 张结果。`;
    res.json({ items, top12_items, drop4_items, summary_text });
  } catch (err) {
    res.status(500).json({ ok:false, error: String(err.message || err) });
  }
});

app.listen(PORT, () => {
  console.log(`Bridge listening on :${PORT}`);
});
