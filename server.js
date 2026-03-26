import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fsSync from "fs";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function resolveImg2ImgWorkflowPath() {
  const envCandidates = [
    process.env.WORKFLOW_TEMPLATE,
    process.env.WORKFLOW_API_PATH
  ]
    .filter(Boolean)
    .map((p) => (path.isAbsolute(p) ? p : path.resolve(__dirname, p)));

  const candidates = [
    ...envCandidates,
    path.resolve(process.cwd(), "workflow_api_img2img.json"),
    path.resolve(__dirname, "workflow_api_img2img.json"),
    path.resolve(process.cwd(), "comfyui", "workflow_api_img2img.json"),
    path.resolve(__dirname, "comfyui", "workflow_api_img2img.json"),
    path.resolve(__dirname, "..", "comfyui", "workflow_api_img2img.json")
  ];

  for (const p of candidates) {
    if (fsSync.existsSync(p)) return p;
  }

  throw new Error(
    "workflow_api_img2img.json not found, tried: " + candidates.join(" | ")
  );
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const PORT = Number(process.env.PORT || 3000);

const BASE_URL =
  process.env.COMFY_CLOUD_BASE_URL ||
  process.env.COMFY_BASE_URL ||
  "https://cloud.comfy.org";

const API_KEY = process.env.COMFY_CLOUD_API_KEY || "";
const PARTNER_KEY = process.env.COMFY_PARTNER_API_KEY || "";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_VISION_MODEL = process.env.OPENAI_VISION_MODEL || "gpt-4.1-mini";

const PUBLIC_BASE_URL =
  process.env.PUBLIC_BASE_URL ||
  (process.env.RAILWAY_PUBLIC_DOMAIN
    ? `https://${process.env.RAILWAY_PUBLIC_DOMAIN}`
    : `http://localhost:${PORT}`);

const COMFY_CKPT_NAME =
  process.env.COMFY_CKPT_NAME || "v2-1768-ema-pruned.safetensors";

const WORKFLOW_TEMPLATE = resolveImg2ImgWorkflowPath();

const DEFAULT_NEG =
  process.env.DEFAULT_NEG_PROMPT ||
  "low quality, extra fingers, duplicate object, distorted product, changed product shape, changed color, bad anatomy, cluttered background, cropped product, extra accessories, watermark, text";

const SINGLE_JOB_TIMEOUT_MS = Number(
  process.env.SINGLE_JOB_TIMEOUT_MS || 300000
);
const BATCH_JOB_TTL_MS = Number(
  process.env.BATCH_JOB_TTL_MS || 12 * 60 * 60 * 1000
);

const batch16Jobs = new Map();

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

function toSafeString(val) {
  return String(val ?? "").trim();
}

function parseMaybeJson(input) {
  if (!input) return {};
  if (typeof input === "object") return input;
  if (typeof input !== "string") return {};
  try {
    return JSON.parse(input);
  } catch {
    return {};
  }
}

function normalizeImgUrls(input) {
  if (!Array.isArray(input)) return [];
  return input.map((x) => String(x || "").trim()).filter(Boolean);
}

function normalizeShotTasks(input) {
  if (Array.isArray(input)) {
    return input
      .map((x) => (typeof x === "object" && x ? x : null))
      .filter(Boolean);
  }
  if (typeof input === "string") {
    try {
      const parsed = JSON.parse(input);
      return Array.isArray(parsed)
        ? parsed
            .map((x) => (typeof x === "object" && x ? x : null))
            .filter(Boolean)
        : [];
    } catch {
      return [];
    }
  }
  return [];
}

function extractJsonObject(text) {
  if (!text) {
    throw new Error("EMPTY_MODEL_TEXT");
  }

  try {
    return JSON.parse(text);
  } catch (_) {}

  const match = text.match(/\{[\s\S]*\}/);
  if (!match) {
    throw new Error(`NO_JSON_OBJECT_IN_MODEL_TEXT: ${text}`);
  }

  return JSON.parse(match[0]);
}

async function loadWorkflowTemplate() {
  const raw = await fs.readFile(WORKFLOW_TEMPLATE, "utf-8");
  return JSON.parse(raw);
}

function setWorkflowInput(workflow, nodeId, inputName, value) {
  if (workflow?.[nodeId]?.inputs) {
    workflow[nodeId].inputs[inputName] = value;
  }
  return workflow;
}

function patchWorkflow(workflow, params) {
  const next = structuredClone(workflow);

  setWorkflowInput(next, "1", "ckpt_name", params.ckpt_name || COMFY_CKPT_NAME);
  setWorkflowInput(next, "2", "image", params.input_image);
  setWorkflowInput(next, "4", "text", params.prompt_text);
  setWorkflowInput(next, "5", "text", params.neg_prompt || DEFAULT_NEG);
  setWorkflowInput(next, "6", "seed", Number(params.seed || 1));
  setWorkflowInput(next, "6", "steps", Number(params.steps || 24));
  setWorkflowInput(next, "6", "cfg", Number(params.cfg || 6.5));
  setWorkflowInput(next, "6", "denoise", Number(params.denoise || 0.55));
  setWorkflowInput(next, "8", "filename_prefix", params.shot_id || "shot");

  return next;
}

async function downloadUrlToBlob(url) {
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`Failed to download input image: HTTP ${res.status}`);
  }

  const buffer = Buffer.from(await res.arrayBuffer());
  const contentType = res.headers.get("content-type") || "image/jpeg";

  let ext = "jpg";
  if (contentType.includes("png")) ext = "png";
  else if (contentType.includes("webp")) ext = "webp";
  else if (contentType.includes("jpeg") || contentType.includes("jpg")) ext = "jpg";

  return { buffer, contentType, ext };
}

async function downloadUrlToDataUrl(url) {
  const { buffer, contentType } = await downloadUrlToBlob(url);
  const mime = contentType || "image/jpeg";
  return `data:${mime};base64,${buffer.toString("base64")}`;
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

  if (PARTNER_KEY) {
    body.extra_data = { api_key_comfy_org: PARTNER_KEY };
  }

  const res = await fetch(`${BASE_URL}/api/prompt`, {
    method: "POST",
    headers: getHeaders(true),
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    throw new Error(`Submit failed: HTTP ${res.status} ${await res.text()}`);
  }

  const data = await res.json();
  return data.prompt_id;
}

async function waitForCompletion(promptId, timeoutMs = SINGLE_JOB_TIMEOUT_MS) {
  const started = Date.now();

  while (Date.now() - started < timeoutMs) {
    const res = await fetch(`${BASE_URL}/api/job/${promptId}/status`, {
      headers: getHeaders(false)
    });

    if (!res.ok) {
      throw new Error(`Status failed: HTTP ${res.status}`);
    }

    const data = await res.json();
    const status = data.status;

    if (status === "completed") return true;
    if (status === "failed" || status === "cancelled") {
      throw new Error(`Job ${status}`);
    }

    await sleep(2000);
  }

  throw new Error("Job timeout");
}

async function fetchOutputs(promptId) {
  const res = await fetch(`${BASE_URL}/api/history_v2/${promptId}`, {
    headers: getHeaders(false)
  });

  if (!res.ok) {
    throw new Error(`History failed: HTTP ${res.status}`);
  }

  const data = await res.json();
  return data.outputs || {};
}

function firstImageFile(outputs) {
  for (const nodeOutputs of Object.values(outputs || {})) {
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

function getShotGroup(shot_task) {
  return (
    shot_task?.shot_group ||
    shot_task?.group ||
    shot_task?.shot_type ||
    "wildcard"
  );
}

function getShotGoal(shot_task) {
  return (
    shot_task?.shot_goal ||
    shot_task?.goal ||
    shot_task?.purpose ||
    shot_task?.brief ||
    ""
  );
}

function compilePrompt({ product_profile, style_strategy, global_style_line, shot_task }) {
  const parts = [];

  const mainProd = product_profile?.main_prod || "product";
  const accProd = product_profile?.acc_prod || "";

  parts.push(`commercial etsy product photography of ${mainProd}`);

  if (accProd && shot_task?.need_acc) {
    parts.push(`with accessory ${accProd}`);
  }

  if (getShotGoal(shot_task)) parts.push(getShotGoal(shot_task));
  if (shot_task?.scene_hint) parts.push(shot_task.scene_hint);
  if (shot_task?.light_hint) parts.push(shot_task.light_hint);
  if (shot_task?.props_hint) parts.push(shot_task.props_hint);
  if (shot_task?.ratio_rule) parts.push(`main to accessory ratio ${shot_task.ratio_rule}`);

  if (style_strategy?.scene_rule) parts.push(style_strategy.scene_rule);
  if (style_strategy?.light_rule) parts.push(style_strategy.light_rule);
  if (style_strategy?.props_rule) parts.push(style_strategy.props_rule);
  if (style_strategy?.comp_rule) parts.push(style_strategy.comp_rule);
  if (style_strategy?.texture_rule) parts.push(style_strategy.texture_rule);
  if (style_strategy?.preservation_rule) parts.push(style_strategy.preservation_rule);
  if (global_style_line) parts.push(global_style_line);

  parts.push(
    "clean background, product shape preserved, realistic materials, hasselblad-like commercial realism"
  );
  parts.push(
    "keep the product exactly 1:1 based on reference images, do not change structure, proportion, material, color, carving details, or add extra accessories"
  );

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

  const shotGroup = getShotGroup(shot_task);
  const w = groupWeight[shotGroup] || 0.8;
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
  const quotas = {
    hero: 1,
    lifestyle: 3,
    detail: 2,
    usage: 2,
    angle: 2,
    feature: 1,
    advertising: 1
  };

  const groups = {};
  for (const item of items) {
    const g = item.shot_group || "wildcard";
    groups[g] ||= [];
    groups[g].push(item);
  }

  for (const g of Object.keys(groups)) {
    groups[g].sort((a, b) => (b.final_score || 0) - (a.final_score || 0));
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
    .filter((x) => !used.has(x.shot_id))
    .sort((a, b) => (b.final_score || 0) - (a.final_score || 0));

  while (selected.length < finalCount && leftovers.length) {
    selected.push(leftovers.shift());
  }

  const selectedIds = new Set(selected.map((x) => x.shot_id));
  const dropped = items.filter((x) => !selectedIds.has(x.shot_id));

  return {
    top12_items: selected.slice(0, finalCount),
    drop4_items: dropped
  };
}

function extractImgUrls(body) {
  if (Array.isArray(body?.img_urls) && body.img_urls.length) {
    return body.img_urls;
  }

  if (Array.isArray(body?.img_inputs) && body.img_inputs.length) {
    return body.img_inputs
      .map((x) => {
        if (typeof x === "string") return x;
        return x?.url || x?.uri || x?.image_url || x?.src || "";
      })
      .filter(Boolean);
  }

  return [];
}

function pickImgUrlForShot(imgUrls, shot, index) {
  const refIndex =
    Number.isInteger(shot?.reference_index)
      ? shot.reference_index
      : Number.isInteger(shot?.img_index)
      ? shot.img_index
      : Number.isInteger(shot?.ref_index)
      ? shot.ref_index
      : index;

  const safeIndex = Math.max(0, Math.min(refIndex, imgUrls.length - 1));
  return imgUrls[safeIndex] || imgUrls[0];
}

async function generateOneInternal(body) {
  assertEnv();

  const imgUrls = extractImgUrls(body);
  if (!imgUrls.length) {
    throw new Error("img_urls is empty");
  }

  const uploaded = await uploadInputImage(imgUrls[0]);
  const inputName =
    uploaded.name ||
    uploaded.filename ||
    uploaded.image ||
    uploaded.file ||
    uploaded?.filename;

  if (!inputName) {
    throw new Error("Upload response does not contain uploaded filename");
  }

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
      ...scoreItem({
        shot_task: { shot_group: body.shot_group, priority: body.priority || 0 },
        success: false
      }),
      fail_reason: "No image output found",
      prompt_text: body.prompt_text
    };
  }

  return {
    image_url: buildProxyUrl(fileInfo),
    shot_id: body.shot_id,
    shot_group: body.shot_group,
    ...scoreItem({
      shot_task: { shot_group: body.shot_group, priority: body.priority || 0 },
      success: true
    }),
    fail_reason: "",
    prompt_text: body.prompt_text,
    file_info: fileInfo
  };
}

function buildSummaryText(items, top12_items) {
  return `已生成 ${items.length} 张候选图，筛选出 ${top12_items.length} 张结果。`;
}

async function runBatch16Internal(body, jobId) {
  const safeBody = parseMaybeJson(body);
  const shotTasks = normalizeShotTasks(safeBody.shot_tasks);
  const imgUrls = extractImgUrls(safeBody);

  if (!imgUrls.length) {
    throw new Error("img_urls or img_inputs is required");
  }

  if (!shotTasks.length) {
    throw new Error("shot_tasks is required");
  }

  const items = [];
  const baseSeed = Number(safeBody.seed || 1);

  setJob(jobId, {
    status: "running",
    progress_total: shotTasks.length,
    progress_done: 0,
    progress_text: "batch started"
  });

  for (let i = 0; i < shotTasks.length; i++) {
    const shot = shotTasks[i];

    const promptText =
      toSafeString(shot?.prompt_text) ||
      compilePrompt({
        product_profile: safeBody.product_profile || {},
        style_strategy: safeBody.style_strategy || {},
        global_style_line: safeBody.global_style_line || "",
        shot_task: shot
      });

    const negPrompt =
      toSafeString(shot?.neg_prompt) ||
      toSafeString(safeBody.neg_prompt) ||
      DEFAULT_NEG;

    const shotId = shot?.shot_id || `shot_${i + 1}`;
    const shotGroup = getShotGroup(shot);
    const shotGoal = getShotGoal(shot);
    const priority = Number(shot?.priority || 0);

    const singleImgUrl = pickImgUrlForShot(imgUrls, shot, i);

    setJob(jobId, {
      status: "running",
      progress_total: shotTasks.length,
      progress_done: i,
      progress_text: `running ${shotId}`
    });

    try {
      const item = await generateOneInternal({
        img_urls: [singleImgUrl],
        prompt_text: promptText,
        neg_prompt: negPrompt,
        shot_id: shotId,
        shot_group: shotGroup,
        steps: shot?.steps ?? safeBody.steps ?? 24,
        cfg: shot?.cfg ?? safeBody.cfg ?? 6.5,
        denoise: shot?.denoise ?? safeBody.denoise ?? 0.55,
        seed: shot?.seed ?? (baseSeed + i),
        ckpt_name: shot?.ckpt_name ?? safeBody.ckpt_name,
        priority
      });

      const score = scoreItem({
        shot_task: shot,
        success: !item.fail_reason
      });

      items.push({
        ...item,
        ...score,
        shot_goal: shotGoal,
        priority
      });
    } catch (err) {
      const score = scoreItem({ shot_task: shot, success: false });

      items.push({
        image_url: "",
        shot_id: shotId,
        shot_group: shotGroup,
        ...score,
        fail_reason: String(err?.message || err),
        prompt_text: promptText,
        shot_goal: shotGoal,
        priority
      });
    }

    setJob(jobId, {
      status: "running",
      progress_total: shotTasks.length,
      progress_done: i + 1,
      progress_text: `done ${shotId}`
    });
  }

  const { top12_items, drop4_items } = pickTop12(
    items,
    Number(safeBody.final_count || 12)
  );

  const summary_text = buildSummaryText(items, top12_items);

  return {
    items,
    top12_items,
    drop4_items,
    summary_text,
    product_profile: safeBody.product_profile || {}
  };
}

function makeJobId() {
  return `job_${Date.now()}_${randomUUID().slice(0, 8)}`;
}

function setJob(jobId, patch) {
  const prev = batch16Jobs.get(jobId) || {};
  const next = {
    ...prev,
    ...patch,
    updated_at: new Date().toISOString(),
    updated_at_ms: Date.now()
  };
  batch16Jobs.set(jobId, next);
  return next;
}

function cleanupOldJobs() {
  const now = Date.now();
  for (const [jobId, job] of batch16Jobs.entries()) {
    const updatedAtMs = Number(job?.updated_at_ms || 0);
    if (updatedAtMs && now - updatedAtMs > BATCH_JOB_TTL_MS) {
      batch16Jobs.delete(jobId);
    }
  }
}

setInterval(cleanupOldJobs, 10 * 60 * 1000).unref();

async function detectProductProfileFromImages({ img_urls, normalized_request }) {
  if (!OPENAI_API_KEY) {
    throw new Error("Missing OPENAI_API_KEY");
  }

  const safeUrls = normalizeImgUrls(img_urls).slice(0, 3);
  if (!safeUrls.length) {
    throw new Error("NO_INPUT_IMAGES");
  }

  const mainOverride = toSafeString(normalized_request?.main_prod_override);
  const accOverride = toSafeString(normalized_request?.accessory_override);
  const ratioOverride = toSafeString(normalized_request?.ratio_override);
  const scenePref = toSafeString(normalized_request?.scene_preference);

  if (mainOverride) {
    return {
      main_prod: mainOverride,
      acc_prod: accOverride,
      is_combo: !!accOverride,
      is_multiview: safeUrls.length > 1,
      prod_cat: "",
      prod_color: "",
      prod_mat: "",
      use_scene: scenePref,
      ratio_rule: ratioOverride
    };
  }

  const imageInputs = [];
  const downloadErrors = [];

  for (const url of safeUrls) {
    try {
      const dataUrl = await downloadUrlToDataUrl(url);
      imageInputs.push({
        type: "input_image",
        image_url: dataUrl,
        detail: "high"
      });
    } catch (err) {
      downloadErrors.push({
        url,
        error: String(err?.message || err)
      });
    }
  }

  console.log("[detect_profile] safeUrls.length =", safeUrls.length);
  console.log("[detect_profile] imageInputs.length =", imageInputs.length);
  console.log("[detect_profile] downloadErrors =", JSON.stringify(downloadErrors));

  if (!imageInputs.length) {
    throw new Error(`IMAGE_DOWNLOAD_FAILED: ${JSON.stringify(downloadErrors.slice(0, 2))}`);
  }

  const schema = {
    type: "object",
    additionalProperties: false,
    properties: {
      main_prod: { type: "string" },
      acc_prod: { type: "string" },
      is_combo: { type: "boolean" },
      is_multiview: { type: "boolean" },
      prod_cat: { type: "string" },
      prod_color: { type: "string" },
      prod_mat: { type: "string" },
      use_scene: { type: "string" },
      ratio_rule: { type: "string" }
    },
    required: [
      "main_prod",
      "acc_prod",
      "is_combo",
      "is_multiview",
      "prod_cat",
      "prod_color",
      "prod_mat",
      "use_scene",
      "ratio_rule"
    ]
  };

  const instruction = [
    "你是电商商品识别器。",
    "任务：根据输入图片识别商品主体，并输出严格符合 JSON Schema 的 product_profile。",
    "",
    "强制规则：",
    "1. 所有图片默认视为同一商品的多角度图、细节图或同组展示图，你必须综合判断主商品。",
    "2. 只要图片中存在清晰商品主体，就必须输出最可能、最具体的商品名称。",
    "3. main_prod 必须具体，不能写“商品”“产品”“物品”“家居用品”“饰品”等泛词。",
    "4. prod_cat 填高层级品类，例如：饰品、箱包、家居、杯具、服饰、文具、美妆工具、厨房用品。",
    "5. prod_color 填主色，prod_mat 填主材质。",
    "6. accessory_override 非空时，优先写入 acc_prod。",
    "7. ratio_override 非空时，优先写入 ratio_rule。",
    "8. scene_preference 非空时，优先写入 use_scene。",
    "9. 只有在图片打不开、没有商品主体、或主体严重遮挡无法判断时，main_prod 才允许输出“识别不确定”。",
    "10. 如果图中有清晰主体，不允许因为不够百分百确定就输出“识别不确定”。",
    "",
    `normalized_request=${JSON.stringify(normalized_request || {})}`
  ].join("\n");

  const payload = {
    model: OPENAI_VISION_MODEL,
    input: [
      {
        role: "user",
        content: [
          { type: "input_text", text: instruction },
          ...imageInputs
        ]
      }
    ],
    text: {
      format: {
        type: "json_schema",
        name: "product_profile",
        strict: true,
        schema
      }
    },
    temperature: 0,
    max_output_tokens: 500
  };

  console.log("[detect_profile] calling OpenAI vision...");

  const res = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    throw new Error(`OPENAI_VISION_FAILED: HTTP ${res.status} ${await res.text()}`);
  }

  const data = await res.json();

  const rawText =
    data.output_text ||
    data.output
      ?.flatMap((item) => item.content || [])
      .find((c) => c.type === "output_text")
      ?.text ||
    "";

  console.log("[detect_profile] rawText =", rawText);

  const parsed = extractJsonObject(rawText);
  console.log("[detect_profile] parsed =", parsed);

  const result = {
    main_prod: toSafeString(parsed.main_prod),
    acc_prod: accOverride || toSafeString(parsed.acc_prod),
    is_combo:
      typeof parsed.is_combo === "boolean"
        ? parsed.is_combo
        : !!(accOverride || toSafeString(parsed.acc_prod)),
    is_multiview:
      typeof parsed.is_multiview === "boolean"
        ? parsed.is_multiview
        : safeUrls.length > 1,
    prod_cat: toSafeString(parsed.prod_cat),
    prod_color: toSafeString(parsed.prod_color),
    prod_mat: toSafeString(parsed.prod_mat),
    use_scene: scenePref || toSafeString(parsed.use_scene),
    ratio_rule: ratioOverride || toSafeString(parsed.ratio_rule)
  };

  if (!result.main_prod) {
    throw new Error(`MODEL_EMPTY_MAIN_PROD: ${rawText}`);
  }

  if (result.main_prod === "识别不确定") {
    throw new Error(`MODEL_RETURNED_UNCERTAIN_WITH_VALID_IMAGES: ${rawText}`);
  }

  console.log("[detect_profile] final result =", result);

  return result;
}

app.get("/health", async (_req, res) => {
  res.json({
    ok: true,
    service: "coze-comfy-bridge",
    base_url: BASE_URL,
    vision_model: OPENAI_VISION_MODEL,
    workflow_template: WORKFLOW_TEMPLATE
  });
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

    if (response.status !== 302) {
      throw new Error(`view failed: HTTP ${response.status}`);
    }

    const signedUrl = response.headers.get("location");
    if (!signedUrl) {
      throw new Error("Missing signed URL");
    }

    const fileRes = await fetch(signedUrl);
    if (!fileRes.ok) {
      throw new Error(`signed fetch failed: HTTP ${fileRes.status}`);
    }

    res.setHeader(
      "Content-Type",
      fileRes.headers.get("content-type") || "application/octet-stream"
    );
    res.setHeader("Cache-Control", "public, max-age=300");

    const arrayBuffer = await fileRes.arrayBuffer();
    res.send(Buffer.from(arrayBuffer));
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err.message || err) });
  }
});

app.post("/generate_one", async (req, res) => {
  try {
    const body = parseMaybeJson(req.body);
    const data = await generateOneInternal(body);
    res.json(data);
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err.message || err) });
  }
});

app.post("/run_batch16", async (req, res) => {
  try {
    assertEnv();

    const body = parseMaybeJson(req.body);
    const shotTasks = normalizeShotTasks(body.shot_tasks);
    const imgUrls = extractImgUrls(body);

    if (!imgUrls.length) {
      return res.status(400).json({
        ok: false,
        error: "img_urls or img_inputs is required"
      });
    }

    if (!shotTasks.length) {
      return res.status(400).json({
        ok: false,
        error: "shot_tasks is required"
      });
    }

    const jobId = makeJobId();

    setJob(jobId, {
      ok: true,
      job_id: jobId,
      status: "queued",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      updated_at_ms: Date.now(),
      progress_total: shotTasks.length,
      progress_done: 0,
      progress_text: "queued",
      items: [],
      top12_items: [],
      drop4_items: [],
      summary_text: "",
      product_profile: body.product_profile || {},
      error: ""
    });

    res.status(202).json({
      ok: true,
      job_id: jobId,
      status: "queued"
    });

    setImmediate(async () => {
      try {
        setJob(jobId, {
          status: "running",
          progress_text: "starting"
        });

        const result = await runBatch16Internal(body, jobId);

        setJob(jobId, {
          ok: true,
          status: "done",
          progress_done: shotTasks.length,
          progress_total: shotTasks.length,
          progress_text: "done",
          items: result.items,
          top12_items: result.top12_items,
          drop4_items: result.drop4_items,
          summary_text: result.summary_text,
          product_profile: result.product_profile || body.product_profile || {},
          error: ""
        });
      } catch (err) {
        setJob(jobId, {
          ok: false,
          status: "failed",
          progress_text: "failed",
          error: String(err?.message || err)
        });
      }
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err.message || err) });
  }
});

app.get("/run_batch16_status", async (req, res) => {
  try {
    const jobId = toSafeString(req.query.job_id);

    if (!jobId) {
      return res.status(400).json({
        ok: false,
        error: "job_id is required"
      });
    }

    cleanupOldJobs();

    const job = batch16Jobs.get(jobId);
    if (!job) {
      return res.status(404).json({
        ok: false,
        error: "job not found",
        job_id: jobId
      });
    }

    if (job.status === "done") {
      return res.json({
        ok: true,
        job_id: jobId,
        status: "done",
        progress_done: Number(job.progress_done || 0),
        progress_total: Number(job.progress_total || 0),
        progress_text: job.progress_text || "",
        items: Array.isArray(job.items) ? job.items : [],
        top12_items: Array.isArray(job.top12_items) ? job.top12_items : [],
        drop4_items: Array.isArray(job.drop4_items) ? job.drop4_items : [],
        summary_text: job.summary_text || "",
        product_profile: job.product_profile || {}
      });
    }

    if (job.status === "failed") {
      return res.json({
        ok: false,
        job_id: jobId,
        status: "failed",
        progress_done: Number(job.progress_done || 0),
        progress_total: Number(job.progress_total || 0),
        progress_text: job.progress_text || "",
        error: job.error || "unknown error"
      });
    }

    return res.json({
      ok: true,
      job_id: jobId,
      status: job.status || "queued",
      progress_done: Number(job.progress_done || 0),
      progress_total: Number(job.progress_total || 0),
      progress_text: job.progress_text || ""
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err.message || err) });
  }
});

app.post("/detect_profile", async (req, res) => {
  try {
    const body = parseMaybeJson(req.body);
    const img_urls = normalizeImgUrls(body.img_urls);
    const normalized_request = parseMaybeJson(body.normalized_request);

    const product_profile = await detectProductProfileFromImages({
      img_urls,
      normalized_request
    });

    return res.json({ product_profile });
  } catch (err) {
    console.error("[detect_profile] error:", err);

    return res.status(500).json({
      error: "detect_profile_failed",
      detail: String(err?.message || err)
    });
  }
});

app.listen(PORT, () => {
  console.log(`Bridge listening on :${PORT}`);
  console.log(`Using workflow template: ${WORKFLOW_TEMPLATE}`);
});
