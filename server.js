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
    const shotTasks = Array.isArray(body.shot_tasks) ? body.shot_tasks : [];
    const imgUrls = extractImgUrls(body);

    if (!imgUrls.length) {
      return res.status(400).json({ ok: false, error: "img_urls or img_inputs is required" });
    }

    if (!shotTasks.length) {
      return res.status(400).json({ ok: false, error: "shot_tasks is required" });
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
          shot_id: shot.shot_id || `shot_${i + 1}`,
          shot_group: shot.shot_group || "wildcard",
          steps: body.steps || 24,
          cfg: body.cfg || 6.5,
          denoise: body.denoise || 0.55,
          seed: baseSeed + i,
          ckpt_name: body.ckpt_name
        });

        const score = scoreItem({
          shot_task: shot,
          success: !item.fail_reason
        });

        items.push({
          ...item,
          ...score,
          shot_goal: shot.shot_goal || "",
          priority: shot.priority || 0
        });
      } catch (err) {
        const score = scoreItem({ shot_task: shot, success: false });

        items.push({
          image_url: "",
          shot_id: shot.shot_id || `shot_${i + 1}`,
          shot_group: shot.shot_group || "wildcard",
          ...score,
          fail_reason: String(err.message || err),
          prompt_text: promptText,
          shot_goal: shot.shot_goal || "",
          priority: shot.priority || 0
        });
      }
    }

    const { top12_items, drop4_items } = pickTop12(
      items,
      Number(body.final_count || 12)
    );

    const summary_text = `已生成 ${items.length} 张候选图，筛选出 ${top12_items.length} 张结果。`;

    res.json({ items, top12_items, drop4_items, summary_text });
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
  console.log(`[boot] BASE_URL = ${BASE_URL}`);
  console.log(`[boot] WORKFLOW_TEMPLATE = ${WORKFLOW_TEMPLATE}`);
});
