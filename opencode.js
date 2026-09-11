import crypto from "crypto";

// --- Configuration & Context ---
const OPENCODE_BASE_URL = "https://opencode.ai";
const OPENCODE_UA = "opencode";
const RESPONSES_MODELS = new Set([
  "muse-spark-1.2-contributor-free",
  "muse-spark-1.3-contributor-free",
]);

// --- Helpers ---
function generateRequestId() {
  return `msg_${crypto.randomUUID().replace(/-/g, "")}`;
}

function generateSessionId() {
  return `ses_${crypto.randomUUID().replace(/-/g, "")}`;
}

function baseModelId(model) {
  return String(model || "").replace(/\([^()]+\)\s*$/, "").trim();
}

function isResponsesModel(model) {
  const base = baseModelId(model);
  return RESPONSES_MODELS.has(base) || base.includes("muse-spark");
}

// --- 1. Fetch Models ---
async function fetchModels() {
  const url = `${OPENCODE_BASE_URL}/zen/v1/models`;
  
  const response = await fetch(url, {
    headers: {
      "Authorization": "Bearer public",
      "User-Agent": OPENCODE_UA
    }
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.status} ${await response.text()}`);
  }

  const data = await response.json();
  return data.data; // Usually OpenAI-compatible APIs return the array in the 'data' field
}

// --- 2. Stream Request ---
async function streamOpenCodeRequest({
  model,
  messages,
  sessionId = generateSessionId(),
  extraBody = {}
}) {
  const body = {
    model,
    messages,
    stream: true, // Force stream to true for stdout.write
    ...extraBody
  };

  const isResponses = isResponsesModel(model);

  // Apply context transformations for Responses models
  if (isResponses) {
    if (body.max_output_tokens === undefined) {
      if (body.max_completion_tokens !== undefined) body.max_output_tokens = body.max_completion_tokens;
      else if (body.max_tokens !== undefined) body.max_output_tokens = body.max_tokens;
    }
    delete body.max_tokens;
    delete body.max_completion_tokens;

    if (body.reasoning_effort) {
      body.reasoning = {
        effort: body.reasoning_effort.toLowerCase().trim(),
        summary: "auto"
      };
      delete body.reasoning_effort;
    }
  }

  const endpoint = isResponses ? "/zen/v1/responses" : "/zen/v1/chat/completions";
  const url = `${OPENCODE_BASE_URL}${endpoint}`;

  const headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer public",
    "User-Agent": OPENCODE_UA,
    "x-opencode-client": "desktop",
    "x-opencode-session": sessionId,
    "x-opencode-request": generateRequestId(),
    "x-opencode-project": "global",
    "Accept": "text/event-stream"
  };

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    throw new Error(`OpenCode API error ${response.status}: ${await response.text()}`);
  }

  // Handle SSE (Server-Sent Events) stream
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  // Node.js native fetch body is an async iterable
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    
    // Split by newlines to process complete SSE lines
    const lines = buffer.split('\n');
    // Keep the last incomplete line in the buffer
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data:")) continue;
      
      const dataString = trimmed.replace(/^data:\s*/, "");
      
      // Stop on [DONE] marker
      if (dataString === "[DONE]") return;

      try {
        const parsed = JSON.parse(dataString);
        // Extract delta content (standard OpenAI format)
        const content = parsed.choices?.[0]?.delta?.content || "";
        if (content) {
          process.stdout.write(content);
        }
      } catch (err) {
        // Ignore JSON parse errors for incomplete chunks (handled by buffering) or comments
      }
    }
  }
}

// --- Execution ---
async function main() {
  try {
    console.log("Fetching available models...");
    const models = await fetchModels();
    
    // Display the first 3 models as an example
    console.log(`Found ${models.length} models. First 3:`);
    models.slice(0, 3).forEach(m => console.log(` - ${m.id}`));
    console.log("\n-----------------------------------\n");

    const targetModel = "mimo-v2.5-free"; // or "muse-spark-1.2-contributor-free"
    console.log(`Sending streaming request to: ${targetModel}\n`);

    await streamOpenCodeRequest({
      model: targetModel,
      messages: [
        { role: "user", content: "Write a step-by-step guide on how to make a great cup of coffee." }
      ]
    });

    console.log("\n\n[Stream Complete]");

  } catch (error) {
    console.error("\nError occurred:", error.message);
  }
}

main();
