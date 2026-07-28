import type { ModelInfo, PredictResponse } from "./types"

const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

async function parseError(res: Response) {
  try {
    const payload = await res.json()
    return payload?.detail || "خطا در ارتباط با سرور."
  } catch {
    return "خطا در ارتباط با سرور."
  }
}

export async function fetchModels(signal?: AbortSignal): Promise<ModelInfo[]> {
  const res = await fetch(`${baseUrl}/api/models`, { signal })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function predictImage(params: {
  file: File
  modelId: string
  confidence: number
  iou?: number
  signal?: AbortSignal
}): Promise<PredictResponse> {
  const form = new FormData()
  form.append("image", params.file)
  form.append("model_id", params.modelId)
  form.append("confidence", String(params.confidence))
  form.append("iou", String(params.iou ?? 0.7))
  const res = await fetch(`${baseUrl}/api/predict`, {
    method: "POST",
    body: form,
    signal: params.signal,
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export function resultImageUrl(path: string) {
  return `${baseUrl}${path}`
}

