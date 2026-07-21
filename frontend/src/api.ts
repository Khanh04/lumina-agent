import type { RetouchResponse } from "./types";

const BASE = "/api/v1";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const jpeg = (b64: string) => `data:image/jpeg;base64,${b64}`;
export const png = (b64: string) => `data:image/png;base64,${b64}`;
export const dataUrl = (b64: string, format: "jpeg" | "png") => (format === "png" ? png(b64) : jpeg(b64));

export interface CreateResponse {
  session_id: string;
  image_base64: string;
  image_format: "jpeg" | "png";
}
export interface ImageResponse {
  status: string;
  processed_image_base64: string;
  image_format: "jpeg" | "png";
}
export interface SelectResponse {
  mask_base64: string;
}

export function createSession(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  return fetch(`${BASE}/sessions/create`, { method: "POST", body: fd }).then(json<CreateResponse>);
}

export function chat(id: string, prompt: string) {
  const fd = new FormData();
  fd.append("prompt", prompt);
  return fetch(`${BASE}/sessions/${id}/chat`, { method: "POST", body: fd }).then(json<RetouchResponse>);
}

export function revert(id: string, step: number) {
  return fetch(`${BASE}/sessions/${id}/revert`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ step }),
  }).then(json<ImageResponse>);
}

export function selectRegion(id: string, x: number, y: number) {
  return fetch(`${BASE}/sessions/${id}/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  }).then(json<SelectResponse>);
}

export function clearSelection(id: string) {
  return fetch(`${BASE}/sessions/${id}/select/clear`, { method: "POST" }).then(json);
}
