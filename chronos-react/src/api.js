const KEY = 'chronos-base'
const DEFAULT = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export const store = {
  get(k) { try { return localStorage.getItem(k) } catch { return null } },
  set(k, v) { try { localStorage.setItem(k, v) } catch { /* private mode */ } },
}
export const getBase = () => (store.get(KEY) || DEFAULT).trim().replace(/\/+$/, '')
export const setBase = (v) => store.set(KEY, v.trim())

export async function call(path, opts) {
  let r
  try { r = await fetch(getBase() + path, opts) }
  catch { throw new Error(`Cannot reach ${getBase()}. Is the backend running, and is this page's origin listed in CORS_ORIGINS?`) }
  let body = null
  try { body = await r.json() } catch { /* non-JSON body */ }
  if (!r.ok) {
    const e = body?.error, d = body?.detail
    const msg = e?.message
      || (Array.isArray(d) ? d.map(x => `${(x.loc || []).slice(1).join('.')}: ${x.msg}`).join('; ') : d)
      || `Request failed (${r.status})`
    throw Object.assign(new Error(msg), { code: e?.code, status: r.status })
  }
  return body
}
export const health = () => call('/health')
export const postForm = (path, fd) => call(path, { method: 'POST', body: fd })
export const postJson = (path, o) => call(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(o) })

// FormData without empty fields / empty file inputs
export function cleanForm(form) {
  const out = new FormData()
  for (const [k, v] of new FormData(form)) {
    if (v instanceof File ? !v.size : !String(v).trim()) continue
    out.append(k, v)
  }
  return out
}
// Move the rubric builder's rows into the `rubric` field (skipped when a rubric file is uploaded instead)
export function applyRubric(fd) {
  const rows = JSON.parse(fd.get('rubric_json') || '[]')
    .map(r => ({ name: r.name.trim(), description: r.description.trim(), max_marks: parseFloat(r.max_marks) }))
    .filter(r => r.name || r.max_marks)
  fd.delete('rubric_json')
  if (rows.some(r => !r.name || !(r.max_marks > 0))) throw new Error('Each rubric criterion needs a name and marks above 0.')
  if (rows.length) fd.append('rubric', JSON.stringify(rows))
}
