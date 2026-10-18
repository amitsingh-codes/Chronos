import { useState } from 'react'

const PATHS = {
  spark: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z',
  book: 'M3 5c3-1 6-1 9 1 3-2 6-2 9-1v14c-3-1-6-1-9 1-3-2-6-2-9-1zM12 6v14',
  file: 'M6 3h8l5 5v13H6zM14 3v5h5M9 14l2 2 4-4',
  check: 'M5 12l5 5 9-10',
  scan: 'M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M8 9h8M8 12h8M8 15h5',
  arrow: 'M7 7l10 10M17 8v9H8',
  refresh: 'M20 12a8 8 0 1 1-3-6.2M20 4v5h-5',
  moon: 'M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10z',
  shield: 'M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6zM9 12l2 2 4-4',
  clock: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3 2',
  lang: 'M4 5h9M8 3v2M6 5c1 4 4 7 7 8M12 5c-1 4-4 7-8 9M14 20l4-9 4 9M15.5 17h5',
}
export const Icon = ({ name }) => <svg viewBox="0 0 24 24" aria-hidden="true"><path d={PATHS[name]} /></svg>

/* ---------- form fields ---------- */
export function FileField({ name, label, multiple, accept = '.pdf,.txt' }) {
  const [names, setNames] = useState('')
  return (
    <label className="f">
      <span>{label}</span>
      <span className="drop">
        <input type="file" name={name} multiple={multiple} accept={accept}
          onChange={e => setNames([...e.target.files].map(f => f.name).join(', '))} />
        <b>{names || `Choose ${multiple ? 'files' : 'a file'}`}</b>
        {accept.includes('txt') ? 'PDF or .txt' : 'PDF'}
      </span>
    </label>
  )
}
export const TextField = ({ name, label, placeholder }) =>
  <label className="f"><span>{label}</span><input type="text" name={name} placeholder={placeholder} /></label>
export const NumberField = ({ name, label, defaultValue, min, max }) =>
  <label className="f"><span>{label}</span><input type="number" name={name} defaultValue={defaultValue} min={min} max={max} /></label>
export const AreaField = ({ name, label, placeholder }) =>
  <label className="f full"><span>{label}</span><textarea name={name} placeholder={placeholder} /></label>

export function RubricField() {
  const blank = { name: '', description: '', max_marks: '' }
  const [rows, setRows] = useState([blank])
  const set = (i, k, v) => setRows(rs => rs.map((r, j) => (j === i ? { ...r, [k]: v } : r)))
  return (
    <div className="f full">
      <span>Rubric — or upload a rubric file below</span>
      <div className="rub">
        {rows.map((r, i) => (
          <div key={i}>
            <input type="text" placeholder="Criterion" aria-label="Criterion" value={r.name} onChange={e => set(i, 'name', e.target.value)} />
            <input type="text" placeholder="What earns the marks" aria-label="Description" value={r.description} onChange={e => set(i, 'description', e.target.value)} />
            <input type="number" min="0.5" step="0.5" placeholder="Marks" aria-label="Marks" value={r.max_marks} onChange={e => set(i, 'max_marks', e.target.value)} />
            <button type="button" className="mini" aria-label="Remove criterion" onClick={() => setRows(rs => rs.filter((_, j) => j !== i))}>×</button>
          </div>
        ))}
      </div>
      <button type="button" className="mini" style={{ justifySelf: 'start' }} onClick={() => setRows(rs => [...rs, blank])}>Add criterion</button>
      <input type="hidden" name="rubric_json" value={JSON.stringify(rows)} />
    </div>
  )
}

/* ---------- result helpers ---------- */
export const Tag = ({ children, tone = '' }) => <span className={`tag ${tone}`}>{children}</span>
export const List = ({ title, items }) => items?.length
  ? <><p><b>{title}</b></p><ul>{items.map((x, i) => <li key={i}>{x}</li>)}</ul></> : null
export const Warn = ({ items }) => items?.length
  ? <div className="r">{items.map((x, i) => <p key={i}><Tag tone="a">note</Tag>{x}</p>)}</div> : null
export const Raw = ({ data }) =>
  <details><summary>API response</summary><pre>{JSON.stringify(data, null, 2)}</pre></details>
export const Bar = ({ value }) => <div className="bar"><i style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} /></div>
