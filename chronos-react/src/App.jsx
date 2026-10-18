import { useCallback, useEffect, useRef, useState } from 'react'
import { getBase, health, setBase, store } from './api.js'
import { Icon } from './ui.jsx'
import { WORKFLOWS } from './workflows.jsx'

const EN = {
  n1: 'Approach', n2: 'Workspace', n3: 'Tools', eb: 'A quieter way to assess',
  h1a: 'Make room for', h1b: 'better feedback.',
  sub: 'A thoughtful workspace for assessment, understanding, and the next step in every learner’s journey.',
  enter: 'Enter workspace',
  ap: 'Spend less time moving between tools and more time making sense of the work in front of you. Upload student work, explore source material, and make feedback more useful.',
  ws: 'The workspace.', pick: 'Choose a workflow to get started.',
}
const HI = {
  n1: 'दृष्टिकोण', n2: 'कार्यस्थान', n3: 'उपकरण', eb: 'आकलन का एक शांत तरीका',
  h1a: 'बेहतर फीडबैक के लिए', h1b: 'जगह बनाइए।',
  sub: 'आकलन, समझ और हर शिक्षार्थी के अगले कदम के लिए एक सोच-समझकर बनाया गया कार्यस्थान।',
  enter: 'कार्यस्थान खोलें',
  ap: 'टूल बदलने में कम समय लगाइए और सामने रखे काम को समझने में ज़्यादा। छात्रों का काम अपलोड कीजिए, स्रोत सामग्री खोजिए और फीडबैक को और उपयोगी बनाइए।',
  ws: 'कार्यस्थान।', pick: 'शुरू करने के लिए एक वर्कफ़्लो चुनें।',
}
const DICT = { en: EN, hi: HI }

const STEPS = [
  ['Start with the material', 'Upload source PDFs and work from what learners are actually studying.'],
  ['Ask better questions', 'Generate questions directly from course material and explore uploaded study documents.'],
  ['Give feedback that helps', 'Evaluate answers and compare submissions, with the API response available for review.'],
]

function Panel({ w, index }) {
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const submit = async e => {
    e.preventDefault()
    const form = e.currentTarget
    setBusy(true); setError(null); setResult(null)
    try { setResult(await w.run(form)) }
    catch (x) { setError(x) }
    finally { setBusy(false) }
  }
  return (
    <div id="panel" aria-live="polite">
      <div className="eyebrow">Workflow 0{index + 1}</div>
      <h3>{w.title}</h3>
      <p>{w.desc}</p>
      <form noValidate onSubmit={submit}>
        <w.Fields />
        <button className="go full" type="submit" disabled={busy}>
          {busy ? 'Working… large PDFs and OCR can take a minute' : `Run ${w.title.toLowerCase()}`}
        </button>
      </form>
      <div id="out">
        {error && <div className="err" role="alert">{error.message}{error.code && <> <span className="tag b">{error.code}</span></>}</div>}
        {result && <w.Result data={result} />}
      </div>
    </div>
  )
}

export default function App() {
  const [themeName, setThemeName] = useState(store.get('chronos-theme') || 'night')
  const [lang, setLang] = useState(store.get('chronos-lang') || 'en')
  const [base, setBaseState] = useState(getBase())
  const [status, setStatus] = useState({ state: 'checking' }) // checking | ok | down
  const [active, setActive] = useState('q')
  const panelRef = useRef(null)
  const t = k => DICT[lang]?.[k] ?? EN[k]

  useEffect(() => {
    document.documentElement.dataset.theme = themeName
    document.documentElement.lang = lang
    store.set('chronos-theme', themeName); store.set('chronos-lang', lang)
  }, [themeName, lang])

  const ping = useCallback(async () => {
    setStatus({ state: 'checking' })
    try { const h = await health(); setStatus({ state: 'ok', ai: h.ai.configured }) }
    catch { setStatus({ state: 'down' }) }
  }, [])
  useEffect(() => { ping() }, [ping, base])

  const ok = status.state === 'ok'
  const heroText = status.state === 'checking' ? 'Checking backend…'
    : !ok ? 'Backend not reachable'
    : status.ai ? 'Backend connected' : 'Backend connected · Gemini key missing'
  const docs = `${base}/docs`
  const idx = WORKFLOWS.findIndex(w => w.id === active)

  const choose = id => { setActive(id); requestAnimationFrame(() => panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })) }
  const commitBase = e => { setBase(e.target.value); setBaseState(getBase()) }
  const Refresh = () => <button onClick={ping} aria-label="Recheck backend"><Icon name="refresh" /></button>

  return (
    <>
      <header><div className="wrap">
        <a className="brand" href="#top">
          <span className="logo"><Icon name="clock" /></span>
          <span><b>Chronos</b><small>TEACHER WORKSPACE</small></span>
        </a>
        <nav><a href="#approach">{t('n1')}</a><a href="#workspace">{t('n2')}</a><a href="#panel">{t('n3')}</a></nav>
        <div className="tools">
          <Icon name="lang" />
          <select value={lang} onChange={e => setLang(e.target.value)} aria-label="Language">
            <option value="en">English</option><option value="hi">हिन्दी</option>
          </select>
          <button className="icon-btn" aria-label="Toggle theme" onClick={() => setThemeName(n => n === 'night' ? 'morning' : 'night')}><Icon name="moon" /></button>
          <a className="docs" href={docs} target="_blank" rel="noopener noreferrer">API docs ↗</a>
        </div>
      </div></header>

      <main id="top">
        <div className="hero">
          <svg className="hills" viewBox="0 0 1200 300" preserveAspectRatio="none" aria-hidden="true"><path d="M0 220 L200 120 L420 230 L700 90 L900 210 L1200 140 V300 H0Z" /></svg>
          <div className="wrap">
            <div className="eyebrow l">{t('eb')}</div>
            <h1><b style={{ fontWeight: 500 }}>{t('h1a')}</b><span>{t('h1b')}</span></h1>
            <p className="sub">{t('sub')}</p>
            <div className="seg" role="group" aria-label="Theme">
              <button className={themeName === 'morning' ? 'on' : ''} onClick={() => setThemeName('morning')}><b>Morning</b><small>Clear &amp; focused</small></button>
              <button className={themeName === 'night' ? 'on' : ''} onClick={() => setThemeName('night')}><b>Night</b><small>Quiet &amp; considered</small></button>
            </div><br />
            <a className="cta" href="#workspace"><span>{t('enter')}</span><Icon name="arrow" /></a>
            <div className={`stat ${ok ? 'ok' : ''}`}><span className="dot" /><span>{heroText}</span><Refresh /></div>
          </div>
          <div className="foot eyebrow"><span>Chronos / Teacher workspace</span><span>Assess with intention</span></div>
        </div>

        <section id="approach"><div className="wrap">
          <div className="eyebrow">A little more human — a lot more clarity</div>
          <div className="two">
            <h2>Good assessment<span>starts with understanding.</span></h2>
            <p>{t('ap')}</p>
          </div>
          <div className="steps">
            {STEPS.map(([h, d], i) => <div key={h}><i>0{i + 1}</i><span><b>{h}</b><small>{d}</small></span></div>)}
          </div>
        </div></section>

        <section id="workspace"><div className="wrap">
          <div className="head">
            <div>
              <div className="eyebrow">Your desk — ready when you are</div>
              <h2>{t('ws')}</h2>
              <p style={{ color: 'var(--mute)', margin: '10px 0 0' }}>{t('pick')}</p>
            </div>
            <div className={`pill ${ok ? 'ok' : ''}`}><span className="dot" /><span>{ok ? 'API connected' : status.state === 'down' ? 'API offline' : 'Checking…'}</span><Refresh /></div>
          </div>
          <div className="ws">
            <div className="wf">
              {WORKFLOWS.map((w, i) => (
                <button key={w.id} className={w.id === active ? 'on' : ''} onClick={() => choose(w.id)}>
                  <em>0{i + 1}</em><span className="ic"><Icon name={w.icon} /></span>
                  <span><b>{w.title}</b><small>{w.desc}</small></span><Icon name="arrow" />
                </button>
              ))}
            </div>
            <aside className="card">
              <span className="logo"><Icon name="shield" /></span>
              <div className="eyebrow" style={{ marginTop: 22 }}>Local development</div>
              <h3>Connected to your own backend.</h3>
              <p>Chronos sends real requests to your existing FastAPI service. File uploads are sent to routes that accept files; JSON workflows extract text from text-based PDFs first.</p>
              <div className="url">
                <span className="dot" style={{ background: ok ? 'var(--ok)' : 'var(--bad)' }} />
                <input type="text" defaultValue={base} key={base} onBlur={commitBase} onKeyDown={e => e.key === 'Enter' && e.currentTarget.blur()} aria-label="Backend URL" spellCheck={false} />
              </div>
              <a href={docs} target="_blank" rel="noopener noreferrer">Explore API endpoints ↗</a>
            </aside>
          </div>
          <div ref={panelRef} style={{ scrollMarginTop: 20 }}><Panel key={active} w={WORKFLOWS[idx]} index={idx} /></div>
        </div></section>
      </main>
    </>
  )
}
