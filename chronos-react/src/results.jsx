import { Tag, List, Warn, Raw, Bar } from './ui.jsx'

export function QuestionsResult({ data: j }) {
  return (
    <>
      <div className="stats">
        <div><b>{j.num_generated}</b>questions</div>
        <div><b>{j.total_marks}</b>marks</div>
        <div><b>{j.difficulty}</b>difficulty</div>
      </div>
      <Warn items={j.warnings} />
      {j.questions.map(q => (
        <div className="r" key={q.id}>
          <h4>Q{q.id}. {q.question}</h4>
          <p>
            <Tag>{q.type.replace(/_/g, ' ')}</Tag><Tag>{q.marks} marks</Tag><Tag>{q.difficulty}</Tag>
            {q.evidence_verified ? <Tag tone="g">evidence verified</Tag> : <Tag tone="b">evidence not verified</Tag>}
          </p>
          {q.options && (
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {q.options.map(o => <li key={o.label} style={o.label === q.correct_option ? { color: 'var(--ok)' } : null}>{o.label}. {o.text}</li>)}
            </ul>
          )}
          <p><b>Answer:</b> {q.correct_answer}</p>
          <p style={{ color: 'var(--mute)' }}>{q.explanation}</p>
          <p className="q">{q.source_excerpt}{q.source_pages?.length ? ` (p. ${q.source_pages.join(', ')})` : ''}</p>
        </div>
      ))}
      <Raw data={j} />
    </>
  )
}

export function RagResult({ data: j }) {
  const tone = j.confidence === 'high' ? 'g' : j.confidence === 'none' ? 'b' : 'a'
  return (
    <>
      <div className="r">
        <p><Tag tone={tone}>confidence: {j.confidence}</Tag>{j.grounded ? <Tag tone="g">grounded</Tag> : <Tag tone="b">not grounded</Tag>}</p>
        <p style={{ fontSize: 17, whiteSpace: 'pre-wrap' }}>{j.answer}</p>
      </div>
      <Warn items={j.warnings} />
      {j.citations.length > 0 && (
        <div className="r">
          <h4>Citations</h4>
          {j.citations.map(c => (
            <p className="q" key={c.source_id}>
              <b>{c.source_id}</b> {c.document_name}{c.pages.length ? ` · p. ${c.pages.join(', ')}` : ''} {c.quote_verified && <Tag tone="g">verified</Tag>}
              <br />{c.quote || c.excerpt}
            </p>
          ))}
        </div>
      )}
      <Raw data={j} />
    </>
  )
}

function EvalCard({ j }) {
  return (
    <>
      <div className="r">
        <div className="big">{j.total_awarded_marks}<span style={{ fontSize: 24, color: 'var(--mute)' }}> / {j.total_max_marks}</span></div>
        <p>{Math.round(j.percentage)}% <Tag tone="a">provisional</Tag>{j.needs_teacher_review && <Tag tone="b">needs teacher review</Tag>}</p>
        <p>{j.overall_feedback}</p>
        <List title="Why review" items={j.review_reasons} />
      </div>
      {j.criteria.map(c => (
        <div className="r" key={c.criterion_id}>
          <h4>{c.name} — {c.awarded_marks} / {c.max_marks}</h4>
          <Bar value={c.max_marks ? c.awarded_marks / c.max_marks : 0} />
          <p><Tag>{c.confidence} confidence</Tag><Tag tone={c.assessment_basis === 'uncertain' ? 'b' : 'g'}>{c.assessment_basis.replace('_', ' ')}</Tag></p>
          <p>{c.feedback}</p>
          {c.evidence.map((e, i) => <p className="q" key={i}>{e.text} {e.verified && <Tag tone="g">verified</Tag>}</p>)}
        </div>
      ))}
      <div className="r">
        <List title="Strengths" items={j.strengths} />
        <List title="Missing concepts" items={j.missing_concepts} />
        <List title="Next steps" items={j.improvement_suggestions} />
        <p style={{ color: 'var(--mute)', fontSize: 13 }}>{j.disclaimer}</p>
      </div>
    </>
  )
}

export const EvalResult = ({ data: j }) => <><EvalCard j={j} /><Warn items={j.warnings} /><Raw data={j} /></>

export function BatchResult({ data: j }) {
  const s = j.summary
  const tone = { failed: 'b', success: 'g' }
  return (
    <>
      <div className="stats">
        <div><b>{s.succeeded}/{s.total}</b>evaluated</div>
        <div><b>{s.average_percentage == null ? '–' : `${Math.round(s.average_percentage)}%`}</b>average</div>
        <div><b>{s.needs_teacher_review}</b>need review</div>
        <div><b>{s.failed}</b>failed</div>
      </div>
      <Warn items={j.warnings} />
      {j.results.map(r => (
        <div className="r" key={r.index}>
          <h4>{r.student_label}{r.status === 'success' && ` — ${r.result.total_awarded_marks} / ${r.result.total_max_marks}`}</h4>
          <p><Tag tone={tone[r.status] || 'a'}>{r.status}</Tag>{r.duplicate_of != null && <Tag>same as #{r.duplicate_of}</Tag>}</p>
          {r.error && <p style={{ color: 'var(--bad)' }}>{r.error.message}</p>}
          {r.result && (
            <details><summary>Full evaluation</summary>
              <div style={{ display: 'grid', gap: 14, marginTop: 12 }}><EvalCard j={r.result} /></div>
            </details>
          )}
        </div>
      ))}
      <Raw data={j} />
    </>
  )
}

export function SimilarityResult({ data: j }) {
  return (
    <>
      <div className="stats">
        <div><b>{j.summary.pairs_compared}</b>pairs compared</div>
        <div><b>{j.summary.pairs_flagged_for_review}</b>flagged</div>
        <div><b>{Math.round(j.summary.highest_lexical_containment * 100)}%</b>highest overlap</div>
      </div>
      <Warn items={j.warnings} />
      {j.pairs.map((p, i) => {
        const lex = p.lexical && Math.max(p.lexical.containment_a_in_b, p.lexical.containment_b_in_a)
        const tone = /exact|high/.test(p.assessment) ? 'b' : p.assessment === 'low_similarity' ? 'g' : 'a'
        return (
          <div className="r" key={i}>
            <h4>{p.document_a} ↔ {p.document_b}</h4>
            <p><Tag tone={tone}>{p.assessment.replace(/_/g, ' ')}</Tag>{p.reliability === 'low' && <Tag>low reliability</Tag>}</p>
            <p>{p.explanation}</p>
            {lex != null && <><p>Shared wording {Math.round(lex * 100)}%</p><Bar value={lex} /></>}
            {p.semantic?.available && <><p>Meaning similarity {Math.round(p.semantic.score * 100)}%</p><Bar value={p.semantic.score} /></>}
            {p.matches.slice(0, 5).map((m, k) => <p className="q" key={k}><Tag>{m.kind}</Tag>{m.document_a_excerpt}<br /><br />{m.document_b_excerpt}</p>)}
          </div>
        )
      })}
      {j.limitations.map((x, i) => <p key={i} style={{ color: 'var(--mute)', fontSize: 13 }}>{x}</p>)}
      <Raw data={j} />
    </>
  )
}
