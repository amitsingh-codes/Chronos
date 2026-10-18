import { applyRubric, cleanForm, postForm, postJson } from './api.js'
import { AreaField, FileField, NumberField, RubricField, TextField } from './ui.jsx'
import { BatchResult, EvalResult, QuestionsResult, RagResult, SimilarityResult } from './results.jsx'

const TYPES = [['mcq', 'Multiple choice'], ['short_answer', 'Short answer'], ['long_answer', 'Long answer'], ['fill_in_the_blanks', 'Fill in the blanks']]

export const WORKFLOWS = [
  {
    id: 'q', icon: 'spark', title: 'Question studio', desc: 'Generate questions from a PDF or lesson material.',
    Result: QuestionsResult,
    Fields: () => (
      <>
        <FileField name="file" label="Lesson PDF" accept=".pdf" />
        <AreaField name="source_text" label="…or paste lesson text (50+ characters)" />
        <NumberField name="num_questions" label="Number of questions" defaultValue={10} min={1} max={50} />
        <label className="f"><span>Difficulty</span>
          <select className="i" name="difficulty" defaultValue="medium"><option>easy</option><option>medium</option><option>hard</option></select>
        </label>
        <div className="f full"><span>Question types</span>
          <div className="chk">{TYPES.map(([v, l]) => <label key={v}><input type="checkbox" name="question_types" value={v} defaultChecked />{l}</label>)}</div>
        </div>
        <NumberField name="total_marks" label="Total marks (optional)" min={1} max={10000} />
        <TextField name="subject" label="Subject (optional)" />
        <AreaField name="instructions" label="Instructions (optional)" />
      </>
    ),
    run: async form => {
      const fd = cleanForm(form), src = fd.get('source_text')
      if (fd.get('file')) { fd.delete('source_text'); return postForm('/api/questions/generate-pdf', fd) }
      if (!src) throw new Error('Upload a PDF or paste some lesson text.')
      const body = { source_text: src, num_questions: +fd.get('num_questions') || 10, difficulty: fd.get('difficulty'), question_types: fd.getAll('question_types') }
      if (fd.get('total_marks')) body.total_marks = +fd.get('total_marks')
      if (fd.get('subject')) body.subject = fd.get('subject')
      if (fd.get('instructions')) body.instructions = fd.get('instructions')
      return postJson('/api/questions/generate', body)
    },
  },
  {
    id: 'r', icon: 'book', title: 'Ask your material', desc: 'Ask questions grounded in an uploaded PDF.',
    Result: RagResult,
    Fields: () => (
      <>
        <FileField name="files" label="Study documents" multiple />
        <div>
          <TextField name="question" label="Your question" placeholder="What does the material say about…" /><br />
          <NumberField name="top_k" label="Passages to retrieve" defaultValue={5} min={1} max={20} />
        </div>
      </>
    ),
    run: async form => {
      const fd = cleanForm(form)
      if (!fd.get('files')) throw new Error('Upload at least one document.')
      if (!fd.get('question')) throw new Error('Enter a question.')
      return postForm('/api/rag/ask', fd)
    },
  },
  {
    id: 'e', icon: 'file', title: 'Answer evaluation', desc: 'Grade a student answer PDF against a question and rubric.',
    Result: EvalResult,
    Fields: () => (
      <>
        <FileField name="answer_file" label="Student answer" />
        <TextField name="subject" label="Subject (optional)" />
        <FileField name="question_file" label="Question paper (or type below)" />
        <AreaField name="question" label="Question text" />
        <FileField name="model_answer_file" label="Model answer (optional)" />
        <RubricField />
        <FileField name="rubric_file" label="Rubric file (optional, AI-extracted)" />
      </>
    ),
    run: async form => { const fd = cleanForm(form); applyRubric(fd); return postForm('/api/evaluation/evaluate', fd) },
  },
  {
    id: 'b', icon: 'check', title: 'Batch evaluation', desc: 'Evaluate multiple student answer PDFs in one workflow.',
    Result: BatchResult,
    Fields: () => (
      <>
        <FileField name="files" label="Student answers (one per student)" multiple />
        <TextField name="student_labels_raw" label="Student names (optional, comma-separated, same order)" />
        <FileField name="question_file" label="Question paper (or type below)" />
        <AreaField name="question" label="Question text" />
        <RubricField />
        <FileField name="rubric_file" label="Rubric file (optional, AI-extracted)" />
      </>
    ),
    run: async form => {
      const fd = cleanForm(form)
      const count = fd.getAll('files').length
      if (!count) throw new Error('Upload at least one answer file.')
      const raw = fd.get('student_labels_raw'); fd.delete('student_labels_raw')
      if (raw) {
        const names = raw.split(',').map(s => s.trim())
        if (names.length !== count) throw new Error('Give exactly one name per uploaded file, or leave names empty.')
        fd.append('student_labels', JSON.stringify(names))
      }
      applyRubric(fd)
      return postForm('/api/evaluation/evaluate-batch', fd)
    },
  },
  {
    id: 's', icon: 'scan', title: 'Similarity check', desc: 'Compare two or more PDF submissions for similarity.',
    Result: SimilarityResult,
    Fields: () => (
      <>
        <FileField name="files" label="Submissions (two or more)" multiple />
        <AreaField name="texts" label="…or paste one extra submission as text" />
        <AreaField name="exclude_text" label="Ignore this text (e.g. the assignment prompt)" />
        <label className="chk full"><input type="checkbox" name="semantic" defaultChecked /> Also compare meaning, not only wording</label>
      </>
    ),
    run: async form => {
      const sem = form.elements.semantic.checked
      const fd = cleanForm(form); fd.delete('semantic'); fd.append('use_semantic', sem)
      if (fd.getAll('files').length + fd.getAll('texts').length < 2) throw new Error('Add at least two submissions to compare.')
      return postForm('/api/plagiarism/check', fd)
    },
  },
]
