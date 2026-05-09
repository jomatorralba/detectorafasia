import { supabase } from './supabase'

export async function getPatients() {
  if (!supabase) return []
  const { data, error } = await supabase.from('patients').select('*').order('name')
  if (error) throw error
  return data ?? []
}

export async function createPatient({ name, birthDate, diagnosis, notes }) {
  const { data, error } = await supabase
    .from('patients')
    .insert({ name, birth_date: birthDate || null, diagnosis: diagnosis || '', notes: notes || '' })
    .select().single()
  if (error) throw error
  return data
}

export async function deletePatient(id) {
  const { error } = await supabase.from('patients').delete().eq('id', id)
  if (error) throw error
}

export async function getEvaluations(patientId) {
  if (!supabase) return []
  const { data, error } = await supabase
    .from('evaluations').select('*').eq('patient_id', patientId)
    .order('date', { ascending: false })
  if (error) throw error
  return data ?? []
}

export async function getEvaluationSummaries() {
  if (!supabase) return {}
  const { data, error } = await supabase
    .from('evaluations').select('patient_id, date').order('date', { ascending: false })
  if (error) throw error
  const out = {}
  for (const ev of data ?? []) {
    if (!out[ev.patient_id]) out[ev.patient_id] = { count: 0, lastDate: ev.date }
    out[ev.patient_id].count++
  }
  return out
}

export async function saveEvaluation(patientId, results, notes) {
  const { data, error } = await supabase
    .from('evaluations')
    .insert({ patient_id: patientId, results, notes: notes || '', date: new Date().toISOString() })
    .select().single()
  if (error) throw error
  return data
}
