import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL ?? 'https://zypbkgtmzmypmntsehbm.supabase.co'
const key = import.meta.env.VITE_SUPABASE_ANON_KEY ?? 'sb_publishable_5GiDuUZa7htfaGhnT3WOzQ_MzDOHYjJ'

export const supabase = createClient(url, key)
export const isConfigured = true
