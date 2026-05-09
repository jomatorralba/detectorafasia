-- ============================================================
-- Disartria · Neuroinn — esquema de base de datos Supabase
-- Ejecutar en: Dashboard → SQL Editor → New query → Run
-- ============================================================

-- Tabla de pacientes
CREATE TABLE IF NOT EXISTS patients (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT        NOT NULL,
  birth_date  DATE,
  diagnosis   TEXT        NOT NULL DEFAULT '',
  notes       TEXT        NOT NULL DEFAULT '',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabla de evaluaciones
CREATE TABLE IF NOT EXISTS evaluations (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id  UUID        NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  date        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  notes       TEXT        NOT NULL DEFAULT '',
  -- resultados guardados como JSON flexible:
  -- { tmf_a, tmf_s, ratio, ddk, wpm, palabra_HOLA, ... }
  results     JSONB       NOT NULL DEFAULT '{}',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Índice para consultas por paciente ordenadas por fecha
CREATE INDEX IF NOT EXISTS evaluations_patient_date
  ON evaluations (patient_id, date DESC);

-- ── Row Level Security ──────────────────────────
ALTER TABLE patients    ENABLE ROW LEVEL SECURITY;
ALTER TABLE evaluations ENABLE ROW LEVEL SECURITY;

-- Política permisiva (cualquiera puede leer y escribir)
CREATE POLICY "public_access_patients"
  ON patients FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "public_access_evaluations"
  ON evaluations FOR ALL USING (true) WITH CHECK (true);
