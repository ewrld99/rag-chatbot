CREATE TABLE IF NOT EXISTS retrieval_aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    term TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    category TEXT,
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_retrieval_aliases_term_lower
ON retrieval_aliases (lower(term));

CREATE INDEX IF NOT EXISTS idx_retrieval_aliases_active_term_lower
ON retrieval_aliases (is_active, lower(term));

CREATE INDEX IF NOT EXISTS idx_retrieval_aliases_aliases_gin
ON retrieval_aliases USING GIN (aliases);

WITH seed(term, aliases, category, weight) AS (
    VALUES
        ('gpa', '["grade point average", "grade point", "grading system", "course weight", "total score"]'::jsonb, 'academic', 1.0),
        ('cgpa', '["cumulative grade point average", "grade point average", "gpa"]'::jsonb, 'academic', 1.0),
        ('ca', '["continuous assessment", "course assessment"]'::jsonb, 'academic', 1.0),
        ('sr', '["sr2", "student records", "student record", "student records system", "student registration", "student information system", "student portal", "udom sr"]'::jsonb, 'portal', 1.0),
        ('sr2', '["sr", "student records", "student records system", "student portal", "udom sr"]'::jsonb, 'portal', 1.0),
        ('oas', '["online application system", "online application", "admission portal"]'::jsonb, 'portal', 1.0),
        ('tcu', '["tanzania commission for universities"]'::jsonb, 'regulatory', 1.0),
        ('nactvet', '["national council for technical and vocational education and training"]'::jsonb, 'regulatory', 1.0)
)
INSERT INTO retrieval_aliases (term, aliases, category, weight)
SELECT seed.term, seed.aliases, seed.category, seed.weight
FROM seed
WHERE NOT EXISTS (
    SELECT 1
    FROM retrieval_aliases existing
    WHERE lower(existing.term) = lower(seed.term)
);
