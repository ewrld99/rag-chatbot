UPDATE system_settings
SET value = 'llama-3.1-8b-instant,llama-3.3-70b-versatile'
WHERE key = 'generation_utility_model_order'
  AND value IN (
    'gemini-3.6-flash,llama-3.1-8b-instant,llama-3.3-70b-versatile',
    'gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant'
  );

UPDATE system_settings
SET description = 'Groq fallback priority for intent classification, query rewriting, clarification checks, and grounding verification'
WHERE key = 'generation_utility_model_order';
