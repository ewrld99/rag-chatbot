UPDATE system_settings
SET value = 'llama3.2:3b'
WHERE key = 'generation_default_model'
  AND value = 'gemma3:4b'
  AND EXISTS (
    SELECT 1
    FROM system_settings
    WHERE key = 'generation_allowed_models'
      AND value = 'llama3.2:3b,gemma3:4b,qwen3.5:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant'
  )
  AND EXISTS (
    SELECT 1
    FROM system_settings
    WHERE key = 'generation_answer_model_order'
      AND value = 'llama3.2:3b,gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant,qwen3.5:4b'
  );
