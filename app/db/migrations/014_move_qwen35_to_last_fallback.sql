UPDATE system_settings
SET value = 'gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant,qwen3.5:4b'
WHERE key = 'generation_answer_model_order'
  AND value IN (
    'qwen3.5:4b,gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant',
    'gemma3:4b,qwen3.5:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant'
  );
