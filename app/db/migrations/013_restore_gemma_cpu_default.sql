UPDATE system_settings
SET value = 'gemma3:4b'
WHERE key = 'generation_default_model'
  AND value = 'qwen3.5:4b';

UPDATE system_settings
SET value = 'gemma3:4b,qwen3.5:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant'
WHERE key = 'generation_allowed_models'
  AND value = 'qwen3.5:4b,gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant';

UPDATE system_settings
SET value = 'gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant,qwen3.5:4b'
WHERE key = 'generation_answer_model_order'
  AND value = 'qwen3.5:4b,gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant';

UPDATE system_settings
SET value = 'qwen2.5:1.5b,gemma3:4b,llama-3.1-8b-instant,llama-3.3-70b-versatile',
    description = 'Fallback priority for intent classification, query rewriting, clarification checks, and grounding verification'
WHERE key = 'generation_utility_model_order'
  AND value = 'qwen2.5:1.5b,qwen3.5:4b,gemma3:4b,llama-3.1-8b-instant,llama-3.3-70b-versatile';
