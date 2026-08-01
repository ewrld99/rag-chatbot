UPDATE system_settings
SET value = 'qwen3.5:4b'
WHERE key = 'generation_default_model'
  AND value IN (
    'gemma3:4b',
    'gemini-3.6-flash',
    'llama-3.3-70b-versatile',
    'openai/gpt-oss-120b',
    'qwen/qwen3.6-27b'
  );

UPDATE system_settings
SET value = 'qwen3.5:4b,gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant'
WHERE key IN ('generation_allowed_models', 'generation_answer_model_order')
  AND value IN (
    'gemma3:4b,gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant',
    'gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant',
    'llama-3.3-70b-versatile,openai/gpt-oss-120b',
    'llama-3.3-70b-versatile,qwen/qwen3.6-27b,llama-3.1-8b-instant'
  );

UPDATE system_settings
SET value = 'qwen2.5:1.5b,qwen3.5:4b,gemma3:4b,llama-3.1-8b-instant,llama-3.3-70b-versatile',
    description = 'Fallback priority for intent classification, query rewriting, clarification checks, and grounding verification'
WHERE key = 'generation_utility_model_order'
  AND value IN (
    'qwen2.5:1.5b,gemma3:4b,llama-3.1-8b-instant,llama-3.3-70b-versatile',
    'gemma3:4b,llama-3.1-8b-instant,llama-3.3-70b-versatile',
    'llama-3.1-8b-instant,llama-3.3-70b-versatile',
    'llama-3.1-8b-instant,qwen/qwen3.6-27b',
    'gemini-3.6-flash,llama-3.1-8b-instant,llama-3.3-70b-versatile',
    'gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant'
  );
