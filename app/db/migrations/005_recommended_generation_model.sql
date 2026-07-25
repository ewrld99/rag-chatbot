UPDATE system_settings
SET value = 'llama-3.3-70b-versatile,openai/gpt-oss-120b,qwen/qwen3.6-27b,openai/gpt-oss-20b,llama-3.1-8b-instant'
WHERE key = 'generation_allowed_models'
  AND value = 'openai/gpt-oss-120b,qwen/qwen3.6-27b,openai/gpt-oss-20b,llama-3.1-8b-instant';

UPDATE system_settings
SET value = 'llama-3.3-70b-versatile,openai/gpt-oss-120b,qwen/qwen3.6-27b,openai/gpt-oss-20b,llama-3.1-8b-instant'
WHERE key = 'generation_answer_model_order'
  AND value = 'openai/gpt-oss-120b,qwen/qwen3.6-27b,openai/gpt-oss-20b,llama-3.1-8b-instant';

UPDATE system_settings
SET value = 'openai/gpt-oss-20b,llama-3.1-8b-instant,llama-3.3-70b-versatile,qwen/qwen3.6-27b,openai/gpt-oss-120b'
WHERE key = 'generation_utility_model_order'
  AND value = 'openai/gpt-oss-20b,llama-3.1-8b-instant,qwen/qwen3.6-27b,openai/gpt-oss-120b';

UPDATE system_settings
SET value = 'llama-3.3-70b-versatile'
WHERE key = 'generation_default_model'
  AND value = 'openai/gpt-oss-120b';
