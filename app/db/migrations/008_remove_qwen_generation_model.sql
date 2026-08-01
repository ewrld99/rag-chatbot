UPDATE system_settings
SET value = TRIM(BOTH ',' FROM REPLACE(REPLACE(value, 'qwen/qwen3.6-27b,', ''), ',qwen/qwen3.6-27b', ''))
WHERE key IN (
  'generation_allowed_models',
  'generation_answer_model_order',
  'generation_utility_model_order'
)
  AND POSITION('qwen/qwen3.6-27b' IN value) > 0;

UPDATE system_settings
SET value = 'llama-3.3-70b-versatile'
WHERE key = 'generation_default_model'
  AND value = 'qwen/qwen3.6-27b';
