UPDATE system_settings
SET value = TRIM(BOTH ',' FROM REPLACE(REPLACE(value, 'openai/gpt-oss-120b,', ''), ',openai/gpt-oss-120b', ''))
WHERE key IN (
  'generation_allowed_models',
  'generation_answer_model_order',
  'generation_utility_model_order'
)
  AND POSITION('openai/gpt-oss-120b' IN value) > 0;

UPDATE system_settings
SET value = 'llama-3.3-70b-versatile'
WHERE key = 'generation_default_model'
  AND value = 'openai/gpt-oss-120b';
