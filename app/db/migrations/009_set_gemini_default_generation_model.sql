UPDATE system_settings
SET value = 'gemini-3.6-flash'
WHERE key = 'generation_default_model';

UPDATE system_settings
SET value = TRIM(BOTH ',' FROM REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(value,
  'openai/gpt-oss-120b,', ''),
  ',openai/gpt-oss-120b', ''),
  'openai/gpt-oss-20b,', ''),
  ',openai/gpt-oss-20b', ''),
  'qwen/qwen3.6-27b,', ''),
  ',qwen/qwen3.6-27b', ''))
WHERE key IN (
  'generation_allowed_models',
  'generation_answer_model_order',
  'generation_utility_model_order'
);

UPDATE system_settings
SET value = CASE
  WHEN value = '' THEN 'gemini-3.6-flash'
  WHEN POSITION('gemini-3.6-flash' IN value) > 0 THEN value
  ELSE 'gemini-3.6-flash,' || value
END
WHERE key IN (
  'generation_allowed_models',
  'generation_answer_model_order',
  'generation_utility_model_order'
);
