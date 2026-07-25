import re

def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if not text or not text.strip():
        return []

    # Preserve single and double newlines for tables, but remove excessive whitespace
    clean_text = re.sub(r'[ \t]+', ' ', text)
    # Strip markdown headings so '## Title' becomes 'Title'
    clean_text = re.sub(r'(?m)^#+\s+', '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
    
    chunks = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a natural break point if we're not at the very end
        if end < text_len:
            # 1. Try paragraph break
            break_idx = clean_text.rfind('\n\n', start, end)
            
            # 2. Try sentence break
            if break_idx == -1 or break_idx <= start + (chunk_size // 2):
                break_idx = max(clean_text.rfind('. ', start, end), 
                                clean_text.rfind('.\n', start, end))
                                
            # 3. Try newline (e.g. table rows)
            if break_idx == -1 or break_idx <= start + (chunk_size // 2):
                break_idx = clean_text.rfind('\n', start, end)
                
            # 4. Try space (word boundary)
            if break_idx == -1 or break_idx <= start + (chunk_size // 2):
                break_idx = clean_text.rfind(' ', start, end)
                
            if break_idx != -1 and break_idx > start:
                end = break_idx + 1  # Include the break character
        
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Calculate next start using overlap, snapping to the next word boundary
        start_raw = max(end - overlap, start + 1)
        space_idx = clean_text.find(' ', start_raw, end)
        start = space_idx + 1 if space_idx != -1 else start_raw

    return chunks

