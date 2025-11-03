import re

# Use LLM with promt and fix

def clean_text(text):
    # Remove Wikipedia specific markup
    text = re.sub(r'\[\[[^]]*?\|([^]]*?)\]\]', r'\1', text)  # Remove piped links but keep display text
    text = re.sub(r'\[\[([^]]*?)\]\]', r'\1', text)  # Remove non-piped links but keep text
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove external links and references
    text = re.sub(r'\{\|[^}]*\|\}', '', text)  # Remove tables
    text = re.sub(r"'''+", '', text)  # Remove bold/italic markup
    text = re.sub(r'http\S+|www\S+', '', text) # Remove hhtp/www
    
    # Remove file and category links
    text = re.sub(r'\[\[(Պատկեր|File|Image|Կատեգորիա|Category):[^]]*\]\]', '', text)
    
    # Remove section headers
    text = re.sub(r'=+\s*(.*?)\s*=+', r'\1', text)
    
    # Remove reference tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    
    # Lower case
    text = text.lower()
    
    allowed_chars = r'|a-z|0-9|\u0561-\u0587|\s|\.|,|;|:|\?|!|\(|\)|—|-|"|\''
    text = re.sub(f'[^{allowed_chars}]', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with single newline
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text


EVAL_PROMTS = [
    "Ով է գրել Եվգենի Օնեգինին:",
    "Պուշկինի անունը",
    "Երկիրը արեգակնային համակարգի մոլորակ է, օրինակ",
    "Լավագույն անիմեն է",
    "Տիտանիկը հայտնի է նրանով",
    "Ռուսաստանը դա է",
    "Հայաստանի լավագույն ուտեստը",
    "ես հիշում եմ մի հրաշալի պահ",
    "Հիմա ծիծաղելի կատակ կլինի,"
]

# Кто написал Евгения Онегина?
# Пушкина звали
# Земля это планета солнечной системы, есть ещё например 
# Самое лучшее аниме это
# Титаник известен тем
# Россия это
# Самое лучшее блюдо Армении
# Я помню чудное мгновенье
# Сейчас будет смешная шутка, 