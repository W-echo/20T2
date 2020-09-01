import re

url_reg = r'[a-z]*[:.]+\S+'  # urls in text

def clean_en_text(text):
    # keeping only <space>, a-z, A-Z, 0-9, and #@$%_
    comp = re.compile('[^A-Z^a-z^0-9^#@$%_^ ]')
    return comp.sub('', text)

# remove URLS

b = '77      @JetBlue w/our #brandmance https://t.co/Bzwgp7aDVE #wemosaictogether #Mint #Love '
b = re.sub(url_reg, '', b)
y = clean_en_text(b)

print(y)