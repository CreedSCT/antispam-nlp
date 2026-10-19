"""Read email text only. Never execute attachments or fetch remote content."""
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
import re, hashlib, tarfile

class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'): self.hidden+=1
    def handle_endtag(self,tag):
        if tag in ('script','style'): self.hidden=max(0,self.hidden-1)
    def handle_data(self,data):
        if not self.hidden:self.parts.append(data)

def clean(text):
    return re.sub(r'\s+',' ',text).strip()

def email_text(raw):
    msg=BytesParser(policy=policy.default).parsebytes(raw)
    subject=str(msg.get('Subject',''))
    plain=[]; html=[]
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition()=='attachment':continue
        if part.get_content_type() not in ('text/plain','text/html'):continue
        try: value=part.get_content()
        except Exception:
            value=(part.get_payload(decode=True) or b'').decode('utf-8',errors='replace')
        if not isinstance(value,str):continue
        if part.get_content_type()=='text/plain':plain.append(value)
        else:
            parser=PlainHTML();parser.feed(value);html.append(' '.join(parser.parts))
    return clean(subject+' '+ ' '.join(plain or html))

def fingerprint(text):return hashlib.sha256(clean(text).lower().encode()).hexdigest()

def read_archive(path,source):
    rows=[]
    with tarfile.open(path,'r:*') as archive:
        for member in archive:
            if not member.isfile() or member.size>5_000_000:continue
            parts=member.name.lower().split('/')
            if 'cmds' in parts or member.name.endswith(('README','readme')):continue
            if source=='LingSpam' and 'bare' not in parts:continue
            if source=='LingSpam':label=int(parts[-1].startswith('spmsg'))
            elif any(x=='spam' or x.startswith('spam_') for x in parts):label=1
            elif any(x in ('ham','easy_ham','easy_ham_2','hard_ham') for x in parts):label=0
            else:continue
            text=email_text(archive.extractfile(member).read())
            if text:rows.append({'text':text,'label':label,'source':source,'file':member.name,'hash':fingerprint(text)})
    return rows
