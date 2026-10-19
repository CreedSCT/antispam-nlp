from pathlib import Path
import urllib.request,hashlib,json
root=Path(__file__).resolve().parent
urls={name:'https://spamassassin.apache.org/old/publiccorpus/'+name for name in ['20021010_easy_ham.tar.bz2','20021010_hard_ham.tar.bz2','20021010_spam.tar.bz2','20030228_spam_2.tar.bz2']}
urls['lingspam_public.tar.gz']='https://raw.githubusercontent.com/oreilly-japan/ml-security-jp/master/ch02/lingspam_public.tar.gz'
if __name__=='__main__':
 (root/'data/raw').mkdir(parents=True,exist_ok=True)
 for name,url in urls.items():
  path=root/'data/raw'/name
  if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
  print(name,hashlib.sha256(path.read_bytes()).hexdigest())
